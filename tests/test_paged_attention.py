import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.config import config
from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.paged_attention import paged_attention
from pico_vllm.core.sequence import Sequence

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER = 0

# A longer sentence, chosen to comfortably cross the 16-token block boundary
TEXT = (
    "The quick brown fox jumps over the lazy dog while the sun sets "
    "slowly behind the distant mountains, painting the sky in brilliant "
    "shades of orange and red."
)


def apply_rope(query, model, seq_len):
    """
    Applies the same rotary positional embedding to our raw captured query
    that HF applies internally before attention (and which is already baked
    into the cached keys we're comparing against, since HF caches keys
    post-RoPE).
    """
    position_ids = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)

    # rotary_emb only needs this dummy tensor for dtype/device inference,
    # not its actual values.
    dummy_hidden = torch.zeros(1, seq_len, model.config.hidden_size, dtype=query.dtype)
    cos, sin = model.model.rotary_emb(dummy_hidden, position_ids)
    # cos, sin: (1, seq_len, head_dim)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.squeeze(0).unsqueeze(0)  # (1, seq_len, head_dim), broadcasts across heads
    sin = sin.squeeze(0).unsqueeze(0)

    query_rotated = (query * cos) + (rotate_half(query) * sin)
    return query_rotated


def test_paged_attention_matches_hf():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # ---- Set up hooks to capture the real query, and the real pre-o_proj attention output ----
    captured = {}

    def capture_query(module, inp, output):
        captured["query"] = output  # raw q_proj output, before reshaping into heads

    def capture_attn_output(module, inp, output):
        captured["attn_output"] = inp[0]  # o_proj's INPUT is the real attention output

    q_proj = model.model.layers[LAYER].self_attn.q_proj
    o_proj = model.model.layers[LAYER].self_attn.o_proj

    q_handle = q_proj.register_forward_hook(capture_query)
    o_handle = o_proj.register_forward_hook(capture_attn_output)

    inputs = tokenizer(TEXT, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]
    assert seq_len > config.num_tokens_per_block, (
        f"Sentence only tokenized to {seq_len} tokens, need more than "
        f"{config.num_tokens_per_block} to actually cross a block boundary"
    )

    with torch.no_grad():
        output = model(**inputs, use_cache=True)

    q_handle.remove()
    o_handle.remove()

    # ---- Reshape the captured raw query into (num_query_heads, seq_len, head_dim) ----
    raw_query = captured["query"]  # shape (1, seq_len, num_query_heads * head_dim)
    query = raw_query.view(seq_len, config.num_heads, config.head_dim).transpose(0, 1)
    # query: (num_query_heads, seq_len, head_dim)

    # Apply RoPE to the query, matching what HF applies internally before
    # attention. The cached keys from past_key_values already have this
    # baked in, so without this step query and key would be misaligned.
    query = apply_rope(query, model, seq_len)

    # ---- Reshape the captured ground-truth attention output the same way, for comparison ----
    ground_truth = captured["attn_output"]  # shape (1, seq_len, num_query_heads * head_dim)
    ground_truth = ground_truth.view(seq_len, config.num_heads, config.head_dim).transpose(0, 1)
    # ground_truth: (num_query_heads, seq_len, head_dim)

    # ---- Write the real K/V (from HF's own cache) into our KVCache via Sequence ----
    first_layer_key = output.past_key_values.layers[LAYER].keys      # (1, num_kv_heads, seq_len, head_dim)
    first_layer_value = output.past_key_values.layers[LAYER].values

    bm = BlockManager(num_block=8)
    kv_cache = KVCache()

    initial_block = bm.allocate_block("test_seq")
    seq = Sequence(seq_id="test_seq", initial_block_index=initial_block)

    for t in range(seq_len):
        token_key = first_layer_key[0, :, t, :]
        token_value = first_layer_value[0, :, t, :]

        # if the current block is about to fill up, pre-allocate a new one
        new_block_index = None
        if seq.seq_len > 0 and seq.seq_len % kv_cache.block_size == 0:
            new_block_index = bm.allocate_block("test_seq")

        seq.append_token(token_key, token_value, kv_cache, layer=LAYER, new_block_index=new_block_index)

    assert seq.seq_len == seq_len
    assert len(seq.block_indices) > 1, "Sequence should have crossed a block boundary"

    # ---- Run our paged attention using the real (RoPE-applied) query and our reconstructed K/V ----
    our_output = paged_attention(query, kv_cache, layer=LAYER, seq=seq, is_prefill=True)
    # our_output: (num_query_heads, seq_len, head_dim)

    # ---- Compare ----
    assert torch.allclose(our_output, ground_truth, atol=1e-2), (
        "paged_attention output does not match real HF attention output"
    )

    print("paged_attention output matches real HF attention output across multiple blocks.")


if __name__ == "__main__":
    test_paged_attention_matches_hf()
