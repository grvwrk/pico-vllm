import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def test_kv_cache_matches_hf():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer("Hello world", return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs, use_cache=True)

    first_layer_key = output.past_key_values.layers[0].keys      # (1, num_kv_heads, seq_len, head_dim)
    first_layer_value = output.past_key_values.layers[0].values  # (1, num_kv_heads, seq_len, head_dim)

    seq_len = first_layer_key.shape[2]

    bm = BlockManager(num_block=8)  # small pool, only need 1 block for this test
    block_index = bm.allocate_block("test_seq")

    cache = KVCache()

    for t in range(seq_len):
        token_key = first_layer_key[0, :, t, :]
        token_value = first_layer_value[0, :, t, :]
        cache.write(
            layer=0,
            block_index=block_index,
            slot_in_block=t,
            key_vector=token_key,
            value_vector=token_value,
        )

    read_key, read_value = cache.read(layer=0, block_index=block_index)

    # only compare the first seq_len slots, the rest of the block is still zeros
    assert torch.allclose(read_key[:, :seq_len, :], first_layer_key[0, :, :seq_len, :])
    assert torch.allclose(read_value[:, :seq_len, :], first_layer_value[0, :, :seq_len, :])

    print("Cache write/read matches HF's real KV cache exactly.")


if __name__ == "__main__":
    test_kv_cache_matches_hf()
