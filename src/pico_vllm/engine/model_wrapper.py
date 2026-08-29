import torch

from pico_vllm.config import config
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.paged_attention import paged_attention
from pico_vllm.core.sequence import Sequence


def apply_rope(x, model, seq_len, position_ids=None):
    """
    Applies rotary positional embedding to a query or key tensor.
    x: (num_heads, seq_len, head_dim)
    """
    if position_ids is None:
        position_ids = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)

    dummy_hidden = torch.zeros(1, seq_len, model.config.hidden_size, dtype=x.dtype)
    cos, sin = model.model.rotary_emb(dummy_hidden, position_ids)
    # cos, sin: (1, seq_len, head_dim)

    def rotate_half(t):
        t1, t2 = t.chunk(2, dim=-1)
        return torch.cat((-t2, t1), dim=-1)

    cos = cos.squeeze(0).unsqueeze(0)  # (1, seq_len, head_dim), broadcasts across heads
    sin = sin.squeeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)


def forward_one_sequence(model, kv_cache: KVCache, block_manager, seq: Sequence,
                          input_ids, is_prefill: bool):
    """
    Runs a full forward pass for ONE sequence through every transformer layer,
    using this project's own paged_attention + KVCache in place of the model's
    built-in attention, but reusing all of the model's real pretrained weights
    (layernorms, projections, MLP) unchanged.

    input_ids: (seq_len,) for prefill, or (1,) for a single decode step
    Returns: logits for the last position, shape (vocab_size,)
    """
    seq_len = input_ids.shape[0]
    position_offset = seq.seq_len  # how many tokens already exist in the cache

    # ---- Embedding lookup ----
    hidden_states = model.model.embed_tokens(input_ids)  # (seq_len, hidden_size)
    hidden_states = hidden_states.unsqueeze(0)  # (1, seq_len, hidden_size) — HF layernorm/MLP modules expect a batch dim

    position_ids = torch.arange(position_offset, position_offset + seq_len).unsqueeze(0)  # (1, seq_len)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden_states

        # ---- Input layernorm ----
        normed = layer.input_layernorm(hidden_states)

        # ---- Q/K/V projections ----
        q = layer.self_attn.q_proj(normed)  # (1, seq_len, num_heads * head_dim)
        k = layer.self_attn.k_proj(normed)  # (1, seq_len, num_kv_heads * head_dim)
        v = layer.self_attn.v_proj(normed)

        q = q.view(seq_len, config.num_heads, config.head_dim).transpose(0, 1)      # (num_heads, seq_len, head_dim)
        k = k.view(seq_len, config.num_kv_heads, config.head_dim).transpose(0, 1)   # (num_kv_heads, seq_len, head_dim)
        v = v.view(seq_len, config.num_kv_heads, config.head_dim).transpose(0, 1)

        # ---- RoPE on Q and K (must match position_ids, not just 0..seq_len,
        # since a decode step's position is offset by however many tokens
        # are already cached) ----
        q = apply_rope(q, model, seq_len, position_ids=position_ids)
        k = apply_rope(k, model, seq_len, position_ids=position_ids)

        # ---- Write this layer's K/V into the cache, one token at a time ----
        for t in range(seq_len):
            token_key = k[:, t, :]
            token_value = v[:, t, :]

            new_block_index = None
            if layer_idx == 0:
                # block allocation only needs to happen once per token, not once per layer,
                # since all layers share the same block_indices list on the Sequence
                if seq.seq_len > 0 and seq.seq_len % kv_cache.block_size == 0:
                    new_block_index = block_manager.allocate_block(seq.seq_id)

            if layer_idx == 0:
                seq.append_token(token_key, token_value, kv_cache, layer=layer_idx,
                                  new_block_index=new_block_index)
            else:
                # seq.seq_len / block_indices already advanced during layer 0's pass;
                # just write this layer's data into the same slot.
                slot_in_block = (seq.seq_len - seq_len + t) % kv_cache.block_size
                block_idx_for_token = seq.block_indices[(seq.seq_len - seq_len + t) // kv_cache.block_size]
                kv_cache.write(layer_idx, block_idx_for_token, slot_in_block, token_key, token_value)

        # ---- Attention using our paged_attention ----
        attn_out = paged_attention(q, kv_cache, layer=layer_idx, seq=seq, is_prefill=is_prefill)
        # attn_out: (num_heads, seq_len, head_dim)

        attn_out = attn_out.transpose(0, 1).reshape(1, seq_len, config.num_heads * config.head_dim)
        attn_out = layer.self_attn.o_proj(attn_out)

        hidden_states = residual + attn_out

        # ---- MLP block ----
        residual = hidden_states
        normed = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(normed)
        hidden_states = residual + mlp_out

    # ---- Final layernorm + LM head ----
    hidden_states = model.model.norm(hidden_states)
    logits = model.lm_head(hidden_states)  # (1, seq_len, vocab_size)

    return logits[0, -1, :]  # logits for the last position only
