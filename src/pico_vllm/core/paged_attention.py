import torch
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence

def gather_kv(kv_cache: KVCache, layer, seq: Sequence):
    """
    Gathers the key and value vectors for a given sequence across all its allocated blocks.
    Returns a tuple of (key_vectors, value_vectors) with shape:
    (num_kv_heads, seq_len, head_dim)
    """
    key_vectors_list = []
    value_vectors_list = []

    for block_index in seq.block_indices:
        key_block, value_block = kv_cache.read(layer, block_index)
        key_vectors_list.append(key_block)
        value_vectors_list.append(value_block)

    key_vectors = torch.cat(key_vectors_list, dim=1)
    value_vectors = torch.cat(value_vectors_list, dim=1)

    key_vectors = key_vectors[:, :seq.seq_len, :]
    value_vectors = value_vectors[:, :seq.seq_len, :]

    return key_vectors, value_vectors


def paged_attention(query, kv_cache: KVCache, layer, seq: Sequence, is_prefill: bool):
    """
    Runs scaled dot-product attention for a sequence using its paged KV-cache.

    Gathers the sequence's real (non-padded) key/value data from its allocated
    blocks, expands the KV heads to match the number of query heads (GQA),
    and computes attention. Uses a causal mask during prefill (query attends
    over the full prompt) and no mask during decode (a single new token's
    query attending over an already-causal cache has nothing future to hide).

    Args:
        query: (num_query_heads, q_len, head_dim)
        kv_cache: the KVCache holding this sequence's stored key/value data
        layer: which transformer layer's cache to read from
        seq: the Sequence whose block_indices and seq_len define what to gather
        is_prefill: True during prompt processing, False during token-by-token decode

    Returns:
        output: (num_query_heads, q_len, head_dim)
    """
    gathered_keys, gathered_values = gather_kv(kv_cache, layer, seq)

    group_size = kv_cache.architecture.num_heads // kv_cache.architecture.num_kv_heads
    keys_expanded = gathered_keys.repeat_interleave(group_size, dim=0)
    values_expanded = gathered_values.repeat_interleave(group_size, dim=0)

    query = query.unsqueeze(0)
    keys_expanded = keys_expanded.unsqueeze(0)
    values_expanded = values_expanded.unsqueeze(0)

    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        keys_expanded,
        values_expanded,
        is_causal=is_prefill,
    )
    output = output.squeeze(0)
    return output
