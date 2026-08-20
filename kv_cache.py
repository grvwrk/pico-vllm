import torch
from config import config

key_cache = torch.zeros(
    config.num_layers,
    config.num_block,
    config.num_kv_heads,
    config.num_tokens_per_block, 
    config.head_dim
)
value_cache = torch.zeros_like(key_cache)

def write_kv(key_cache, value_cache, layer, block_index, slot_in_block, key_vector, value_vector):
    """
    Writes one token's key and value vectors into the cache
    at the given layer, block, and slot.
    """
    if slot_in_block >= config.num_tokens_per_block or slot_in_block <0:
        raise ValueError(f"slot_in_block {slot_in_block} exceeds num_tokens_per_block {config.num_tokens_per_block}")
    key_cache[layer, block_index, :, slot_in_block, :] = key_vector
    value_cache[layer, block_index, :, slot_in_block, :] = value_vector

def read_kv(key_cache, value_cache, layer, block_index):
    """
    Reads the key and value vectors for a given layer and block.
    Returns a tuple of (key_vectors, value_vectors).
    """
    return key_cache[layer, block_index], value_cache[layer, block_index]