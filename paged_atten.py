import torch

from config import config
from sequence import Sequence
from kv_cache import KVCache

def gather_kv(kv_cache: KVCache, layer, seq : Sequence):
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

    # Concatenate along the token dimension
    key_vectors = torch.cat(key_vectors_list, dim=1)  # Concatenate along the token dimension
    value_vectors = torch.cat(value_vectors_list, dim=1)

    # Slice to the actual sequence length
    key_vectors = key_vectors[ :, :seq.seq_len, :]
    value_vectors = value_vectors[ :, :seq.seq_len, :]

    return key_vectors, value_vectors
