import torch
from pico_vllm.config import config


class KVCache:
    def __init__(self, *, num_blocks=None, block_size=None, device=None, dtype=None):
            num_blocks = config.num_blocks if num_blocks is None else num_blocks
            block_size = config.num_tokens_per_block if block_size is None else block_size
            self.key_cache = torch.zeros(
                config.num_layers,
                num_blocks,
                config.num_kv_heads,
                block_size,
                config.head_dim,
                device=device,
                dtype=dtype,
            )
            self.value_cache = torch.zeros_like(self.key_cache)
            self.block_size = block_size

    def write(self, layer, block_index, slot_in_block, key_vector, value_vector):
        """
        Writes one token's key and value vectors into the cache
        at the given layer, block, and slot.
        """
        if slot_in_block >= config.num_tokens_per_block or slot_in_block < 0:
            raise ValueError(
                f"slot_in_block {slot_in_block} out of range "
                f"(0 to {config.num_tokens_per_block - 1})"
            )
        self.key_cache[layer, block_index, :, slot_in_block, :] = key_vector
        self.value_cache[layer, block_index, :, slot_in_block, :] = value_vector

    def read(self, layer, block_index):
        """
        Reads the key and value vectors for a given layer and block.
        Returns a tuple of (key_vectors, value_vectors).
        """
        return self.key_cache[layer, block_index], self.value_cache[layer, block_index]
