import torch
from pico_vllm.config import config
from pico_vllm.models.base import ModelArchitecture


class KVCache:
    def __init__(
        self, *, architecture: ModelArchitecture | None = None, num_blocks=None,
        block_size=None, device=None, dtype=None,
    ):
            self.architecture = architecture or ModelArchitecture(
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                hidden_size=config.embed_dim,
                num_kv_heads=config.num_kv_heads,
            )
            num_blocks = config.num_blocks if num_blocks is None else num_blocks
            block_size = config.num_tokens_per_block if block_size is None else block_size
            self.key_cache = torch.zeros(
                self.architecture.num_layers,
                num_blocks,
                self.architecture.num_kv_heads,
                block_size,
                self.architecture.head_dim,
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
