from pico_vllm.config import config
from pico_vllm.core.kv_cache import KVCache

class Sequence:
    def __init__(self, seq_id, initial_block_index):
        self.seq_id = seq_id
        self.block_indices = [initial_block_index]  # List of block indices allocated for this sequence
        self.seq_len = 0

    def append_token(self, key_vector, value_vector, kv_cache: KVCache, layer, new_block_index=None):
        if self.seq_len > 0 and self.seq_len % kv_cache.block_size == 0:
            # A block just filled up completely — need a new one
            if new_block_index is None:
                raise ValueError("New block index must be provided when allocating a new block.")
            self.block_indices.append(new_block_index)

        # Calculate the current block index and slot within the block
        current_block_index = self.block_indices[-1]
        slot_in_block = self.seq_len % kv_cache.block_size

        # Write the key and value vectors to the KV cache
        kv_cache.write(
            layer=layer,
            block_index=current_block_index,
            slot_in_block=slot_in_block,
            key_vector=key_vector,
            value_vector=value_vector,
        )

        # Increment the sequence length
        self.seq_len += 1
