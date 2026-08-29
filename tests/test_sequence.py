import torch

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence


def make_fake_kv(kv_cache):
    """Helper: build a random key/value vector matching kv_cache's shape."""
    key = torch.randn(kv_cache.key_cache.shape[2], kv_cache.key_cache.shape[4], dtype=torch.bfloat16)
    value = torch.randn(kv_cache.value_cache.shape[2], kv_cache.value_cache.shape[4], dtype=torch.bfloat16)
    return key, value


def test_append_token_within_first_block():
    bm = BlockManager(num_block=8)
    kv_cache = KVCache()

    first_block = bm.allocate_block("seq_A")
    seq = Sequence(seq_id="seq_A", initial_block_index=first_block)

    # Append a few tokens, fewer than block_size, so no new block should be needed
    for _ in range(5):
        key, value = make_fake_kv(kv_cache)
        seq.append_token(key, value, kv_cache, layer=0)

    assert seq.seq_len == 5
    assert seq.block_indices == [first_block]  # still just the one block


def test_append_token_crosses_block_boundary():
    bm = BlockManager(num_block=8)
    kv_cache = KVCache()
    block_size = kv_cache.block_size

    first_block = bm.allocate_block("seq_B")
    seq = Sequence(seq_id="seq_B", initial_block_index=first_block)

    # Fill the first block exactly
    for _ in range(block_size):
        key, value = make_fake_kv(kv_cache)
        seq.append_token(key, value, kv_cache, layer=0)

    assert seq.seq_len == block_size
    assert seq.block_indices == [first_block]  # still one block, exactly full

    # The next token should require a new block
    key, value = make_fake_kv(kv_cache)
    try:
        seq.append_token(key, value, kv_cache, layer=0)
        assert False, "Expected ValueError when block is full and no new_block_index given"
    except ValueError:
        pass

    # Now retry, correctly providing a new block
    second_block = bm.allocate_block("seq_B")
    seq.append_token(key, value, kv_cache, layer=0, new_block_index=second_block)

    assert seq.seq_len == block_size + 1
    assert seq.block_indices == [first_block, second_block]


def test_written_data_is_retrievable():
    bm = BlockManager(num_block=8)
    kv_cache = KVCache()

    first_block = bm.allocate_block("seq_C")
    seq = Sequence(seq_id="seq_C", initial_block_index=first_block)

    key, value = make_fake_kv(kv_cache)
    seq.append_token(key, value, kv_cache, layer=0)

    read_key, read_value = kv_cache.read(layer=0, block_index=first_block)

    assert torch.allclose(read_key[:, 0, :], key)
    assert torch.allclose(read_value[:, 0, :], value)


if __name__ == "__main__":
    test_append_token_within_first_block()
    test_append_token_crosses_block_boundary()
    test_written_data_is_retrievable()
    print("All Sequence tests passed.")
