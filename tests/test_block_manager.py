from block_manager import BlockManager

def test_no_collision_between_sequences():
    bm = BlockManager(num_block=10)

    blocks_a = [bm.allocate_block("A") for _ in range(3)]
    blocks_b = [bm.allocate_block("B") for _ in range(3)]

    assert set(blocks_a).isdisjoint(set(blocks_b)), "Sequences A and B share a block index!"
    assert bm.allocated_blocks["A"] == blocks_a
    assert bm.allocated_blocks["B"] == blocks_b

def test_pool_exhaustion_raises():
    bm = BlockManager(num_block=3)

    bm.allocate_block("A")
    bm.allocate_block("A")
    bm.allocate_block("A")

    try:
        bm.allocate_block("A")
        assert False, "Expected RuntimeError when pool is exhausted"
    except RuntimeError:
        pass  # expected

def test_freed_blocks_are_reusable():
    bm = BlockManager(num_block=3)

    blocks = [bm.allocate_block("A") for _ in range(3)]
    bm.free_blocks("A")

    assert len(bm.free_block_pool) == 3
    assert "A" not in bm.allocated_blocks

    # should succeed without raising, since blocks were returned
    reused = bm.allocate_block("B")
    assert reused in blocks  # must be one of the originally allocated indices

if __name__ == "__main__":
    test_no_collision_between_sequences()
    test_pool_exhaustion_raises()
    test_freed_blocks_are_reusable()
    print("All tests passed.")