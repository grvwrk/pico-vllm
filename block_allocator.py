total_capacity_in_tokens = 32000  # example: total tokens worth of KV-cache space
num_tokens_per_block = 16
num_block = total_capacity_in_tokens // num_tokens_per_block

free_block_pool = set(range(num_block))

allocated_blocks = {}  # seq_id -> list of block indices

def allocate_block(seq_id):
    if not free_block_pool:
        raise RuntimeError("No free blocks available for allocation.")
    if seq_id not in allocated_blocks:
        allocated_blocks[seq_id] = []
    block_index = free_block_pool.pop()
    allocated_blocks[seq_id].append(block_index)
    return block_index

def free_blocks(seq_id):
    if seq_id not in allocated_blocks:
        raise RuntimeError(f"No blocks allocated for seq_id {seq_id}.")
    for block_index in allocated_blocks[seq_id]:
        free_block_pool.add(block_index)
    del allocated_blocks[seq_id]