class BlockManager:
    def __init__(self, num_block):
        self.free_block_pool = set(range(num_block))
        self.allocated_blocks = {} # seq_id -> list of block indices

    def allocate_block(self, seq_id):
        if not self.free_block_pool:
            raise RuntimeError("No free blocks available for allocation.")
        if seq_id not in self.allocated_blocks:
            self.allocated_blocks[seq_id] = []
        block_index = self.free_block_pool.pop()
        self.allocated_blocks[seq_id].append(block_index)
        return block_index

    def free_blocks(self, seq_id):
        if seq_id not in self.allocated_blocks:
            raise RuntimeError(f"No blocks allocated for seq_id {seq_id}.")
        for block_index in self.allocated_blocks[seq_id]:
            self.free_block_pool.add(block_index)
        del self.allocated_blocks[seq_id]

