# pico-vllm : Day 1 Progress

Today I started building the core memory management system for pico-vllm —
a from-scratch, single-GPU inference engine implementing PagedAttention-style
KV-cache management and continuous batching. I focused on two pieces: the
block allocator and the real KV-cache tensors it manages.

## 1. Sizing (keeping units clean)

I worked out how many fixed-size blocks I get out of a given total capacity
and a chosen block size. I made sure to reason about this entirely in
**tokens**, not bytes. Bytes only matter later when I actually allocate the
real tensor — the block manager itself should never need to know about
byte sizes at all.

## 2. Block allocator : pure bookkeeping, no real data

I built a class that tracks two things: a pool of currently free blocks,
and a mapping from each sequence to the list of blocks it currently owns.
It supports allocating a block to a sequence and freeing all of a
sequence's blocks at once.

Design decisions I made here:
- I used a set for the free-block pool, since I need O(1) "grab any free
  block" and O(1) "return a block to the pool," and any free block is as
  good as any other — order doesn't matter there.
- I used a dictionary mapping each sequence to an ordered list of block
  indices, since blocks are always appended in order as a sequence grows,
  and I'll need to iterate them in order later when gathering KV data for
  attention. I don't need fast membership checks or mid-list removal —
  only append, iterate, and free-all-at-once — so a plain list was the
  right call, nothing fancier needed.

I tested this with pytest, covering:
- No block index collisions between two different sequences
- Correct failure behavior when the pool is exhausted
- Freed blocks are correctly reusable by a new allocation

All three tests passed.

## 3. Real KV-cache tensors : where block indices become real memory

I allocated the actual tensors that back these block indices. A few key
decisions and my reasoning:
- I used separate key and value tensors, rather than one combined tensor
  with an extra "K-or-V" dimension — simpler to reason about at read/write
  time.
- I put each token's key/value vector as the innermost, contiguous chunk
  of memory, rather than spreading it across the tensor — this makes
  writing a new token's data cheap, instead of writing into a strided,
  non-contiguous slice every time.
- I ordered the tensor so that layer and block come first, since that
  matches how I'll actually access the cache — each transformer layer has
  its own independent K/V data and doesn't share it with other layers, so
  indexing by layer and then block gives me exactly the slice I need.

I confirmed the indexing works exactly as expected — indexing by layer and
block gives me back the full key data for that block, across all heads and
all token slots in the block.

## Where this fits in the bigger picture

This covers the block manager and the KV-cache tensor storage from my repo
plan. The block manager decides *which* block index a sequence can use;
the KV-cache tensors are the real GPU memory where that index becomes an
actual writable/readable slice.

## Next up

Writing a token's key/value into a specific block and slot when a sequence
generates a new token, then verifying the result against a standard
Hugging Face model's real cached output for the same sequence — this is my
correctness anchor before I move on to the paged attention gather step.