# pico-vllm — Day 2 Progress

Today I moved from pure storage (block allocation + KV-cache tensors) into
actually wiring things together at the sequence level, and started on the
paged attention gather step — pulling scattered block data back into
something attention can actually consume.

## 1. Fixing a naming collision in config

Ran into `AttributeError: module 'config.config' has no attribute 'num_layers'`
when trying to use my config from other modules. Turned out I had a package
folder called `config/` containing a module also called `config.py`, which
itself defined a variable also called `config` — so `from config import config`
was resolving ambiguously. Fixed it by re-exporting the instance explicitly
from `config/__init__.py`.

Also caught a subtler bug after that: my `config.yaml` still had leftover
placeholder values (`num_heads: 2`, `embed_dim: 120`, `num_layers: 4`) from
before I'd actually inspected the real model architecture. Since my whole
point was verifying against the real Qwen2.5-0.5B model, I updated these to
the real values I'd already worked out — 24 layers, 14 query heads, 896
embedding dimension, 2 KV heads (GQA) — which fixed a `head_dim` mismatch
(was computing 60 instead of the correct 64).

One more environment issue along the way: a dtype mismatch between my
cache tensors (float32 by default) and the real model's output
(bfloat16), which I resolved by making my cache tensors bfloat16 too,
since the whole point of the verification test is to faithfully match
what the model actually produces — not to introduce my own precision.

## 2. Building the `Sequence` class

Before this, I only had two decoupled pieces: `BlockManager` (which block
indices are free/allocated) and `KVCache` (the real tensors). Neither knew
about sequences, token counts, or block *ordering* over time. I needed
somewhere to hold that per-sequence state.

I considered letting `Sequence` hold a reference to `BlockManager` and
self-allocate new blocks whenever it ran out of room, but talked through
why that's the wrong call: allocation is a *scheduling* decision (who gets
memory, when, and what happens under contention), not something a single
sequence should decide unilaterally. It would also make `Sequence`
untestable in isolation, since every test would need a live `BlockManager`
too. Instead, `Sequence.append_token(...)` accepts an optional
`new_block_index` parameter and raises an exception if a block just filled
up and no new block was provided — pushing the allocation decision back
out to whoever is calling it.

Hit a real edge-case bug here: my first attempt used
`seq_len % block_size == 0` to detect "block just filled up, need a new
one" — but that condition is also true at `seq_len == 0`, before any
tokens have been written at all. Fixed it with an added
`seq_len > 0 and ...` guard, so it only fires once a block has genuinely
been filled.

Also cleaned up a design smell: `Sequence.append_token` needed to know the
block size, and I initially reached for the global `config` object
directly inside `sequence.py`. Talked through why that's worse than it
sounds — every module that reaches into global config for one value
duplicates that assumption everywhere, instead of owning it in one place.
Fixed it by giving `KVCache` its own `self.block_size` attribute at
construction time, so `Sequence` only ever asks the specific `KVCache`
object it already has a reference to, and never needs to know `config`
exists.

Wrote and passed three isolated tests for `Sequence`: normal appends
within a single block, the block-boundary-crossing case (including the
error path when no new block is provided), and a correctness check that
written data is actually retrievable afterward, not just that the
counters look right.

## 3. Paged attention — the gather step

Started on step 4: reconstructing a sequence's real (non-padded) key/value
data from its scattered blocks, ready to feed into real attention.

Key reasoning before writing any code: attention computes a weighted sum
over every key/value position it's given, so if I gathered a partially
full block without trimming the padding, the model would end up attending
to zero-vectors that were never real tokens — silently corrupting the
result, not just losing precision. So `gather_kv` needs the sequence's
real token count, not just its list of blocks.

Worked out that only the *last* block in a sequence can ever be partially
filled — every block before it is guaranteed full, since `Sequence` only
allocates a new block once the current one is completely full. That
meant the simplest correct approach was to concatenate all blocks first
via `torch.cat`, then do a single final slice down to the real sequence
length, rather than trying to trim padding block-by-block during the loop.

Got the dimension bookkeeping wrong on the first pass — `kv_cache.read(...)`
returns a block with the `layer` dimension already sliced away, so the
result is 3D (`num_kv_heads, tokens, head_dim`), not 4D. Fixed the
`torch.cat` dimension and the final slice to match.

## Where this leaves me

I now have a full, tested chain: `BlockManager` → `KVCache` → `Sequence`
→ `gather_kv`, each piece properly decoupled and independently testable.
Started reasoning through the next real problem — this model uses
grouped-query attention (14 query heads, only 2 KV heads), so before I
can run real attention on the gathered keys/values, I need to expand each
KV head to line up against the 7 query heads it serves, using
`repeat_interleave` (not a plain `repeat`, since the head ordering has to
stay grouped correctly: head0 repeated 7 times, then head1 repeated 7
times — not the whole sequence tiled).

## Next up

Finish wiring the actual `scaled_dot_product_attention` call: expand the
gathered K/V via `repeat_interleave`, figure out the causal masking
situation for autoregressive generation, and then write the verification
test comparing this against real HF attention output — this time using a
longer sentence that actually crosses a block boundary, to properly
exercise `Sequence` and `gather_kv` together rather than just testing a
single block in isolation.