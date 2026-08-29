# pico-vllm — Progress Documentation

A running record of everything built and verified so far, kept in first
person as a reference to pick back up from at any point.

---

## 1. Sizing (units kept clean)

I reason about cache capacity entirely in **tokens**, never bytes, inside
the block manager. Bytes only matter later, when the real tensor gets
allocated.

- `total_capacity_in_tokens`, `num_tokens_per_block` → `num_blocks`
- Config lives in `config.yaml` + `config.py`, loaded as a `Config`
  dataclass exposing `num_layers`, `num_heads`, `num_kv_heads`, `embed_dim`,
  `num_tokens_per_block`, and computed properties `num_blocks` and
  `head_dim`.
- Config values are the **real** architecture numbers for
  `Qwen/Qwen2.5-0.5B-Instruct`: 24 layers, 14 query heads, 2 KV heads
  (GQA), 896 embed_dim, head_dim 64.
- Hit and fixed a real naming collision: a `config/` package containing a
  `config.py` module containing a `config` variable caused ambiguous
  imports. Fixed by re-exporting the instance from `config/__init__.py`.

## 2. `BlockManager` — pure bookkeeping, no real data

Tracks which block indices are free/allocated, with no knowledge of
tensors, layers, or bytes at all.

- `free_block_pool`: a `set` — O(1) grab-any, O(1) return, order doesn't
  matter since any free block is as good as any other.
- `allocated_blocks`: a `dict[seq_id -> list]` — ordered list since blocks
  are always appended in generation order and need to be read back in
  that order later.
- `allocate_block(seq_id)` raises `RuntimeError` if the pool is exhausted,
  letting the caller (not the allocator) decide what to do.
- `free_blocks(seq_id)` returns all of a sequence's blocks at once and
  removes the `seq_id` entry entirely (no stale empty-list entries left
  behind).
- Tested in isolation: no cross-sequence collisions, correct exhaustion
  behavior, correct reuse after freeing.

## 3. `KVCache` — real tensors, verified against a real model

- Separate `key_cache` / `value_cache` tensors (not one combined tensor
  with a K/V dimension) — simpler to reason about at read/write time.
- Shape: `(num_layers, num_blocks, num_kv_heads, num_tokens_per_block,
  head_dim)` — `head_dim` last so each token's vector is one contiguous
  memory chunk, cheap to write/read. `num_layers`/`num_blocks` first,
  matching the natural `key_cache[layer, block]` access pattern.
- Sized using **`num_kv_heads`, not `num_heads`** — this model uses
  grouped-query attention (14 query heads, only 2 KV heads), discovered by
  inspecting the model's actual `k_proj`/`v_proj` output sizes. Sizing the
  cache with the larger query head count would have wasted memory storing
  data that never exists.
- `self.block_size` stored directly on the `KVCache` instance (rather than
  having every caller reach into global `config`) — a deliberate
  decoupling decision, so other modules only ever need a reference to the
  `KVCache` object they're given, never `config` itself.
- `write(layer, block_index, slot_in_block, key_vector, value_vector)` —
  bounds-checks `slot_in_block` (both too-high and negative), then writes
  directly into the tensor slice.
- `read(layer, block_index)` — returns the full block's key/value data.
- **Verified against real HF output**: ran `Qwen2.5-0.5B-Instruct` on real
  text, pulled `past_key_values.layers[0].keys/values`, wrote them
  token-by-token into `KVCache`, read back, and confirmed exact match via
  `torch.allclose` — including correctly handling the model's real
  bfloat16 dtype (had to explicitly match cache dtype to the model's,
  since a plain `torch.zeros(...)` defaults to float32).

## 4. `Sequence` — per-sequence state

Owns everything specific to one generating sequence: `seq_id`,
`block_indices` (ordered list, starts with one initial block passed in at
construction so it's never empty), and `seq_len` (starts at 0).

- **Design decision**: `Sequence` does *not* hold a reference to
  `BlockManager` and does not self-allocate. Reasoned through why:
  allocation is a scheduling decision (who gets memory, under what
  contention) that doesn't belong to a single sequence's own logic, and
  coupling them would make `Sequence` untestable without a live
  `BlockManager` always present. Instead, `append_token(...)` accepts an
  optional `new_block_index` and raises `ValueError` if a block just
  filled up and none was provided — pushing the allocation decision back
  to the caller.
- **Edge case caught**: the naive check `seq_len % block_size == 0` is
  true both when a block has just filled up AND when `seq_len == 0`
  (nothing written yet) — these need opposite behavior. Fixed with an
  added `seq_len > 0` guard so the "need a new block" check only fires
  once a block is genuinely full.
- Tested in isolation: normal appends within one block, the
  boundary-crossing case (including the raised-error path), and a
  correctness check that written data is actually retrievable afterward.

## 5. `gather_kv` — reconstructing a sequence's real KV data

Given a `Sequence`'s `block_indices` and `seq_len`, reconstructs one
clean, padding-free tensor per K and V, ready for attention.

- Reasoned through why padding matters: attention computes a weighted sum
  over every position it's given, so feeding in zero-padded slots from a
  partially-filled block would silently corrupt the output by attending
  to garbage — not just lose precision.
- Realized only the **last** block in a sequence can ever be partially
  filled (every earlier block is guaranteed full, since `Sequence` only
  allocates a new block once the current one is completely full) — so the
  simplest correct approach is concatenate-everything-first via
  `torch.cat`, then a single final slice down to `seq_len`, rather than
  trimming block-by-block during the loop.
- Fixed a dimension-indexing bug: `kv_cache.read(...)` already slices away
  the `layer` dimension, so each block is 3D
  (`num_kv_heads, tokens, head_dim`), not 4D — concatenation and slicing
  needed to target `dim=1` (tokens), not `dim=2`.

## 6. `paged_attention` — full attention using the paged cache

Runs `scaled_dot_product_attention` using `gather_kv`'s reconstructed
data, correctly handling this model's GQA setup.

- **GQA head expansion**: 14 query heads, 2 KV heads → each KV head serves
  7 query heads. Used `repeat_interleave(7, dim=0)` (not plain `.repeat`)
  specifically because `repeat_interleave` produces
  `[head0×7, head1×7]` — the correct grouped layout — whereas `.repeat`
  would tile the whole sequence and misalign the grouping.
- **Causal masking**: `is_causal` is tied directly to an `is_prefill`
  parameter — `True` during prefill (a full prompt's queries must not see
  future tokens within the same prompt), `False` during decode (a single
  new token's query attending over a cache of strictly-past tokens has
  nothing future to hide).
- Adds/removes the batch dimension (`unsqueeze(0)` / `squeeze(0)`) around
  the `scaled_dot_product_attention` call, since it expects a
  `(batch, heads, seq_len, head_dim)` layout.
- **Verified against real HF attention output**, using a sentence long
  enough to force a block-boundary crossing. This required learning and
  using **forward hooks** for the first time — registering a hook on
  `q_proj` to capture the raw query, and on `o_proj` to capture its input
  (the real pre-projection attention output) as ground truth.
- **Real bug found and fixed during this test**: the raw captured query
  was pre-RoPE, but HF's cached keys (from `past_key_values`) are already
  post-RoPE — comparing a rotated key against an unrotated query produced
  a consistent, non-random-looking error that grew with position,
  correctly diagnosed as a RoPE misalignment rather than a deeper logic
  bug. Fixed by applying the same rotary embedding to the captured query
  before comparison. Test passed after this fix.

## 7. `model_wrapper.forward_one_sequence` — full multi-layer forward pass

A complete 24-layer forward pass for **one sequence at a time**, reusing
the model's real pretrained weights (layernorms, projections, MLP)
unchanged, but substituting this project's own `paged_attention` +
`KVCache` in place of HF's built-in attention.

- Clarified an important scoping question first: this isn't "reimplement
  a transformer from scratch" in the sense of inventing new math — it's
  writing the loop that calls the same pretrained weight matrices in the
  same order, with one component (attention) swapped out. This is
  actually the same approach real serving engines like vLLM take per
  supported architecture.
- Per layer: input layernorm → q/k/v projections → reshape into heads →
  RoPE on Q and K (K needs RoPE too, before being written to cache, since
  the cache stores whatever it's given) → write K/V into `KVCache` via
  `Sequence.append_token` → `paged_attention` → `o_proj` → residual →
  post-attention layernorm → MLP → residual.
- **Design detail**: block allocation and `seq.seq_len` incrementing only
  happen once per token (driven by `layer_idx == 0`), not once per token
  per layer — since a `Sequence`'s block indices are shared state across
  all layers, not per-layer state. Every other layer just writes into the
  already-determined slot directly via `kv_cache.write(...)`.
- **Position IDs matter for correctness**: RoPE must be applied using the
  token's true absolute position (`position_offset = seq.seq_len`,
  captured before the current call's writes), not always starting from 0
  — critical for decode steps, where new tokens are at nonzero positions.

### Verification — full pipeline vs. real HF generation

- First test attempted exact token-for-token match against
  `model.generate(...)` (greedy decoding). Tokens 0-2 matched exactly;
  token 3 diverged. Diagnosed as likely **numerical drift** (bf16
  precision), not a logic bug — reasoning: a real logic bug would likely
  cause either a huge discrepancy or errors that compound and grow
  sharply, not a small, delayed, single-token divergence.
- Rewrote the test to compare **logits closeness** at each step
  (`torch.allclose`) against HF re-run on the *exact same prefix* my
  pipeline actually produced, rather than requiring two independently-run
  greedy trajectories to match exactly — this correctly isolates whether
  the math is right, independent of downstream argmax sensitivity when
  two candidate tokens are nearly tied.
- In bf16, logits differed by a small, roughly constant amount at each
  decode step (~0.17-0.2 max abs diff) — consistent with drift baked into
  the cache once during prefill and persisting, not compounding wildly.
- **Confirmed the diagnosis experimentally**: switched both the model and
  `KVCache` tensors to float32, reran with the original tight tolerance —
  test passed cleanly. This confirmed the discrepancy was genuinely bf16
  precision drift, not a bug in the implementation.

## Where this leaves me

I have a complete, correctness-verified single-sequence generation
pipeline: `BlockManager` → `KVCache` → `Sequence` → `gather_kv` →
`paged_attention` → `model_wrapper.forward_one_sequence`, proven correct
end-to-end against real HF model output, for both prefill and multi-step
decode, in float32 (with bf16 precision behavior understood and
explained, not just observed).

## Next up — naive batching baseline (step 5)

Wrapping `forward_one_sequence` to run **multiple sequences** concurrently
as a naive/static batch, using one **shared** `KVCache` and
`BlockManager` (each sequence just gets its own non-overlapping block
indices from the same pool — giving each sequence a private `KVCache`
would waste memory by multiplying total capacity by the batch size for no
reason).

Key design decisions already reasoned through:
- All sequences prefill first, then decode proceeds in lockstep steps.
- The defining inefficiency of naive/static batching (the thing the later
  continuous-batching comparison is meant to demonstrate): once a
  sequence hits EOS, the naive batch keeps calling
  `forward_one_sequence` for it anyway every remaining step — burning
  real compute generating tokens nobody wants — purely because the batch
  as a whole cannot advance until every sequence in it has finished.
- Per-sequence state (the `Sequence` object, its `generated_ids` so far,
  and a `finished` flag) needs a small wrapper — a dict keyed by
  `seq_id`, rather than growing `Sequence` itself to hold generation
  bookkeeping that isn't really about cache/block state.

Still to write: the setup loop (tokenize each prompt, allocate an initial
block per sequence, construct each `Sequence`), the prefill loop (run
`forward_one_sequence` once per sequence with `is_prefill=True`, check for
EOS), and the decode loop (keep stepping every sequence, finished or not,
until all are finished or a max token count is hit).