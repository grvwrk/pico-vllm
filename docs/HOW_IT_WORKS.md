# pico-vllm — Architecture

This document explains every module in pico-vllm — what each class and
function does, why it's designed that way, and how a request actually
flows through the system from an HTTP call down to a single tensor write.
It assumes no prior context beyond general Python and a basic sense of
what a transformer is. Where a design choice isn't obvious, the reasoning
is included, not just the "what."

---

## 1. The big picture

pico-vllm is a small, from-scratch reimplementation of the two ideas that
make vLLM fast: **paged KV-cache memory management** and **continuous
(iteration-level) batching**. It runs a real Hugging Face model
(`Qwen2.5-0.5B-Instruct`) but replaces the model's built-in attention with
a custom paged-attention implementation, and replaces "run one fixed batch
until everyone finishes" with a scheduler that can admit and evict
requests every single step.

At a glance, a request's journey looks like this:

```
HTTP request
  → SchedulerService (async bridge)
    → Scheduler.step()            (admit waiting requests, decide who runs)
      → Qwen2Runner.forward_one_sequence()   (one prefill or decode pass)
        → paged_attention()                  (per layer, per sequence)
          → gather_kv()                       (reconstruct real K/V from blocks)
          → KVCache.read() / .write()          (the actual tensor storage)
        → BlockManager.allocate_block()        (get more room when a block fills up)
```

Nothing here trains anything or reinvents transformer math — every linear
layer, layernorm, and MLP block uses the model's real pretrained weights,
untouched. The only thing pico-vllm replaces is **how attention reads its
key/value history** and **how multiple requests share GPU memory over
time**.

---

## 2. `core/` — memory management (the part with no model-specific logic)

Everything in `core/` knows nothing about transformers, layers, or
tokens-as-language. It only knows about block indices and tensors. This
separation is deliberate: it means the hardest-to-get-right code (memory
safety) can be tested completely in isolation, with no GPU, no model
download, and no randomness.

### 2.1 `block_manager.py` — `BlockManager`

**What it solves:** the KV-cache is one big pre-allocated tensor, divided
into fixed-size "blocks" (chunks of `num_tokens_per_block` token-slots
each). Every sequence needs some number of these blocks to store its
growing history. `BlockManager` is the bookkeeper that tracks which block
indices are currently free, and which sequence owns which blocks. It never
touches the actual tensor — it only deals in integers (block indices).

```python
class BlockManager:
    def __init__(self, num_block):
        self.free_block_pool = set(range(num_block))
        self.allocated_blocks = {}   # seq_id -> [block indices], in order
```

- **`free_block_pool` is a `set`.** Any free block is as good as any
  other — there's no ordering requirement — so a `set` gives O(1) "give me
  any free one" (`.pop()`) and O(1) "return this one" (`.add()`).
- **`allocated_blocks` is a `dict` of `seq_id -> list`.** It has to be a
  `list`, not a `set`, because a sequence's blocks are meaningful *in
  order*: block 0 holds tokens 0–15, block 1 holds tokens 16–31, and so
  on. Later code (`gather_kv`) depends on iterating them in that order.

**`allocate_block(seq_id)`** — pops one index out of `free_block_pool`,
appends it to `allocated_blocks[seq_id]` (creating the list if this is the
sequence's first block), and returns the index. If the pool is empty, it
raises `RuntimeError` rather than blocking or silently failing — it is
**not** this class's job to decide what happens when memory runs out
(wait? evict someone else? reject the request?). That's a scheduling
decision, made one layer up, in `Scheduler`.

**`free_blocks(seq_id)`** — returns every block a sequence owns back to
`free_block_pool` in one call, then deletes the sequence's entry entirely
(not just empties the list — a lingering empty list would make
`if seq_id in allocated_blocks` incorrectly return `True` for a sequence
that's actually finished).

### 2.2 `kv_cache.py` — `KVCache`

**What it solves:** this is the actual GPU memory. Two tensors,
`key_cache` and `value_cache`, both shaped:

```
(num_layers, num_blocks, num_kv_heads, block_size, head_dim)
```

Why this exact shape and ordering, dimension by dimension:

- **`num_layers` first.** Every transformer layer computes and caches its
  own independent keys/values — layer 3 never sees layer 7's cache. Since
  a forward pass processes one layer at a time, `key_cache[layer]` should
  be a clean slice, not a strided gather.
- **`num_blocks` second.** This is what `BlockManager`'s integers actually
  index into. `key_cache[layer, block_index]` gives you everything stored
  for one sequence's one block, at one layer.
- **`num_kv_heads`, not `num_heads`.** This model uses **grouped-query
  attention (GQA)**: 14 query heads, but only 2 actual key/value heads
  (`Qwen2.5-0.5B-Instruct`'s `k_proj`/`v_proj` output far fewer dimensions
  than `q_proj`). Sizing the cache with the larger query head count would
  silently waste memory storing K/V data that never exists. This is
  exactly what real vLLM does too — the KV-cache is sized by *KV* heads.
- **`block_size` (tokens-per-block) before `head_dim`.** This is a
  memory-layout choice: writing one new token's key vector should touch
  one contiguous chunk of memory, not a strided slice. Putting `head_dim`
  last means `key_cache[layer, block, :, slot, :]` — the actual write
  target for one token — is contiguous in memory.

```python
class KVCache:
    def __init__(self, *, architecture=None, num_blocks=None,
                 block_size=None, device=None, dtype=None):
        self.architecture = architecture or ModelArchitecture(...)  # from config, if not given
        ...
        self.key_cache = torch.zeros(num_layers, num_blocks, num_kv_heads,
                                      block_size, head_dim, device=device, dtype=dtype)
        self.value_cache = torch.zeros_like(self.key_cache)
        self.block_size = block_size
```

Note `self.block_size` is stored **on the instance**, not read from a
global `config` by every caller. This is a deliberate decoupling: any code
that has a `KVCache` object can ask it for its own block size, without
also needing to import and know about the global config module. (See
§2.3 — `Sequence` relies on exactly this.)

**`write(layer, block_index, slot_in_block, key_vector, value_vector)`**
— bounds-checks `slot_in_block` (must be `0 <= slot_in_block < block_size`,
catching both "too high" and "negative" — a negative index would silently
wrap around and corrupt the *wrong* slot instead of erroring), then writes
directly into the tensor slice.

**`read(layer, block_index)`** — returns the full block's key and value
data: shape `(num_kv_heads, block_size, head_dim)`. It does not know or
care whether every slot in that block actually holds real data yet — that
distinction is `gather_kv`'s job (§4.1), not `KVCache`'s.

### 2.3 `sequence.py` — `Sequence`

**What it solves:** per-request state that `BlockManager` and `KVCache`
don't track — specifically, *how many real tokens* a given sequence has
generated so far, and *which* blocks (in order) belong to it.

```python
class Sequence:
    def __init__(self, seq_id, initial_block_index):
        self.seq_id = seq_id
        self.block_indices = [initial_block_index]  # never starts empty
        self.seq_len = 0
```

`block_indices` is deliberately seeded with one block at construction
time, rather than starting empty — this avoids an edge case where "no
blocks yet" and "current block just filled up" would otherwise look
identical (see below).

**`append_token(key_vector, value_vector, kv_cache, layer, new_block_index=None)`**
— the only method. Given one new token's key/value, it works out *which*
block and *which slot within that block* this token belongs in, purely
from `seq_len` and `kv_cache.block_size`:

```python
if self.seq_len > 0 and self.seq_len % kv_cache.block_size == 0:
    # current block just became completely full — need a new one
    if new_block_index is None:
        raise ValueError(...)
    self.block_indices.append(new_block_index)

current_block_index = self.block_indices[-1]
slot_in_block = self.seq_len % kv_cache.block_size
kv_cache.write(layer, current_block_index, slot_in_block, key_vector, value_vector)
self.seq_len += 1
```

Two things worth understanding here:

1. **`self.seq_len > 0` guard.** The naive check
   `seq_len % block_size == 0` is `True` both when a block has genuinely
   just filled up (`seq_len` = 16, 32, ...) *and* when `seq_len == 0`
   (nothing written yet). Without the `> 0` guard, the very first token
   ever written would incorrectly demand a new block, even though the
   sequence already has its initial one.
2. **`Sequence` never allocates blocks itself.** It doesn't hold a
   reference to `BlockManager`. When it needs a new block, it raises
   `ValueError` and expects the *caller* to catch that, call
   `BlockManager.allocate_block(...)`, and retry with the result passed in
   as `new_block_index`. This is a deliberate separation: deciding *who*
   gets memory when the system is under pressure is a scheduling decision
   (see `Scheduler`, §5), not something one sequence should decide
   unilaterally. It also means `Sequence` can be unit-tested with zero
   dependency on a live `BlockManager`.

### 2.4 `paged_attention.py` — `gather_kv` and `paged_attention`

**`gather_kv(kv_cache, layer, seq)`** reconstructs one sequence's *real*
(non-padded) key/value history from its scattered blocks, ready for
attention:

```python
def gather_kv(kv_cache, layer, seq):
    key_vectors_list, value_vectors_list = [], []
    for block_index in seq.block_indices:
        k, v = kv_cache.read(layer, block_index)
        key_vectors_list.append(k)
        value_vectors_list.append(v)

    key_vectors = torch.cat(key_vectors_list, dim=1)   # tokens dimension
    value_vectors = torch.cat(value_vectors_list, dim=1)

    key_vectors = key_vectors[:, :seq.seq_len, :]        # trim padding
    value_vectors = value_vectors[:, :seq.seq_len, :]
    return key_vectors, value_vectors
```

Why concatenate-then-slice, rather than trimming each block individually
inside the loop: **only the last block a sequence owns can ever be
partially full** — every earlier block is guaranteed complete, because
`Sequence.append_token` only ever requests a new block once the current
one is entirely full. So the padding problem only ever exists at the very
end of the concatenated tensor, and one final slice handles it correctly
regardless of how many blocks are involved.

Why this trimming matters at all: attention computes a weighted sum over
every key/value position it's given. Feeding in an unpadded, all-zero
"phantom token" from an unfilled slot wouldn't just be imprecise — it
would actively corrupt the output by attending to data that was never
part of the real sequence.

**`paged_attention(query, kv_cache, layer, seq, is_prefill)`** — runs the
actual attention math:

```python
def paged_attention(query, kv_cache, layer, seq, is_prefill):
    gathered_keys, gathered_values = gather_kv(kv_cache, layer, seq)

    group_size = kv_cache.architecture.num_heads // kv_cache.architecture.num_kv_heads
    keys_expanded = gathered_keys.repeat_interleave(group_size, dim=0)
    values_expanded = gathered_values.repeat_interleave(group_size, dim=0)

    query = query.unsqueeze(0)
    keys_expanded = keys_expanded.unsqueeze(0)
    values_expanded = values_expanded.unsqueeze(0)

    output = torch.nn.functional.scaled_dot_product_attention(
        query, keys_expanded, values_expanded, is_causal=is_prefill,
    )
    return output.squeeze(0)
```

Two subtleties worth understanding:

- **`repeat_interleave`, not `.repeat()`, for GQA head expansion.** With
  14 query heads and 2 KV heads, each KV head has to serve 7 query heads.
  `repeat_interleave(7, dim=0)` produces `[kv_head0 ×7, kv_head1 ×7]` —
  query heads 0–6 correctly line up against KV head 0, and 7–13 against
  KV head 1. A plain `.repeat(7, ...)` would instead tile the *whole
  sequence* of KV heads 7 times (`[kv_head0, kv_head1, kv_head0,
  kv_head1, ...]`), which misaligns the grouping entirely.
- **`is_causal=is_prefill`, not always `True` or always `False`.** During
  **prefill**, the query represents the *entire* prompt at once, so a
  causal mask is required — token 5 must not see token 8. During
  **decode**, the query is a single new token, and everything in the
  gathered cache is by construction already in the past — there is no
  "future" position present to mask out, so the mask would be a no-op at
  best.

This function was verified against real Hugging Face attention output
using PyTorch forward hooks (see §7) — including catching a real bug
where the captured query was pre-RoPE while HF's cached keys are already
post-RoPE, causing a silent-but-real mismatch until fixed.

---

## 3. `models/` — the model-specific forward pass

This is the layer that knows about transformers, layers, RoPE, and GQA
head counts — everything in `core/` above is architecture-agnostic, but
someone has to actually drive a real model through it.

### 3.1 `base.py` — `ModelArchitecture` and `ModelRunner`

`ModelArchitecture` is a small frozen dataclass holding the four numbers
that actually matter for cache sizing: `num_layers`, `num_heads`,
`hidden_size`, `num_kv_heads` (plus a computed `head_dim` property). This
exists so `KVCache` doesn't need to know *which* model it's serving — it
just needs these four numbers, however they were derived.

`ModelRunner` is an abstract base class defining the "contract" any
supported model architecture must fulfill:

```python
class ModelRunner(ABC):
    model_id: str
    model: torch.nn.Module
    tokenizer: object
    architecture: ModelArchitecture

    @abstractmethod
    def forward_one_sequence(self, kv_cache, block_manager, sequence,
                              input_ids, *, is_prefill: bool):
        """Run one prefill or decode forward pass and return last-token logits."""
```

This mirrors how real vLLM works: it supports 200+ architectures by
having one adapter class per family, each implementing that family's
specific layer order, RoPE variant, and attention head layout — while
everything above this layer (scheduler, KV-cache, API) stays completely
architecture-agnostic.

### 3.2 `qwen2.py` — `Qwen2Runner`

The one concrete implementation currently registered. `from_pretrained(...)`
loads the real Hugging Face model and tokenizer and reads
`num_hidden_layers`, `num_attention_heads`, `hidden_size`, and
`num_key_value_heads` straight from the model's own config — no
hardcoded numbers, so it stays correct if the model checkpoint changes.

**`forward_one_sequence(kv_cache, block_manager, sequence, input_ids, is_prefill)`**
is the real per-layer loop. Per layer, in order:

1. **Input layernorm**, then **Q/K/V projections** — all direct calls
   into the model's own pretrained weights, untouched.
2. **Reshape into heads**, then **RoPE applied to both Q and K** — using
   `position_ids` offset by `sequence.seq_len` (how many tokens are
   already cached), so a decode step at absolute position 47 gets rotated
   correctly, not as if it were position 0.
3. **Write K/V into the cache**, one token at a time, via
   `sequence.append_token(...)`. One important detail: block allocation
   and `sequence.seq_len` incrementing only happen when `layer_idx == 0`
   — a sequence's block list is *shared state across all 24 layers*, not
   per-layer, so advancing it once per token (not once per token per
   layer) is essential. Every other layer just computes the already-known
   slot and calls `kv_cache.write(...)` directly:
   ```python
   cache_position = sequence.seq_len - seq_len + token_index
   block_index = sequence.block_indices[cache_position // kv_cache.block_size]
   kv_cache.write(layer_idx, block_index, cache_position % kv_cache.block_size, ...)
   ```
4. **`paged_attention(...)`** — the custom attention from §2.4, in place
   of the model's own attention module.
5. **`o_proj`**, **residual connection**, **post-attention layernorm**,
   **MLP**, **second residual** — again, all direct calls into pretrained
   weights.

After all layers: final layernorm, then `lm_head`, returning logits for
only the last position (the next-token prediction).

### 3.3 `registry.py` — `ModelRegistry`

A small lookup table mapping a Hugging Face model ID string to its
`ModelRunner` subclass:

```python
class ModelRegistry:
    _runners = {"Qwen/Qwen2.5-0.5B-Instruct": Qwen2Runner}

    @classmethod
    def create(cls, model_id, *, device, dtype):
        runner_type = cls._runners[model_id]   # raises a clear error if unsupported
        return runner_type.from_pretrained(model_id, device=device, dtype=dtype)
```

This is what the server uses at startup — it fails fast with a clear
message if asked to load a model with no registered runner, rather than
silently loading incompatible weights.

---

## 4. `engine/` — batching strategies

Two different strategies for running multiple sequences at once, both
built on the exact same `core/` and `models/` pieces above. Comparing
them is the whole point of the benchmarks (§8).

### 4.1 `naive_batching.py` — `run_naive_batch`

The baseline. Every prompt gets its own `Sequence`, all sharing one
`KVCache`/`BlockManager` (each just gets its own non-overlapping block
indices from the shared pool — giving each sequence a fully private
`KVCache` would multiply total memory use by the batch size for no
reason).

The flow: prefill every sequence once, then decode in lockstep — one
step advances *every* sequence in the batch by one token, repeated until
either everyone's finished or `max_new_tokens` is hit.

**The deliberate flaw, demonstrated on purpose:** once a sequence hits
EOS, naive batching keeps calling its forward pass anyway, every
remaining step, purely because the batch as a whole can't advance until
the *slowest* sequence finishes:

```python
for step in range(max_new_tokens - 1):
    for seq_id, entry in sequences.items():
        # naive batching keeps stepping this sequence even if it already
        # finished — that's exactly the wasted compute this baseline
        # is meant to demonstrate
        ...
```

This isn't a bug — it's the thing the benchmark exists to measure (see
§8.1).

### 4.2 `model_wrapper.py`

A thin backward-compatibility shim. Early in this project's development,
`forward_one_sequence` was a free function (before the `ModelRunner`
abstraction existed). This file now just wraps `Qwen2Runner` so that
older benchmarks/tests written against the old function signature keep
working without modification.

### 4.3 `scheduler.py` — `Scheduler` (the actual continuous batching)

This is the core contribution of the project. Instead of "run one fixed
batch until everyone finishes," it re-evaluates **who should be running**
before every single forward-pass step.

**Sequence states**, tracked via a small wrapper class:

```python
WAITING = "waiting"    # queued, not yet admitted
RUNNING = "running"    # actively being stepped
FINISHED = "finished"  # done, blocks freed

class ManagedSequence:
    def __init__(self, seq_id, prompt_ids, max_new_tokens):
        self.seq_id = seq_id
        self.prompt_ids = prompt_ids
        self.max_new_tokens = max_new_tokens
        self.generated_ids = []
        self.state = WAITING
        self.seq = None                  # real Sequence, created at admission time
        self.has_been_prefilled = False
```

Note `ManagedSequence` is deliberately a *separate* class from `Sequence`
(§2.3) — `Sequence` only tracks cache/block bookkeeping, while
`ManagedSequence` tracks scheduling bookkeeping (generated tokens so far,
whether prefill has run, per-request `max_new_tokens`). Keeping these
separate means `Sequence` stays reusable outside a scheduling context
(e.g. in the naive batching baseline, or in isolated tests).

**`Scheduler.__init__`** accepts either a `ModelRunner` directly, or (for
backward compatibility with older benchmarks) a raw Hugging Face model
plus tokenizer, which it silently wraps in a `Qwen2Runner`.

**`add_request(seq_id, prompt, max_new_tokens)`** — tokenizes the prompt
and appends a new `ManagedSequence` to `waiting_queue`. Notice it does
**not** allocate any blocks yet — a request sits in the queue as pure
bookkeeping until there's actually room for it.

**`_evict(managed)`** — called the moment a sequence finishes. Frees its
blocks back to `BlockManager` *immediately*, moves it into `self.finished`,
and removes it from `self.running`. This immediate reclaim — not waiting
for the whole batch — is the entire mechanism that makes continuous
batching better than naive batching.

**`_try_admit()`** — pulls sequences out of `waiting_queue` into
`running`, first-come-first-served, as long as `BlockManager` has at
least one free block:

```python
while self.waiting_queue and self.bm.free_block_pool:
    managed = self.waiting_queue.pop(0)
    initial_block = self.bm.allocate_block(managed.seq_id)
    managed.seq = Sequence(seq_id=managed.seq_id, initial_block_index=initial_block)
    managed.state = RUNNING
    self.running[managed.seq_id] = managed
```

Only one free block is required to admit — enough for the sequence's
*first* token. If it later needs more blocks as it grows, it goes through
the exact same `Sequence.append_token` → `ValueError` → allocate flow as
any other sequence, handled inside `_step_sequence` below.

**`_step_sequence(managed)`** — runs exactly one forward pass (prefill if
this is the sequence's first call, otherwise one decode step), picks the
next token via greedy argmax, and checks whether the sequence should be
evicted (hit EOS, or hit its own `max_new_tokens` — note each sequence
can have a *different* limit, unlike naive batching's one shared limit
for the whole batch).

**`step()`** — the actual unit of work, used both by the synchronous
benchmark scripts and the async server (§5):

```python
def step(self) -> bool:
    self._try_admit()
    if not self.running:
        return bool(self.waiting_queue)

    tokens_processed = 0
    for _, managed in list(self.running.items()):
        token_cost = len(managed.prompt_ids) if not managed.has_been_prefilled else 1
        if tokens_processed and self.max_batched_tokens is not None and (
            tokens_processed + token_cost > self.max_batched_tokens
        ):
            break
        self._step_sequence(managed)
        tokens_processed += token_cost
    return bool(self.running or self.waiting_queue)
```

Two details worth calling out:

- **`max_batched_tokens`** caps how much work one `step()` call can do —
  a prefill costs as many "tokens" as the prompt is long, a decode step
  costs 1. This exists so a single very long prompt's prefill can't
  monopolize an entire iteration and starve other sequences waiting to be
  stepped; if the budget would be exceeded, remaining sequences simply
  wait for the *next* `step()` call instead.
- **Iterating over `list(self.running.items())`, not `self.running.items()`
  directly** — because `_step_sequence` can call `_evict`, which mutates
  `self.running` mid-loop. Iterating a snapshot avoids the classic
  "changed dict size during iteration" error.

**`run(max_steps=1000)`** — a simple synchronous loop that calls `step()`
repeatedly until it returns `False` (nothing running, nothing waiting) or
a safety cap is hit. Used directly by the benchmark scripts, where there's
no need for async concurrency.

---

## 5. `server/` — making it reachable over HTTP

### 5.1 `scheduler_service.py` — `SchedulerService`

**The problem this solves:** `Scheduler` is entirely synchronous — one
`step()` call blocks until that iteration's forward passes are done. A
real HTTP server needs to accept *concurrent* requests without spinning up
a separate scheduler (and separate model, separate KV-cache!) per request
— that would defeat the entire point of continuous batching.

`SchedulerService` is the async bridge: it owns exactly one `Scheduler`
and runs a single background `asyncio.Task` that calls `step()` in a
loop, forever, for the lifetime of the server:

```python
async def _run(self):
    while not self._stopping:
        has_work = self.scheduler.step()
        self._publish_generated_tokens()
        self._resolve_finished()
        await asyncio.sleep(0 if has_work else self.idle_sleep_seconds)
```

Individual HTTP requests never touch the scheduler directly. Instead:

- **`submit(prompt, max_new_tokens)`** creates an `asyncio.Future`, calls
  `scheduler.add_request(...)` to enqueue the work, and `await`s that
  future — which only resolves once the background loop notices this
  specific sequence has finished (`_resolve_finished`) and sets the
  result. Meanwhile, the *same* background loop is simultaneously
  stepping every other concurrent request through the exact same shared
  scheduler — this is what makes concurrent HTTP requests actually get
  continuously batched together, rather than each spinning up isolated
  work.
- **`submit_stream(...)`** does the same thing but also hands back a
  `GenerationStream` wrapping an `asyncio.Queue` — as each new token is
  generated for this sequence, `_publish_generated_tokens()` (called once
  per background loop iteration) pushes it into the queue, and the HTTP
  handler can `async for token in stream.tokens()` to stream results out
  as Server-Sent Events, token by token, as they're produced.
- **`idle_sleep_seconds`** — when `step()` returns `False` (truly nothing
  to do), the loop sleeps briefly instead of busy-spinning the CPU.

### 5.2 `api.py` — the FastAPI application

Thin HTTP plumbing on top of `SchedulerService`. The interesting logic
all lives below this layer — this file just translates HTTP requests into
`service.submit(...)` calls and formats the results as JSON or SSE.

**`lifespan(app)`** — runs once at server startup: reads `RuntimeSettings`
(from environment variables or CLI flags — see `config/runtime.py`),
resolves the actual device (`auto` → `cuda` if available, else `cpu`) and
dtype, creates the model runner via `ModelRegistry.create(...)`, builds a
`BlockManager` and `KVCache` sized from settings, wraps everything in one
`Scheduler` and one `SchedulerService`, and starts the background loop.
Everything is torn down cleanly in the `finally` block on shutdown.

**Routes:**
- `GET /health` — liveness check, also reports which model/device is loaded.
- `POST /generate` — submits a prompt, awaits the full result, returns it as JSON.
- `POST /generate/stream` — same, but streams tokens as SSE `token` events, ending with a `done` event.
- `GET /v1/models` — OpenAI-compatible model listing (just the one loaded model).
- `POST /v1/chat/completions` — OpenAI-compatible chat endpoint. Applies the tokenizer's real chat template if one exists (`tokenizer.apply_chat_template`), falling back to a simple `role: content` text format otherwise. Supports `stream: true` for standard OpenAI-style `data: {...}` SSE chunks ending in `data: [DONE]`.

**`main()`** is the actual entry point wired up in `pyproject.toml` as the
`pico-vllm` console command — parses CLI args into `RuntimeSettings` and
runs the server with `uvicorn`.

### 5.3 `config/runtime.py` — `RuntimeSettings`

A frozen dataclass holding every server-tunable setting (model ID, device,
dtype, KV-cache size/block-size, the scheduler's `max_batched_tokens`
budget, host/port). Every field can come from either an environment
variable (`PICO_VLLM_MODEL_ID`, etc.) or a matching CLI flag
(`--model-id`, etc.), with CLI flags taking precedence via
`from_cli()` defaulting to `from_env()`. `validate()` catches obviously
invalid combinations (zero/negative block counts, out-of-range ports)
before the server tries to start.

### 5.4 `config/config.py` — the static model config

Separate from `RuntimeSettings` — this is the older, simpler config used
as a *fallback* when a `KVCache` is constructed without an explicit
`ModelArchitecture` (e.g. in tests that don't go through the full server
startup path). Loaded once from a packaged `config.yaml`:

```yaml
total_capacity_in_tokens: 32000
num_tokens_per_block: 16
num_layers: 24
num_heads: 14
embed_dim: 896
num_kv_heads: 2
```

These are the real architecture numbers for `Qwen2.5-0.5B-Instruct`,
discovered by actually inspecting the loaded model's `k_proj`/`v_proj`
output sizes (this model uses GQA, so `num_heads` ≠ `num_kv_heads`) —
not guessed or copied from documentation.

---

## 6. `ui.py` — the Streamlit chat UI

A small standalone Streamlit app that talks to the running API server
over plain HTTP (defaults to `http://127.0.0.1:8000`, configurable from
the sidebar). It's a separate process from the server — run
`uv run streamlit run src/pico_vllm/ui.py` alongside `uv run pico-vllm`.
This file has no knowledge of scheduling, caching, or the model — it's
purely a client of the HTTP API described in §5.2.

---

## 7. How correctness was actually verified

Every layer of the system was checked against real Hugging Face model
output — not just assumed correct because the shapes lined up:

- **`KVCache`** — real tokens run through the actual model, their
  `past_key_values` written into `KVCache` via `write()`, read back via
  `read()`, and compared with `torch.allclose` against HF's original
  cache. This is also what caught the GQA head-count issue and a
  bfloat16/float32 dtype mismatch early on.
- **`Sequence`** — tested in isolation with no model at all: normal
  appends within one block, the block-boundary-crossing case (including
  the `ValueError` path when no new block is supplied), and a check that
  written data is genuinely retrievable afterward.
- **`paged_attention`** — verified using PyTorch forward hooks registered
  on the real model's `q_proj` (to capture the raw query) and `o_proj`
  (whose *input* is the real pre-projection attention output, i.e. ground
  truth). This test initially failed with a small but real mismatch,
  traced to the captured query being pre-RoPE while HF's cached keys
  (from `past_key_values`) are already post-RoPE — fixed by applying the
  same rotary embedding to the query before comparing.
- **Full multi-layer generation (`Qwen2Runner.forward_one_sequence`)** —
  compared per-step logits against HF's own forward pass run on identical
  input prefixes (rather than requiring two independently-generated
  greedy trajectories to match token-for-token, which is a stricter and
  less meaningful bar once floating-point noise enters the picture). In
  bfloat16, logits showed small, *consistent* drift rather than compounding
  error — diagnosed as floating-point precision rather than a logic bug,
  and confirmed by rerunning the same comparison in float32, where logits
  matched within tight tolerance.
- **Naive batching correctness** — confirmed that running several
  sequences sharing one `KVCache`/`BlockManager` produces identical
  output to running each sequence completely alone with its own private
  cache, ruling out cross-sequence data corruption from the shared tensor
  pool.

---

## 8. The benchmarks

### 8.1 `scripts/run_naive_vs_continuous.py` — eviction efficiency

Runs the same four prompts (three capped at 5 tokens, one at 50 —
deliberately mismatched, since matched lengths hide the difference)
through both `run_naive_batch` and `Scheduler`, instrumented to count
"wasted" forward-pass steps (steps run on a sequence that had already
finished).

Result: naive batching ran **135 wasted forward passes** against 65 real
tokens generated — more wasted computation than real computation — while
continuous batching ran zero, by design. Throughput: **2.47 → 6.93
tokens/sec (2.8x)**, same model, same hardware, same prompts.

### 8.2 `scripts/run_admission_under_pressure.py` — admission under memory pressure

Four requests, but `BlockManager` is only given 3 blocks — not enough for
all four to hold an initial block simultaneously. This exercises
`_try_admit()`, which the first benchmark never touched (it had enough
blocks for everyone upfront).

Result: three sequences admitted immediately at step 0; the fourth sat in
`waiting_queue` for 8 steps and was only admitted once an earlier sequence
finished and freed its block. All four completed successfully. Naive
batching has no concept of a waiting queue at all — it would fail outright
trying to allocate blocks for every sequence up front.

---