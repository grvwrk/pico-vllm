"""
scheduler.py

Implements iteration-level continuous batching, as opposed to the naive
static batching in run_naive_batch(). The key difference: instead of
running a fixed batch until EVERY sequence finishes, this scheduler
re-evaluates who's running BEFORE every single forward-pass step —
evicting finished sequences immediately (freeing their blocks) and
admitting new waiting sequences into the freed-up room, so a GPU slot
never sits idle just because one sequence in the batch happens to be done.
"""

import torch

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence
from pico_vllm.engine.model_wrapper import forward_one_sequence


# ---- Sequence states ----
# WAITING  -> hasn't been admitted yet, sitting in the queue
# RUNNING  -> currently active, gets stepped every iteration
# FINISHED -> hit EOS or max length, blocks have been freed, no longer stepped
WAITING = "waiting"
RUNNING = "running"
FINISHED = "finished"


class ManagedSequence:
    """
    Wraps a Sequence with the extra bookkeeping the scheduler needs on top
    of what Sequence itself tracks (which is only cache/block state).
    """
    def __init__(self, seq_id, prompt_ids, max_new_tokens):
        self.seq_id = seq_id
        self.prompt_ids = prompt_ids          # tokenized prompt, not yet prefilled
        self.max_new_tokens = max_new_tokens
        self.generated_ids = []
        self.state = WAITING
        self.seq = None                       # real Sequence object, created at admission time
        self.has_been_prefilled = False        # tracks whether prefill has run yet


class Scheduler:
    def __init__(
        self, model, tokenizer, block_manager: BlockManager, kv_cache: KVCache,
        *, max_batched_tokens: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.bm = block_manager
        self.kv_cache = kv_cache
        self.max_batched_tokens = max_batched_tokens
        self.device = next(model.parameters()).device

        self.waiting_queue = []   # list of ManagedSequence, not yet admitted
        self.running = {}         # seq_id -> ManagedSequence, currently active
        self.finished = {}        # seq_id -> ManagedSequence, done

    def add_request(self, seq_id, prompt, max_new_tokens):
        """Add a new request to the waiting queue. Doesn't allocate anything yet —
        admission (and block allocation) only happens once there's room."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0].to(self.device)
        managed = ManagedSequence(seq_id, prompt_ids, max_new_tokens)
        self.waiting_queue.append(managed)

    def _evict(self, managed: ManagedSequence):
        """Called when a sequence finishes: free its blocks back to the shared
        pool immediately, so the very next step can admit someone else into
        that freed space. This immediate reclaim is the core of continuous
        batching — naive batching would keep the slot 'occupied' until the
        whole batch ends, even though nothing useful is happening in it."""
        self.bm.free_blocks(managed.seq_id)
        managed.state = FINISHED
        self.finished[managed.seq_id] = managed
        del self.running[managed.seq_id]

    def _try_admit(self):
        """Pull sequences from the waiting queue into the running set, as long
        as there's at least one free block available. We only need enough
        room for the FIRST token here — like any other sequence, an admitted
        one will request more blocks later via the normal
        Sequence.append_token() -> ValueError -> allocate flow, handled the
        same way as everyone else, not specially."""
        while self.waiting_queue and self.bm.free_block_pool:
            managed = self.waiting_queue.pop(0)  # FCFS: first-come-first-served

            initial_block = self.bm.allocate_block(managed.seq_id)
            managed.seq = Sequence(seq_id=managed.seq_id, initial_block_index=initial_block)
            managed.state = RUNNING
            self.running[managed.seq_id] = managed

    def _step_sequence(self, managed: ManagedSequence):
        """Run exactly one forward pass for one sequence — either its prefill
        (first call) or a single decode step (every call after)."""
        with torch.no_grad():
            if not managed.has_been_prefilled:
                # First time this sequence runs: process the whole prompt at once
                logits = forward_one_sequence(
                    self.model, self.kv_cache, self.bm, managed.seq,
                    input_ids=managed.prompt_ids, is_prefill=True,
                )
                managed.has_been_prefilled = True
            else:
                # Every call after: decode exactly one new token, using the
                # last generated token as input
                last_token = managed.generated_ids[-1]
                next_input = torch.tensor([last_token], device=self.device)
                logits = forward_one_sequence(
                    self.model, self.kv_cache, self.bm, managed.seq,
                    input_ids=next_input, is_prefill=False,
                )

        next_token = torch.argmax(logits).item()
        managed.generated_ids.append(next_token)

        # Check finishing conditions: EOS token, or hit this sequence's own
        # max_new_tokens (each sequence can have a different limit — unlike
        # naive batching, which was locked to one shared max_new_tokens for
        # the whole batch)
        hit_eos = next_token == self.tokenizer.eos_token_id
        hit_max_len = len(managed.generated_ids) >= managed.max_new_tokens
        if hit_eos or hit_max_len:
            self._evict(managed)

    def step(self) -> bool:
        """Advance the scheduler by one iteration.

        This is the unit of work used by both the synchronous benchmark
        runner and the long-lived HTTP scheduler. It admits newly queued
        requests before stepping every active request once.

        Returns ``True`` when work remains after this iteration. A caller
        with a background loop can use this to avoid a busy loop while idle.
        """
        self._try_admit()

        if not self.running:
            return bool(self.waiting_queue)

        # Work from a snapshot because a completed sequence removes itself
        # from ``running`` during _step_sequence().
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

    def run(self, max_steps=1000):
        """
        Main iteration loop. Each iteration:
          1. Try to admit waiting sequences into any free room
          2. Step every currently-running sequence exactly once
        Repeats until nothing is running and nothing is waiting, or max_steps
        is hit as a safety cap (in case of a bug causing an infinite loop).

        Note the key difference from naive batching: step 1 happens EVERY
        iteration, not just once at the start — so a sequence that finishes
        on step 5 can free its blocks and let a brand new waiting sequence
        start on step 6, rather than everyone waiting for the slowest
        sequence in an original fixed batch to finish.
        """
        for _ in range(max_steps):
            if not self.step():
                break

        return self.finished
