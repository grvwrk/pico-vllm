"""
benchmarks/run_admission_under_pressure.py

Demonstrates the OTHER half of continuous batching's value, which
run_naive_vs_continuous.py doesn't exercise: admission. That benchmark had
enough blocks for every sequence to start immediately, so the scheduler's
waiting_queue was never actually used.

Here, num_blocks is set deliberately low enough that not every sequence can
get an initial block at once. This forces some requests into
Scheduler.waiting_queue, and only admits them once an earlier sequence
finishes and frees its blocks — proving the scheduler can keep total
throughput moving under memory pressure that would otherwise stall
naive batching entirely (naive batching has no concept of "waiting" at
all — it would simply fail outright trying to allocate blocks for
everyone upfront).
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from block_manager import BlockManager
from kv_cache import KVCache
from sequence import Sequence
from model_wrapper import forward_one_sequence
from scheduler import Scheduler, WAITING, RUNNING, FINISHED

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 4 requests, but only enough blocks for 2 to be admitted at once.
PROMPTS = [
    ("The capital of France is", 8),
    ("Water boils at", 8),
    ("The sky is", 8),
    ("Once upon a time, in a land far away, there lived", 8),
]
NUM_BLOCKS = 3  # deliberately tight — only 2 sequences can hold an initial block at once


class InstrumentedScheduler(Scheduler):
    """
    Same as Scheduler, but records a timeline of admission events, so we
    can show WHEN each sequence actually started relative to others —
    proving late-admitted sequences started only after an earlier one
    freed its block, not all at once.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.admission_log = []  # list of (step_number, seq_id)
        self._step_counter = 0

    def _try_admit(self):
        before = set(self.running.keys())
        super()._try_admit()
        after = set(self.running.keys())
        newly_admitted = after - before
        for seq_id in newly_admitted:
            self.admission_log.append((self._step_counter, seq_id))

    def run(self, max_steps=1000):
        for step in range(max_steps):
            self._step_counter = step
            self._try_admit()

            if not self.running and not self.waiting_queue:
                break

            for seq_id, managed in list(self.running.items()):
                self._step_sequence(managed)

        return self.finished


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()

    bm = BlockManager(num_block=NUM_BLOCKS)
    kv_cache = KVCache()
    sched = InstrumentedScheduler(model, tokenizer, bm, kv_cache)

    for i, (prompt, max_new_tokens) in enumerate(PROMPTS):
        sched.add_request(f"seq_{i}", prompt, max_new_tokens=max_new_tokens)

    print(f"Running {len(PROMPTS)} requests through the scheduler with only "
          f"{NUM_BLOCKS} blocks available (not enough for all requests at once)...\n")

    start = time.time()
    finished = sched.run()
    elapsed = time.time() - start

    print("Admission timeline (step, seq_id) — shows WHEN each sequence was let in:")
    for step, seq_id in sched.admission_log:
        print(f"  step {step:>3}: {seq_id} admitted")

    print(f"\nAll {len(finished)} requests completed in {elapsed:.2f}s, "
          f"despite only {NUM_BLOCKS} blocks ever being available at once.")
    print("This would not be possible under naive batching, which has no "
          "concept of a waiting queue and would fail trying to allocate "
          "blocks for every sequence up front.")

    for seq_id, managed in finished.items():
        text = tokenizer.decode(managed.generated_ids, skip_special_tokens=True)
        print(f"  {seq_id}: {text!r}")


if __name__ == "__main__":
    main()