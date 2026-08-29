"""
benchmarks/run_naive_vs_continuous.py

Compares naive static batching against the continuous batching Scheduler
on the SAME set of prompts, using deliberately mismatched generation
lengths (some short, some long) so the difference actually shows up.

Naive batching's flaw: once a short sequence finishes, it keeps getting
stepped anyway (generating discarded tokens) until the longest sequence
in the batch finishes too. Continuous batching evicts finished sequences
immediately and admits new work into the freed room.

Metrics reported:
  - total wall-clock time
  - total REAL tokens generated (tokens that were actually wanted)
  - total forward-pass steps executed per sequence (naive runs extra,
    wasted steps on already-finished sequences; continuous does not)
  - effective throughput (real tokens / second)
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence
from pico_vllm.engine.model_wrapper import forward_one_sequence
from pico_vllm.engine.scheduler import Scheduler

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Deliberately mismatched lengths: 3 short, 1 long — this is what exposes
# naive batching's wasted-compute problem. If all lengths matched, naive
# and continuous would look identical.
PROMPTS = [
    ("The capital of France is", 5),
    ("Water boils at", 5),
    ("The sky is", 5),
    ("Once upon a time, in a land far away, there lived", 50),
]


def run_naive_batch_instrumented(model, tokenizer, prompts, num_blocks=64):
    """
    Same logic as run_naive_batch, but also tracks how many forward-pass
    steps were run on already-finished sequences (wasted compute) — the
    number this whole benchmark exists to surface.
    """
    bm = BlockManager(num_block=num_blocks)
    kv_cache = KVCache()

    sequences = {}
    for i, (prompt, max_new_tokens) in enumerate(prompts):
        seq_id = f"seq_{i}"
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0]
        initial_block = bm.allocate_block(seq_id)
        seq = Sequence(seq_id=seq_id, initial_block_index=initial_block)
        sequences[seq_id] = {
            "seq": seq,
            "prompt_ids": prompt_ids,
            "max_new_tokens": max_new_tokens,
            "generated_ids": [],
            "finished": False,
            "real_tokens": 0,    # tokens generated while NOT yet finished
            "wasted_steps": 0,   # steps run AFTER this sequence had already finished
        }

    max_steps_needed = max(mnt for _, mnt in prompts)

    with torch.no_grad():
        for seq_id, entry in sequences.items():
            logits = forward_one_sequence(
                model, kv_cache, bm, entry["seq"],
                input_ids=entry["prompt_ids"], is_prefill=True,
            )
            next_token = torch.argmax(logits).item()
            entry["generated_ids"].append(next_token)
            entry["real_tokens"] += 1
            if next_token == tokenizer.eos_token_id or len(entry["generated_ids"]) >= entry["max_new_tokens"]:
                entry["finished"] = True

        for step in range(max_steps_needed - 1):
            for seq_id, entry in sequences.items():
                if entry["finished"]:
                    # naive batching keeps stepping this sequence anyway —
                    # this is the wasted compute we're measuring
                    entry["wasted_steps"] += 1

                last_token = entry["generated_ids"][-1]
                next_input = torch.tensor([last_token])
                logits = forward_one_sequence(
                    model, kv_cache, bm, entry["seq"],
                    input_ids=next_input, is_prefill=False,
                )
                next_token = torch.argmax(logits).item()
                entry["generated_ids"].append(next_token)

                if not entry["finished"]:
                    entry["real_tokens"] += 1
                    if next_token == tokenizer.eos_token_id or len(entry["generated_ids"]) >= entry["max_new_tokens"]:
                        entry["finished"] = True

    return sequences


def run_continuous_batch_instrumented(model, tokenizer, prompts, num_blocks=64):
    bm = BlockManager(num_block=num_blocks)
    kv_cache = KVCache()
    sched = Scheduler(model, tokenizer, bm, kv_cache)

    for i, (prompt, max_new_tokens) in enumerate(prompts):
        sched.add_request(f"seq_{i}", prompt, max_new_tokens=max_new_tokens)

    finished = sched.run()
    return finished


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    print("Running naive static batching...")
    start = time.time()
    naive_result = run_naive_batch_instrumented(model, tokenizer, PROMPTS)
    naive_time = time.time() - start

    naive_real_tokens = sum(e["real_tokens"] for e in naive_result.values())
    naive_wasted_steps = sum(e["wasted_steps"] for e in naive_result.values())

    print("Running continuous batching...")
    start = time.time()
    continuous_result = run_continuous_batch_instrumented(model, tokenizer, PROMPTS)
    continuous_time = time.time() - start

    continuous_real_tokens = sum(len(m.generated_ids) for m in continuous_result.values())

    print("\n" + "=" * 60)
    print(f"{'Metric':<35}{'Naive':<15}{'Continuous':<15}")
    print("=" * 60)
    print(f"{'Total wall-clock time (s)':<35}{naive_time:<15.2f}{continuous_time:<15.2f}")
    print(f"{'Real tokens generated':<35}{naive_real_tokens:<15}{continuous_real_tokens:<15}")
    print(f"{'Wasted forward-pass steps':<35}{naive_wasted_steps:<15}{'0 (by design)':<15}")
    print(f"{'Throughput (real tokens/sec)':<35}"
          f"{naive_real_tokens / naive_time:<15.2f}"
          f"{continuous_real_tokens / continuous_time:<15.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
