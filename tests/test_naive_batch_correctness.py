"""
tests/test_naive_batch_correctness.py

Verifies that run_naive_batch's shared KVCache/BlockManager across multiple
concurrent sequences doesn't corrupt any individual sequence's output —
i.e., generating prompt A alongside prompt B in the same batch produces
EXACTLY the same tokens as generating prompt A completely alone, with its
own fresh KVCache and BlockManager.

This is the one thing single-sequence tests structurally cannot catch:
whether sequences sharing one physical tensor pool interfere with each
other's data.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence
from pico_vllm.engine.model_wrapper import forward_one_sequence
from pico_vllm.engine.naive_batching import run_naive_batch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    ("The capital of France is", 5),
    ("Water boils at", 5),
    ("Once upon a time, in a land far away, there lived", 8),
]


def run_single_sequence_alone(model, tokenizer, prompt, max_new_tokens):
    """Generates from ONE prompt using a completely fresh, private
    KVCache/BlockManager — this is the already-proven-correct path,
    used here as ground truth."""
    bm = BlockManager(num_block=32)
    kv_cache = KVCache()

    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0]

    initial_block = bm.allocate_block("solo_seq")
    seq = Sequence(seq_id="solo_seq", initial_block_index=initial_block)

    generated_ids = []

    with torch.no_grad():
        logits = forward_one_sequence(
            model, kv_cache, bm, seq,
            input_ids=prompt_ids, is_prefill=True,
        )
        next_token = torch.argmax(logits).item()
        generated_ids.append(next_token)

        for _ in range(max_new_tokens - 1):
            if next_token == tokenizer.eos_token_id:
                break
            next_input = torch.tensor([next_token])
            logits = forward_one_sequence(
                model, kv_cache, bm, seq,
                input_ids=next_input, is_prefill=False,
            )
            next_token = torch.argmax(logits).item()
            generated_ids.append(next_token)

    return generated_ids


def test_naive_batch_matches_isolated_generation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    prompts_only = [p for p, _ in PROMPTS]
    max_new_tokens_list = [m for _, m in PROMPTS]

    # ---- Run all prompts together in one naive batch ----
    batched_result = run_naive_batch(
        model, tokenizer, prompts_only,
        max_new_tokens=max(max_new_tokens_list),
        num_blocks=64,
    )

    # ---- Run each prompt completely alone, as ground truth ----
    mismatches = []
    for i, (prompt, max_new_tokens) in enumerate(PROMPTS):
        seq_id = f"seq_{i}"
        solo_ids = run_single_sequence_alone(model, tokenizer, prompt, max_new_tokens)

        # Batched result may have generated MORE tokens than max_new_tokens
        # for this specific prompt (since run_naive_batch shares one
        # max_new_tokens across the whole batch in this simple version) —
        # only compare the first `max_new_tokens` positions, which is what
        # this specific prompt "should" have produced.
        batched_ids = batched_result[seq_id]["generated_ids"][:max_new_tokens]

        if batched_ids != solo_ids:
            mismatches.append({
                "seq_id": seq_id,
                "prompt": prompt,
                "solo": solo_ids,
                "batched": batched_ids,
            })

    if mismatches:
        for m in mismatches:
            print(f"\nMismatch for {m['seq_id']} ({m['prompt']!r}):")
            print(f"  solo:    {m['solo']}")
            print(f"  batched: {m['batched']}")

    assert not mismatches, f"{len(mismatches)} sequence(s) diverged between solo and batched generation"

    print("Naive batching matches isolated single-sequence generation for all prompts.")


if __name__ == "__main__":
    test_naive_batch_matches_isolated_generation()
