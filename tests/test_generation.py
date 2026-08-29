import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.config import config
from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence
from pico_vllm.engine.model_wrapper import forward_one_sequence

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TEXT = "The capital of France is"
NUM_STEPS = 5  # prefill + this many decode steps compared


def test_generation_logits_match_hf():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    inputs = tokenizer(TEXT, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0]  # (prompt_len,)
    prompt_len = prompt_ids.shape[0]

    # ---- Our pipeline: prefill + token-by-token decode ----
    bm = BlockManager(num_block=16)
    kv_cache = KVCache()

    initial_block = bm.allocate_block("gen_seq")
    seq = Sequence(seq_id="gen_seq", initial_block_index=initial_block)

    our_logits_per_step = []
    our_generated_ids = []

    with torch.no_grad():
        # Prefill: process the whole prompt at once
        logits = forward_one_sequence(
            model, kv_cache, bm, seq,
            input_ids=prompt_ids, is_prefill=True,
        )
        our_logits_per_step.append(logits)
        next_token = torch.argmax(logits).item()
        our_generated_ids.append(next_token)

        # Decode: one new token at a time, greedily following OUR OWN predictions
        # (not HF's), so both trajectories are compared at the same input at each step.
        for _ in range(NUM_STEPS - 1):
            next_input = torch.tensor([next_token])
            logits = forward_one_sequence(
                model, kv_cache, bm, seq,
                input_ids=next_input, is_prefill=False,
            )
            our_logits_per_step.append(logits)
            next_token = torch.argmax(logits).item()
            our_generated_ids.append(next_token)

    print("Ours generated:", our_generated_ids, tokenizer.decode(our_generated_ids))

    # ---- Ground truth: run HF forward pass at each of the SAME input prefixes
    # our pipeline actually saw, so we're comparing logits step-by-step on
    # identical inputs rather than requiring identical multi-step trajectories ----
    full_sequence = torch.cat([prompt_ids, torch.tensor(our_generated_ids)])

    mismatches = []
    with torch.no_grad():
        for step in range(NUM_STEPS):
            prefix_len = prompt_len + step
            hf_logits = model(
                full_sequence[:prefix_len].unsqueeze(0), use_cache=False
            ).logits[0, -1, :]

            close = torch.allclose(our_logits_per_step[step], hf_logits, atol=1e-1, rtol=1e-2)
            if not close:
                diff = (our_logits_per_step[step] - hf_logits).abs().max().item()
                mismatches.append((step, diff))

    if mismatches:
        print("Mismatches (step, max abs diff):", mismatches)

    assert not mismatches, (
        f"Logits diverged beyond tolerance at steps: {mismatches}"
    )

    print("Per-step logits match real HF forward pass within tolerance across all steps.")


if __name__ == "__main__":
    test_generation_logits_match_hf()
