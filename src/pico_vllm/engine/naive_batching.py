import torch 

from pico_vllm.core.block_manager import BlockManager
from pico_vllm.core.kv_cache import KVCache
from pico_vllm.core.sequence import Sequence
from pico_vllm.engine.model_wrapper import forward_one_sequence

def run_naive_batch(model, tokenizer, prompts, max_new_tokens, num_blocks=64):
    bm = BlockManager(num_block=num_blocks)
    kv_cache = KVCache()

    # 1. Set up one Sequence per prompt, tokenize each prompt
    sequences = {}  # seq_id -> {"seq": Sequence, "generated_ids": [...], "finished": bool}
    for i, prompt in enumerate(prompts):
        seq_id = f"seq_{i}"

        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_ids = inputs["input_ids"][0]

        initial_block = bm.allocate_block(seq_id)
        seq = Sequence(seq_id=seq_id, initial_block_index=initial_block)

        sequences[seq_id] = {
            "seq": seq,
            "prompt_ids": prompt_ids,
            "generated_ids": [],
            "finished": False,
        }
    
    # 2. Prefill: run forward_one_sequence once per sequence with is_prefill=True
    with torch.no_grad():
        for seq_id, entry in sequences.items():
            logits = forward_one_sequence(
                model, kv_cache, bm, entry["seq"],
                input_ids=entry["prompt_ids"], is_prefill=True,
            )
            next_token = torch.argmax(logits).item()
            entry["generated_ids"].append(next_token)
            if next_token == tokenizer.eos_token_id:
                entry["finished"] = True

    # 3. Decode loop: keep calling forward_one_sequence for EVERY sequence
    #    (even finished ones — that's the naive/wasteful part) until ALL are finished
    #    or max_new_tokens is reached
    with torch.no_grad():
        for step in range(max_new_tokens - 1):
            if all(entry["finished"] for entry in sequences.values()):
                break

            for seq_id, entry in sequences.items():
                # naive batching: keep stepping this sequence even if it already
                # finished — a real system would skip it, but that's exactly the
                # wasted compute this baseline is meant to demonstrate
                last_token = entry["generated_ids"][-1]
                next_input = torch.tensor([last_token])

                logits = forward_one_sequence(
                    model, kv_cache, bm, entry["seq"],
                    input_ids=next_input, is_prefill=False,
                )
                next_token = torch.argmax(logits).item()
                entry["generated_ids"].append(next_token)

                if next_token == tokenizer.eos_token_id:
                    entry["finished"] = True

    return sequences
