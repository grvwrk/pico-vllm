"""Backward-compatible import for the Qwen2 runner's explicit forward pass.

New code should use ``pico_vllm.models.ModelRunner`` and ``ModelRegistry``.
"""

from pico_vllm.models.qwen2 import Qwen2Runner


def forward_one_sequence(model, kv_cache, block_manager, seq, input_ids, is_prefill: bool):
    """Compatibility shim for existing benchmarks and tests.

    The scheduler no longer uses this Qwen-specific function directly.
    """
    runner = Qwen2Runner("Qwen/Qwen2.5-0.5B-Instruct", model, tokenizer=None)
    return runner.forward_one_sequence(
        kv_cache, block_manager, seq, input_ids, is_prefill=is_prefill
    )
