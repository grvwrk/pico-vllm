"""Runtime configuration for the local inference server."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch



def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class RuntimeSettings:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"
    dtype: str = "float32"
    kv_cache_blocks: int = 64
    kv_cache_block_size: int = 16
    max_batched_tokens: int = 512
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            model_id=os.getenv("PICO_VLLM_MODEL_ID", cls.model_id),
            device=os.getenv("PICO_VLLM_DEVICE", cls.device),
            dtype=os.getenv("PICO_VLLM_DTYPE", cls.dtype),
            kv_cache_blocks=_env_int("PICO_VLLM_KV_CACHE_BLOCKS", cls.kv_cache_blocks),
            kv_cache_block_size=_env_int("PICO_VLLM_KV_CACHE_BLOCK_SIZE", cls.kv_cache_block_size),
            max_batched_tokens=_env_int("PICO_VLLM_MAX_BATCHED_TOKENS", cls.max_batched_tokens),
            host=os.getenv("PICO_VLLM_HOST", cls.host),
            port=_env_int("PICO_VLLM_PORT", cls.port),
        )

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "RuntimeSettings":
        defaults = cls.from_env()
        parser = argparse.ArgumentParser(description="Run the pico-vLLM API server.")
        parser.add_argument("--model-id", default=defaults.model_id)
        parser.add_argument("--device", default=defaults.device, choices=["auto", "cpu", "cuda"])
        parser.add_argument("--dtype", default=defaults.dtype, choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--kv-cache-blocks", type=int, default=defaults.kv_cache_blocks)
        parser.add_argument("--kv-cache-block-size", type=int, default=defaults.kv_cache_block_size)
        parser.add_argument("--max-batched-tokens", type=int, default=defaults.max_batched_tokens)
        parser.add_argument("--host", default=defaults.host)
        parser.add_argument("--port", type=int, default=defaults.port)
        args = parser.parse_args(argv)
        settings = cls(**vars(args))
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.kv_cache_blocks < 1 or self.kv_cache_block_size < 1:
            raise ValueError("KV-cache block count and block size must both be positive.")
        if self.max_batched_tokens < 1:
            raise ValueError("max_batched_tokens must be positive.")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535.")

    def resolved_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PICO_VLLM_DEVICE=cuda was requested, but CUDA is unavailable.")
        return self.device

    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]
