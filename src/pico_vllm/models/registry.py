"""Registry for model IDs supported by pico-vLLM runners."""

from __future__ import annotations

from pico_vllm.models.qwen2 import Qwen2Runner


class ModelRegistry:
    _runners = {"Qwen/Qwen2.5-0.5B-Instruct": Qwen2Runner}

    @classmethod
    def supported_model_ids(cls) -> tuple[str, ...]:
        return tuple(cls._runners)

    @classmethod
    def create(cls, model_id: str, *, device: str, dtype):
        try:
            runner_type = cls._runners[model_id]
        except KeyError as error:
            supported = ", ".join(repr(item) for item in cls.supported_model_ids())
            raise ValueError(
                f"Unsupported model {model_id!r}. Install or implement a ModelRunner for it. "
                f"Currently supported: {supported}."
            ) from error
        return runner_type.from_pretrained(model_id, device=device, dtype=dtype)
