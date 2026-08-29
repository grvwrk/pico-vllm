import json

import pytest

from pico_vllm.config import RuntimeSettings
from pico_vllm.server.api import ChatMessage, _chat_prompt, _openai_sse


def test_runtime_settings_accept_cli_overrides():
    settings = RuntimeSettings.from_cli([
        "--model-id", "Qwen/Qwen2.5-0.5B-Instruct",
        "--device", "cpu",
        "--dtype", "float32",
        "--kv-cache-blocks", "8",
        "--kv-cache-block-size", "4",
        "--max-batched-tokens", "32",
        "--host", "0.0.0.0",
        "--port", "9000",
    ])

    assert settings.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert settings.kv_cache_blocks == 8
    assert settings.kv_cache_block_size == 4
    assert settings.max_batched_tokens == 32
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000


def test_runtime_settings_rejects_unsupported_model():
    with pytest.raises(ValueError, match="currently supports only"):
        RuntimeSettings(model_id="meta-llama/Llama-3.2-1B-Instruct").validate()


def test_openai_helpers_use_chat_template_and_valid_sse():
    class Tokenizer:
        chat_template = "available"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return messages[0]["content"] + " prompt"

    prompt = _chat_prompt(Tokenizer(), [ChatMessage(role="user", content="Hello")])
    event = _openai_sse("chatcmpl-id", 123, "model", {"content": "Hi"})

    assert prompt == "Hello prompt"
    payload = json.loads(event.removeprefix("data: ").strip())
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"] == {"content": "Hi"}
