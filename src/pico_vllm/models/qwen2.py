"""Explicit Qwen2 forward-pass implementation for pico-vLLM."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pico_vllm.core.paged_attention import paged_attention
from pico_vllm.models.base import ModelArchitecture, ModelRunner


class Qwen2Runner(ModelRunner):
    """Runner for Qwen2's decoder layout, RoPE, GQA, and MLP blocks."""

    def __init__(self, model_id: str, model, tokenizer):
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        config = model.config
        self.architecture = ModelArchitecture(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            hidden_size=config.hidden_size,
            num_kv_heads=config.num_key_value_heads,
        )

    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str, dtype: torch.dtype) -> "Qwen2Runner":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        model.eval()
        return cls(model_id, model, tokenizer)

    def _apply_rope(self, x, seq_len, position_ids):
        dummy_hidden = torch.zeros(
            1, seq_len, self.model.config.hidden_size, dtype=x.dtype, device=x.device
        )
        cos, sin = self.model.model.rotary_emb(dummy_hidden, position_ids)

        def rotate_half(t):
            first, second = t.chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

        cos = cos.squeeze(0).unsqueeze(0)
        sin = sin.squeeze(0).unsqueeze(0)
        return (x * cos) + (rotate_half(x) * sin)

    def forward_one_sequence(self, kv_cache, block_manager, sequence, input_ids, *, is_prefill: bool):
        """Run the Qwen2 layers using pico-vLLM's paged KV-cache attention."""
        seq_len = input_ids.shape[0]
        position_offset = sequence.seq_len
        hidden_states = self.model.model.embed_tokens(input_ids).unsqueeze(0)
        position_ids = torch.arange(
            position_offset, position_offset + seq_len, device=input_ids.device
        ).unsqueeze(0)
        architecture = self.architecture

        for layer_idx, layer in enumerate(self.model.model.layers):
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            q = layer.self_attn.q_proj(normed)
            k = layer.self_attn.k_proj(normed)
            v = layer.self_attn.v_proj(normed)

            q = q.view(seq_len, architecture.num_heads, architecture.head_dim).transpose(0, 1)
            k = k.view(seq_len, architecture.num_kv_heads, architecture.head_dim).transpose(0, 1)
            v = v.view(seq_len, architecture.num_kv_heads, architecture.head_dim).transpose(0, 1)
            q = self._apply_rope(q, seq_len, position_ids)
            k = self._apply_rope(k, seq_len, position_ids)

            for token_index in range(seq_len):
                new_block_index = None
                if layer_idx == 0 and sequence.seq_len > 0 and sequence.seq_len % kv_cache.block_size == 0:
                    new_block_index = block_manager.allocate_block(sequence.seq_id)

                if layer_idx == 0:
                    sequence.append_token(
                        k[:, token_index, :], v[:, token_index, :], kv_cache,
                        layer=layer_idx, new_block_index=new_block_index,
                    )
                else:
                    cache_position = sequence.seq_len - seq_len + token_index
                    block_index = sequence.block_indices[cache_position // kv_cache.block_size]
                    kv_cache.write(
                        layer_idx, block_index, cache_position % kv_cache.block_size,
                        k[:, token_index, :], v[:, token_index, :],
                    )

            attention = paged_attention(q, kv_cache, layer=layer_idx, seq=sequence, is_prefill=is_prefill)
            attention = attention.transpose(0, 1).reshape(
                1, seq_len, architecture.num_heads * architecture.head_dim
            )
            hidden_states = residual + layer.self_attn.o_proj(attention)
            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.model.model.norm(hidden_states)
        return self.model.lm_head(hidden_states)[0, -1, :]
