from dataclasses import dataclass
import yaml


@dataclass
class Config:
    total_capacity_in_tokens: int
    num_tokens_per_block: int
    num_layers: int
    num_heads: int
    embed_dim: int
    num_kv_heads: int

    @property
    def num_blocks(self):
        return self.total_capacity_in_tokens // self.num_tokens_per_block

    @property
    def head_dim(self):
        return self.embed_dim // self.num_heads


with open("C:\\research-eng\\pico-vllm\\pico-vllm\\config\\config.yaml", "r") as f:
    data = yaml.safe_load(f)

config = Config(**data)