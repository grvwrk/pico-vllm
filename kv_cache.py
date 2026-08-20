import torch
from config import config

key_cache = torch.zeros(
    config.num_layers,
    config.num_block,
    config.num_heads,
    config.num_tokens_per_block, 
    config.head_dim
)
value_cache = torch.zeros_like(key_cache)