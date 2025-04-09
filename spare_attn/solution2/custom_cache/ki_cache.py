import torch
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple, Union


class FastCache(DynamicCache):
    def __init__(self, quanter):
        super().__init__()
        self.quanter = quanter

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            key_index = self.quanter.quant(key_states)
            self.key_cache.append(key_index)
            self.value_cache.append(value_states.cpu())
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        else:
            key_index = self.quanter.quant(key_states)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_index], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states.cpu()], dim=-2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx],self.quanter.vectors
