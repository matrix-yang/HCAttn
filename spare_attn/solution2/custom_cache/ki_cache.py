import torch
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple, Union


class FastCache(DynamicCache):
    def __init__(self, quanter):
        super().__init__()
        self.quanter = quanter
        self.key_cache = []
        self.value_cache = []
        self.vectors = quanter.vectors  # 缓存 vectors，避免重复获取
        # 创建用于异步传输的 CUDA 流
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        # 存储每个层的异步传输事件
        self.transfer_events = []
        # 存储每个层的异步量化结果
        self.quant_results = []
        # 存储每个层的处理状态
        self.processing_states = []

    def async_quant_and_transfer(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ) -> None:
        """
        第一部分：异步量化和转移至 CPU
        
        Args:
            key_states: 键状态张量
            value_states: 值状态张量
            layer_idx: 层索引
        """
        # 确保层索引对应的缓存存在
        if len(self.key_cache) <= layer_idx:
            # 填充缺失的层
            for _ in range(len(self.key_cache), layer_idx + 1):
                self.key_cache.append([])
                self.value_cache.append([])
                self.transfer_events.append(None)
                self.quant_results.append(None)
                self.processing_states.append(False)
        
        # 标记为处理中
        self.processing_states[layer_idx] = True
        
        # 异步量化
        with torch.cuda.stream(self.stream):
            key_index = self.quanter.quant(key_states)
            
            # 异步将 value_states 传输到 CPU
            value_cpu = torch.empty_like(value_states, device='cpu')
            value_cpu.copy_(value_states, non_blocking=True)
            
            # 记录结果
            event = torch.cuda.Event()
            event.record(self.stream)
            
            self.quant_results[layer_idx] = (key_index, value_cpu, event)

    def get_processed_kvcache(
            self,
            layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        第二部分：获得处理好的 KV cache
        
        Args:
            layer_idx: 层索引
        
        Returns:
            处理好的 key_cache, value_cache 和 vectors
        """
        # 检查是否正在处理
        if not self.processing_states[layer_idx]:
            # 如果没有处理，直接返回现有缓存
            if isinstance(self.key_cache[layer_idx], list):
                raise ValueError(f"No cache initialized for layer {layer_idx}")
            return self.key_cache[layer_idx], self.value_cache[layer_idx], self.vectors
        
        # 等待异步操作完成
        if self.quant_results[layer_idx] is not None:
            key_index, value_cpu, event = self.quant_results[layer_idx]
            event.synchronize()
        else:
            raise ValueError(f"No quant results for layer {layer_idx}")
        
        # 更新缓存
        if isinstance(self.key_cache[layer_idx], list):
            # 首次更新
            self.key_cache[layer_idx] = key_index
            self.value_cache[layer_idx] = value_cpu
        else:
            # 后续更新
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_index], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_cpu], dim=-2)
        
        # 标记为处理完成
        self.processing_states[layer_idx] = False
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.vectors

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        原始的 update 方法，保持兼容性
        
        Args:
            key_states: 键状态张量
            value_states: 值状态张量
            layer_idx: 层索引
            cache_kwargs: 缓存参数
        
        Returns:
            处理好的 key_cache, value_cache 和 vectors
        """
        # 调用第一部分：异步量化和转移
        self.async_quant_and_transfer(key_states, value_states, layer_idx)
        # 调用第二部分：获得处理好的 KV cache
        return self.get_processed_kvcache(layer_idx)
