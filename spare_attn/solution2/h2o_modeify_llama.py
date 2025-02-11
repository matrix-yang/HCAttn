from typing import Optional, Tuple
import os
import torch
from torch import nn
import types
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache, enable_tuple_kv_cache_for_llama
import numpy as np
import math

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    List,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from spare_attn.solution2.spare_decode import decode_token, decode_token_ori


def enable_llama_approx_attention_eval(
        model: LlamaForCausalLM, attn_sum, radio_bag
):
    enable_tuple_kv_cache_for_llama(model)
    new_forward = warp_forward(attn_sum, radio_bag)
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        module.forward = types.MethodType(
            new_forward, module
        )


def warp_forward(attn_sum, radio_bag):
    def new_func(self,
                 hidden_states: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 position_ids: Optional[torch.LongTensor] = None,
                 past_key_value: Optional[Tuple[torch.Tensor]] = None,
                 output_attentions: bool = False,
                 use_cache: bool = False,
                 **kwargs, ):
        return llama_approx_attention_forward(self,
                                              hidden_states,
                                              attention_mask,
                                              position_ids,
                                              past_key_value,
                                              output_attentions,
                                              use_cache,
                                              attn_sum=attn_sum,
                                              radio_bag=radio_bag,
                                              **kwargs)

    return new_func


def llama_approx_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        attn_sum=0.95,
        radio_bag=[],
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # print(self.config._attn_implementation)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print(past_key_value[0].shape,key_states.shape)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    kv_seq_len=key_states.shape[2]

    #print('qkv--',query_states.shape,key_states.shape,value_states.shape,self.head_dim)
    past_key_value = (key_states, value_states) if use_cache else None

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if q_len==kv_seq_len:
        attention_mask = torch.full((kv_seq_len, kv_seq_len), fill_value=torch.finfo(query_states.dtype).min, dtype=query_states.dtype,device=query_states.device)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        attention_mask = attention_mask[None, None, :, :].expand(bsz, 1, -1, -1)
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))


    heavy_budget_ratio=0.5
    recent_budget_ratio=0.5
    ### Heavy + Recent
    heavy_budget = int(heavy_budget_ratio * attn_weights.shape[-1])
    recent_budget = int(recent_budget_ratio * attn_weights.shape[-1])

    # # Heavy Hitter Mask (Based on local statistics)
    # if heavy_budget > 0:
    #     mask_bottom = local_heavy_hitter_mask(attn_weights, heavy_budget) # Default: No padding applied to input
    # else:
    #     mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)

    # ones = torch.ones_like(attn_weights, dtype=torch.bool)
    # ones = torch.triu(ones, diagonal=-recent_budget)
    # mask_bottom = torch.logical_or(mask_bottom, ones)

    # mask_bottom = torch.tril(mask_bottom, diagonal=0)

    # # mask_bottom = ones
    # attn_weights[~mask_bottom] = torch.min(attention_mask)

    # Heavy Hitter Mask (Based on global statistics)
    tmp_attn = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
    tmp_sum = torch.sum(tmp_attn, dim=-2)
    _, tmp_topk = tmp_sum.topk(k=heavy_budget, dim=-1)

    zeros = torch.zeros_like(tmp_sum, dtype=torch.bool)
    mask_bottom = zeros.scatter(-1, tmp_topk, True).unsqueeze(2)
    mask_bottom = mask_bottom.expand(mask_bottom.shape[0], mask_bottom.shape[1], attn_weights.shape[-2],
                                     mask_bottom.shape[-1])

    ones = torch.ones_like(attn_weights, dtype=torch.bool)
    ones = torch.tril(ones, diagonal=recent_budget)
    ones = torch.triu(ones, diagonal=-recent_budget)
    mask_bottom = torch.logical_or(mask_bottom, ones)
    # mask_bottom = ones
    attn_weights[~mask_bottom] = torch.finfo(attn_weights.dtype).min

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
