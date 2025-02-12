from typing import Optional, Tuple
import os
import torch
from torch import nn
import types
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache, enable_tuple_kv_cache_for_llama
import numpy as np
import math
import torch.nn.functional as F

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
        attn_sum=0.5,
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

    kv_seq_len = key_states.shape[2]

    ## old forward
    # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # attn_output = torch.matmul(attn_weights, value_states)
    #
    # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
    #     raise ValueError(
    #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
    #         f" {attn_output.size()}"
    #     )
    #
    # attn_output = attn_output.transpose(1, 2).contiguous()

    # fast attn
    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        causal=True,
        dropout_p=0.0,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    # 选cache

    past_key_value = [key_states, value_states] if use_cache else None

    attn_weights = cal_attn_weight(query_states, key_states, self.num_key_value_groups)
    # attn_weights = attn_weights_bag
    if q_len == kv_seq_len:
        cache_size = int(key_states.shape[-2] * 0.5)
        past_key_value = reduce_kv(past_key_value, attn_weights, cache_size)
    else:
        cache_size = key_states.shape[-2] - 1
        past_key_value = reduce_kv(past_key_value, attn_weights, cache_size)
    del attn_weights

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def reduce_kv(past_key_value, attn_weights, cache_size):
    bsz, _, _, num_keys = attn_weights.size()
    _, num_kv_heads, _, _ = past_key_value[0].size()
    # print('k--', past_key_value[0].size())
    # Utilize the average attention weights to select the top-k keys and values
    mean_attn_weights = torch.mean(attn_weights[:, :, -1, :], dim=1).clone().detach()
    # print('mean_attn_weights--', mean_attn_weights.size(),cache_size)
    vals, ind = torch.topk(mean_attn_weights, k=cache_size, dim=-1)
    ind = torch.sort(ind).values  # stabelizes some things for some reason
    expand_ind = ind.unsqueeze(1).unsqueeze(-1).expand(bsz, num_kv_heads, ind.size(-1),
                                                       past_key_value[0].size(-1))

    # Reduce the size of the cache to self.cache_size
    past_key_value[0] = torch.gather(past_key_value[0], dim=2, index=expand_ind)
    past_key_value[1] = torch.gather(past_key_value[1], dim=2, index=expand_ind)

    return past_key_value


def reduce_kv_with_sink(past_key_value_ori, attn_weights, cache_size):
    bsz, _, _, num_keys = attn_weights.size()
    sink_size = 4
    sink_k = past_key_value_ori[0][:, :, :sink_size, :]
    sink_v = past_key_value_ori[1][:, :, :sink_size, :]
    past_key_value = [past_key_value_ori[0][:, :, sink_size:, :], past_key_value_ori[1][:, :, sink_size:, :]]
    _, num_kv_heads, _, _ = past_key_value[0].size()
    # print('k--', past_key_value[0].size())
    # Utilize the average attention weights to select the top-k keys and values
    mean_attn_weights = torch.mean(attn_weights[:, :, -1, :], dim=1).clone().detach()
    # print('mean_attn_weights--', mean_attn_weights.size(),cache_size)
    vals, ind = torch.topk(mean_attn_weights[:,sink_size:], k=cache_size - sink_size, dim=-1)
    ind = torch.sort(ind).values  # stabelizes some things for some reason
    expand_ind = ind.unsqueeze(1).unsqueeze(-1).expand(bsz, num_kv_heads, ind.size(-1),
                                                       past_key_value[0].size(-1))

    # Reduce the size of the cache to self.cache_size
    past_key_value[0] = torch.gather(past_key_value[0], dim=2, index=expand_ind)
    past_key_value[1] = torch.gather(past_key_value[1], dim=2, index=expand_ind)
    new_past_key_value = [
        torch.cat([sink_k, past_key_value[0]], dim=2),
        torch.cat([sink_v, past_key_value[1]], dim=2)
    ]

    return new_past_key_value


# def low_cal_attn(query_states, key_states, num_key_value_groups):
#     key_states = repeat_kv(key_states, num_key_value_groups)
#     # query_states = torch.rand(1, 32, 45, 128)
#     # key_states = torch.rand(1, 32, 45, 128)
#     bsz = query_states.shape[0]
#     head = query_states.shape[1]
#     q_len = query_states.shape[-2]
#     kv_seq_len = key_states.shape[-2]
#     attn = torch.zeros((bsz, head, q_len, kv_seq_len), dtype=key_states.dtype, device=key_states.device)
#     for i in range(0, head, 4):
#         s = i * 4
#         e = s + 4
#         q = query_states[:, s:e, :, :]
#         k = key_states[:, s:e, :, :]
#         attn[:, s:e, :, :] = cal_attn_weight(q, k)
#     return attn


def cal_attn_weight(query_states, key_states, num_key_value_groups):
    key_states = repeat_kv(key_states, num_key_value_groups)
    bsz = query_states.shape[0]
    q_len = 1
    query_states = query_states[:, :, -1, :]
    # kv_seq_len = key_states.shape[-2]

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(128)
    #
    # if q_len == kv_seq_len:
    #     attention_mask = torch.full((kv_seq_len, kv_seq_len), fill_value=torch.finfo(key_states.dtype).min,
    #                                 dtype=key_states.dtype, device=key_states.device)
    #     attention_mask = torch.triu(attention_mask, diagonal=1)
    #
    #     attention_mask = attention_mask[None, None, :, :].expand(bsz, 1, -1, -1)
    #     if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
    #         raise ValueError(
    #             f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
    #         )
    #     attn_weights = attn_weights + attention_mask
    #     del attention_mask
    # upcast attention to fp32
    # attn_weights_bag = torch.zeros_like(attn_weights)
    # step = 1000
    # for i in range(0, attn_weights.shape[-2], step):
    #     s = i * step
    #     e = min(attn_weights.shape[-2], s + step)
    #     taw = attn_weights[:, :, s:e,: ]
    #     attn_weights_bag[:, :, s:e, :] = nn.functional.softmax(taw, dim=-1, dtype=torch.bfloat16).to(key_states.dtype)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(key_states.dtype)
    return attn_weights
