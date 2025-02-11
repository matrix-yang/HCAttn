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

    kv_seq_len = key_states.shape[2]

    # print('qkv--',query_states.shape,key_states.shape,value_states.shape,self.head_dim)

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    #
    # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
    #     raise ValueError(
    #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
    #         f" {attn_weights.size()}"
    #     )

    sink = 4
    recent = int(0.5 * kv_seq_len) - 4
    if q_len == kv_seq_len:

        attention_mask = torch.full((kv_seq_len, kv_seq_len), fill_value=torch.finfo(query_states.dtype).min,
                                    dtype=query_states.dtype, device=query_states.device)
        if recent > kv_seq_len:
            attention_mask = torch.triu(attention_mask, diagonal=1) + torch.tril(attention_mask, diagonal=-recent)
            attention_mask[sink:, :sink] = 0
        else:
            attention_mask = torch.triu(attention_mask, diagonal=1)

        attention_mask = attention_mask[None, None, :, :].expand(bsz, 1, -1, -1)
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
    if query_states.shape[1] == 1:
        # decode 删除掉sink 后的第一个token
        cal_key_states = torch.cat([key_states[:, :, :sink, :], key_states[:, :, -recent:, :]], dim=2)
        cal_value_states = torch.cat([value_states[:, :, :sink, :], value_states[:, :, -recent:, :]], dim=2)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            cal_key_states,
            cal_value_states,
            dropout_p=0.0,
            scale=0.088388
        )
    else:
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout_p=0.0,
            scale=0.088388
        )
        # prefill 完毕之后删除多余的kv
        # key_states = torch.cat([key_states[:, :, :sink, :], key_states[:, :, -recent:, :]], dim=2)
        # value_states = torch.cat([value_states[:, :, :sink, :], value_states[:, :, -recent:, :]], dim=2)

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    past_key_value = (key_states, value_states) if use_cache else None
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
