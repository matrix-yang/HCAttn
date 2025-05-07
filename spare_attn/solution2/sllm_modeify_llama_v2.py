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
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    # new data structure for past_key_value
    # past_key_value = (full_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)

    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    # bsz len head dim
    # print(key_states.shape)

    if past_key_value is not None:
        # reuse k, v, self_attention
        # transpose(1,2) 是因为是past_key_value存的时候transpose
        key_states = torch.cat([past_key_value[0].transpose(1, 2), key_states], dim=1)
        value_states = torch.cat([past_key_value[1].transpose(1, 2), value_states], dim=1)
        # torch.Size([1, 1344, 32, 128]) torch.Size([1, 1, 32, 128])
        # print(key_states.shape,query_states.shape)
    # decode
    if query_states.shape[1] == 1:
        #bsz, q_len, self.num_heads, self.head_dim
        sink = 64
        recent = int(attn_sum * kv_seq_len) + sink
        sllm_key_states=torch.cat([key_states[:,:sink,:,:], key_states[:,recent:,:,:]], dim=1)
        sllm_value_states = torch.cat([value_states[:,:sink,:,:], value_states[:,recent:,:,:]], dim=1)
        attn_output = flash_attn_func(
            query_states,
            sllm_key_states,
            sllm_value_states,
            causal=True,
            dropout_p=0.0,
        )
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    # transpose(1,2) 是因为是识别seq维度
    past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2)) if use_cache else None
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value
