from typing import Optional, Tuple
import os
import torch
from torch import nn
import types
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache, enable_tuple_kv_cache_for_llama
import numpy as np

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    List,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
#from flash_attn import flash_attn_func, flash_attn_with_kvcache
from spare_attn.solution2.spare_decode import decode_token, decode_token_ori
from flash_attn import flash_attn_func, flash_attn_with_kvcache

def enable_qwen_approx_attention_eval(
        model: Qwen2ForCausalLM, attn_sum, radio_bag
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
                 cache_position: Optional[torch.LongTensor] = None,
                 position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 **kwargs, ):
        return qwen_approx_attention_forward(self,
                                              hidden_states,
                                              attention_mask,
                                              position_ids,
                                              past_key_value,
                                              output_attentions,
                                              use_cache,
                                              cache_position,
                                              position_embeddings,
                                              attn_sum=attn_sum,
                                              radio_bag=radio_bag,
                                              **kwargs)

    return new_func


def qwen_approx_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
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

    if position_embeddings is None:
        # logger.warning_once(
        #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
        #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
        #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
        #     "removed and `position_embeddings` will be mandatory."
        # )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        #cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        #key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        #print(past_key_value[0].shape,key_states.shape)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)


    # repeat k/v heads if n_kv_heads < n_heads
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)

    # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    # if attention_mask is not None:  # no matter the length, we just slice it
    #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
    #     attn_weights = attn_weights + causal_mask
    #
    # # upcast attention to fp32
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    # attn_output = torch.matmul(attn_weights, value_states)

    # qkv  [bsz,head,len,dim] but flashattn need [bsz,len,head,dim]
    query_states=query_states.transpose(1, 2)
    key_states=key_states.transpose(1, 2)
    value_states=value_states.transpose(1, 2)
    if q_len == 1:
        #print('qwen decode---------------------',query_states.shape)
        attn_output = decode_token(query_states, key_states, value_states, sum_value=attn_sum, radio_bag=radio_bag)
        # attn_output = flash_attn_func(
        #     query_states,
        #     key_states,
        #     value_states,
        #     causal=True,
        #     dropout_p=0.0,
        # )
        # print((attn_output-attn_output1)/attn_output)
    else:
        #print('qwen prefilling---------------------',query_states.shape)
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )

    if attn_output.size() != (bsz,  q_len,self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz,  q_len,self.num_heads, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    # attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2)) if use_cache else None
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
