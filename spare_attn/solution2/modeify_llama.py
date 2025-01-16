from typing import Optional, Tuple
import os
import torch
from torch import nn
import types
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache, enable_tuple_kv_cache_for_llama

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


def decode_token(query_states_ori, key_states_ori, value_states_ori, sum_value, radio_bag):
    query_states = query_states_ori.transpose(1, 2).squeeze(0)
    key_states = key_states_ori.transpose(1, 2).squeeze(0)
    value_states = value_states_ori.transpose(1, 2).squeeze(0)
    # bsz 1 heads 32 seqlen 128
    attention_scores = torch.bmm(query_states, key_states.transpose(-1, -2)) * 0.088388
    # 32 1000
    attention_probs = torch.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    # print('attention_probs',attention_probs.shape)
    # 获取降序排序的索引
    sorted_indices = torch.argsort(attention_probs, descending=True, dim=-1)
    # print(sorted_indices.shape)
    search_index_ls = [16, 32, 64, 96, 128, 256, 384, 512, 768, attention_probs.shape[-1]]
    for i in search_index_ls:
        result = torch.gather(attention_probs, dim=-1, index=sorted_indices[:, :, :i])
        # print(result.shape,result.sum(-1))
        if all(result.sum(-1) > sum_value):
            break
    selected_indices = sorted_indices[:, :, :i]
    # print('selected_indices',selected_indices)

    selected_indices = selected_indices.sort(dim=-1).values
    # print('selected_indices sort',selected_indices)
    # print('radio',i,key_states.shape[-2],i/key_states.shape[-2])
    radio_bag.append(i / key_states.shape[-2])
    # 按原始顺序去除这些索引
    # print(len(selected_indices),selected_indices)
    # print(selected_indices.shape)
    # print(value_states.shape)
    attn_weights = torch.gather(attention_probs, dim=-1, index=selected_indices)
    # print('attn_weights',attn_weights.shape)
    v_index = selected_indices.squeeze(1).unsqueeze(-1).expand(-1, -1, 128)
    # print('v_index',v_index.shape)
    select_value_states = torch.gather(value_states, dim=-2, index=v_index)
    # print(attn_weights.shape,value_states.shape)
    # print('select_value_states',v_index.shape)
    # print(select_value_states.shape,select_value_states[1,255]==value_states[1,v_index[1,255,0]])
    attn_output = torch.matmul(attn_weights, select_value_states)
    attn_output = attn_output.unsqueeze(0).transpose(1, 2)
    # print(attn_output.shape)
    return attn_output


def decode_token_ori(query_states, key_states, value_states):
    query = query_states.transpose(1, 2)
    key = key_states.transpose(1, 2)
    value = value_states.transpose(1, 2)
    # print(query_states.shape,key_states.shape)
    # bsz 1 heads 32 seqlen 128

    attn_weights = torch.matmul(query, key.transpose(2, 3))
    attn_weights /= 11.3137

    # 32 1000
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


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
