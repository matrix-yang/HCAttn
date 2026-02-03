from typing import Optional, Tuple, List, Union
import os
import torch
from torch import nn
import types
import numpy as np

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    repeat_kv,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from spare_attn.solution2.true_infer.fast_qkt import fast_qK, fast_fwd, sparse_v_sum
import time

def enable_llama_approx_attention_eval(
        model: LlamaForCausalLM, attn_sum, radio_bag
):
    #enable_tuple_kv_cache_for_llama(model)
    model.model.forward = types.MethodType(old_llama_model_forward, model.model)
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
                 position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 **kwargs, ):
        result = llama_approx_attention_forward(self,
                                              hidden_states,
                                              attention_mask,
                                              position_ids,
                                              past_key_value,
                                              output_attentions,
                                              use_cache,
                                              attn_sum=attn_sum,
                                              radio_bag=radio_bag,
                                              position_embeddings=position_embeddings,
                                              **kwargs)
        # 确保只返回2个值，与原始方法兼容
        return result[0], result[2]

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 检查hidden_states的形状
    if hidden_states.dim() == 2:
        # 如果只有2个维度，添加batch维度
        hidden_states = hidden_states.unsqueeze(0)
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    hidden_size = self.config.hidden_size

    query_states = query_states.view(bsz, q_len, num_heads, head_dim)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim)

    # new data structure for past_key_value
    # past_key_value = (full_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)

    # kv_seq_len = key_states.shape[1]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value[0].shape[2]

    if position_embeddings is None:
        cos, sin = self.rotary_fn(query_states, query_states, position_ids, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    # bsz len head dim

    # decode
    if query_states.shape[1] == 1:
        s4 = time.time()
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        key_index, value_states, C = past_key_value.update(key_states, value_states, self.layer_idx)
        s5 = time.time()
        query_states=query_states.transpose(1, 2)
        attn_output = fast_fwd(query_states, key_index, value_states, C, th=attn_sum)
        s6= time.time()
        #print(f'layer {self.layer_idx} decode use time {s6 - s5} quant use {s5 - s4}')
        # attn_output = flash_attn_func(
        #     query_states,
        #     key_states,
        #     value_states,
        #     causal=True,
        #     dropout_p=0.0,
        # )
        # print((attn_output-attn_output1)/attn_output)
    else:
        s1=time.time()
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )
        s2=time.time()
        past_key_value.update(key_states.transpose(1, 2),
                              value_states.transpose(1, 2),
                              self.layer_idx)
        s3=time.time()
        #print(f'layer {self.layer_idx} prefiling use time {s2-s1} quant use {s3-s2} shape {past_key_value.value_cache[self.layer_idx].shape}')
        # key_states = key_states.transpose(1, 2)
        # value_states = value_states.transpose(1, 2)

    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.cache_utils import Cache, DynamicCache
def old_llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and past_key_values is None:
        return_legacy_cache = True
        # past_key_values = DynamicCache()
        logger.warning_once(
            "We detected that you are passing `past_key_values` as None. Using custom FastCache instead."
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = attention_mask
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if output_attentions and len(layer_outputs) > 1:
            all_self_attns += (layer_outputs[1],)

        if use_cache:
            if output_attentions and len(layer_outputs) > 2:
                next_decoder_cache = layer_outputs[2]
            elif len(layer_outputs) > 1:
                next_decoder_cache = layer_outputs[1]

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    # 确保hidden_states是3维张量
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )