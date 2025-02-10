model="Qwen2.5-7B-Instruct-1M"
model_provider=LLaMA
context_lengths_min=80000
pretrained_len=1000000
# NOT USE
sparsity=1
attn_sum=0.5
attn_pattern="attn_${attn_sum}"
quant_path=/ms/FM/ydq/notebook/duo_attn/quant/no_norm_4bits_8196_32K.npy

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/niah_ours.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider $attn_sum $quant_path