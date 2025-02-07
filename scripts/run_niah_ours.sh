model="Llama-3-8B-Instruct-Gradient-1048k"
model_provider=LLaMA
context_lengths_min=80000
pretrained_len=1048000
# NOT USE
sparsity=1
attn_pattern="attn_0.5_noquant"
attn_sum=0.5
quant_path=/ms/FM/ydq/notebook/duo_attn/quant/no_norm_4bits_8196_32K.npy

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/niah_ours.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider $attn_sum $quant_path


#model="Llama-2-7B-32K-Instruct"
#model_provider=LLaMA
#context_lengths_min=2000
#pretrained_len=32000
#sparsity=1
#attn_pattern="attn_0.5_noqiant"
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/niah_ours.sh $attn_pattern $sparsity $model $context_lengths_min 0 $pretrained_len $model_provider $attn_sum $quant_path
