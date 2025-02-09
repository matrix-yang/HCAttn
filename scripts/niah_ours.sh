cd eval/needle
mkdir -p logs img results

attn_pattern=$1
sparsity=$2
model=$3
context_lengths_min=$4
s_len=$5
pretrained_len=$6
model_provider=$7
attn_sum=$8
quant_path=$9

# cut the last part of the path of the attn_pattern to get the name
attn_pattern_name=$(echo $attn_pattern | rev | cut -d'/' -f1 | rev)

suffix="${attn_pattern_name}-attn_sum_${attn_sum}"
(
    python -u needle_in_haystack_ours.py --s_len $s_len \
        --e_len $pretrained_len \
        --context_lengths_min $context_lengths_min \
        --context_lengths_max $pretrained_len \
        --model_provider $model_provider \
        --model_name_suffix $suffix \
        --sparsity $sparsity \
        --simulation_length 0 \
        --context_lengths_num_intervals 13 \
        --document_depth_percent_intervals 10 \
        --prefilling_chunk_size 32000 \
        --model_path /ms/FM/ydq/kvcache/${model} \
        --quant_path $quant_path \
        --attn_sum $attn_sum \
        --modify

) 2>&1 | tee logs/eval_${model}_${suffix}.log

python visualize.py \
    --folder_path "results/${model}_${suffix}/" \
    --model_name "${model}_attn_${attn_sum}" \
    --pretrained_len $pretrained_len
