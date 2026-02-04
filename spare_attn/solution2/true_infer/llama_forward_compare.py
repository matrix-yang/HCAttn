import sys
import os
import torch
import time

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入两个模型类
from llama_forward import LlamaInference
from llama_forward_ki import LlamaInferenceKI

# 测试提示
test_prompts = [
    "Hello, how are you today?",
    "Explain quantum computing in simple terms",
    "Write a short poem about artificial intelligence",
    "What are the benefits of using renewable energy?"
]

# 最大生成 token 数
max_new_tokens = 50

def test_inference_time(model, prompts, max_new_tokens):
    """
    测试模型推理时间
    
    Args:
        model: 模型实例
        prompts: 测试提示列表
        max_new_tokens: 最大生成 token 数
    
    Returns:
        平均推理时间（秒）
    """
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(prompts):
        print(f"\nTest {i+1}/{len(prompts)}:")
        print(f"Prompt: {prompt}")
        
        # 测量时间
        start_time = time.time()
        
        # 生成文本
        generated_text = model.generate(prompt, max_new_tokens=max_new_tokens)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 计算生成的 token 数
        generated_tokens = model.tokenizer.encode(generated_text, add_special_tokens=False)
        num_tokens = len(generated_tokens)
        
        # 更新统计
        total_time += inference_time
        total_tokens += num_tokens
        
        print(f"Generated tokens: {num_tokens}")
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Tokens per second: {num_tokens/inference_time:.2f} tokens/s")
        print(f"Generated text: {generated_text[:100]}..." if len(generated_text) > 100 else f"Generated text: {generated_text}")
    
    # 计算平均值
    avg_time = total_time / len(prompts)
    avg_tokens_per_second = total_tokens / total_time
    
    return avg_time, avg_tokens_per_second

def main():
    """
    主函数，比较两个模型的推理时间
    """
    print("=" * 80)
    print("Testing Llama inference time comparison")
    print("=" * 80)
    
    # 测试原始模型
    print("\n" + "-" * 80)
    print("Testing original model (llama_forward.py)")
    print("-" * 80)
    
    original_model = LlamaInference()
    original_avg_time, original_avg_tokens_per_second = test_inference_time(
        original_model, test_prompts, max_new_tokens
    )
    
    # 清理内存
    del original_model
    torch.cuda.empty_cache()
    
    # 测试 KI 模型
    print("\n" + "-" * 80)
    print("Testing KI model (llama_forward_ki.py)")
    print("-" * 80)
    
    ki_model = LlamaInferenceKI()
    ki_avg_time, ki_avg_tokens_per_second = test_inference_time(
        ki_model, test_prompts, max_new_tokens
    )
    
    # 清理内存
    del ki_model
    torch.cuda.empty_cache()
    
    # 比较结果
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)
    
    print(f"Original model average time: {original_avg_time:.4f} seconds")
    print(f"Original model average tokens/s: {original_avg_tokens_per_second:.2f}")
    print(f"KI model average time: {ki_avg_time:.4f} seconds")
    print(f"KI model average tokens/s: {ki_avg_tokens_per_second:.2f}")
    
    # 计算加速比
    speedup = original_avg_time / ki_avg_time
    print(f"\nSpeedup: {speedup:.2f}x faster with KI cache")
    
    # 计算性能提升百分比
    performance_gain = (ki_avg_tokens_per_second - original_avg_tokens_per_second) / original_avg_tokens_per_second * 100
    print(f"Performance gain: {performance_gain:.2f}% increase in tokens/s")

if __name__ == "__main__":
    main()
