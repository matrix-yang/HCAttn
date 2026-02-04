import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from spare_attn.solution2.custom_cache.ki_cache import FastCache
from spare_attn.solution2.true_infer.modeify_llama_cpu import enable_llama_approx_attention_eval
from spare_attn.solution2.quanters.ki_quanter import MultiGroupQuanter
import logging

# 禁用一些警告
logging.basicConfig(level=logging.ERROR)

# 模型路径
MODEL_PATH = "/nfs/FM/ydq/model_zoo/Llama-2-7b-chat-hf"

class LlamaInferenceKI:
    def __init__(self, model_path=MODEL_PATH, device=None, attn_sum=0.95, radio_bag=[]):
        """
        初始化 Llama 模型推理器，使用 KI 缓存
        
        Args:
            model_path: 模型路径
            device: 运行设备，默认为自动检测（优先使用 GPU）
            attn_sum: 注意力阈值
            radio_bag: 压缩比例袋
        """
        # 自动检测设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # 加载分词器
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,  # 使用半精度浮点数以节省内存
            device_map=self.device  # 自动分配设备
        )
        
        # 设置模型为评估模式
        self.model.eval()
        
        # 初始化量化器和缓存
        print("Initializing KI cache...")
        # 这里假设我们使用 MultiGroupQuanter，需要提供向量
        # 实际使用时，需要根据具体情况调整
        # 这里使用随机向量作为示例
        import numpy as np
        import os
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        vectors_path = os.path.join(project_root, "spare_attn", "C_npy", "dim4_equal_4bits_4096_32K_vec2.npy")
        vectors = np.load(vectors_path)
        vectors = torch.from_numpy(vectors).unsqueeze(0).repeat(32, 1, 1).to(self.device)
        self.quanter = MultiGroupQuanter(vectors, self.device, torch.float16)
        self.attn_sum = attn_sum
        self.radio_bag = radio_bag
        
        # 启用近似注意力
        enable_llama_approx_attention_eval(self.model, self.attn_sum, self.radio_bag)
    
    def generate(self, prompt, max_new_tokens=100):
        """
        生成文本
        
        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
        
        Returns:
            生成的文本
        """
        # 对输入进行编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # 初始化 KI 缓存
        past_key_value = FastCache(self.quanter)
        
        # 存储生成的 token
        generated_tokens = []
        
        # 生成文本
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 调用 model.forward
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_value,
                    use_cache=True
                )
                
                # 获取 logits
                logits = outputs.logits[:, -1, :]
                
                # 使用贪婪解码
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # 检查是否生成了结束标记
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # 添加到生成的 token 列表
                generated_tokens.append(next_token.item())
                
                # 更新 input_ids 为下一个 token
                input_ids = next_token
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()

if __name__ == "__main__":
    # 示例使用
    print("Initializing Llama inference with KI cache...")
    llama_inference = LlamaInferenceKI()
    
    # 示例提示
    prompt = "Hello, how are you today?"
    print(f"\nInput prompt: {prompt}")
    
    # 生成回复
    print("Generating response...")
    response = llama_inference.generate(prompt)
    
    print(f"\nGenerated response: {response}")
    
    # 另一个示例
    prompt = "Explain quantum computing in simple terms"
    print(f"\nInput prompt: {prompt}")
    
    print("Generating response...")
    response = llama_inference.generate(prompt, max_new_tokens=200)
    
    print(f"\nGenerated response: {response}")
