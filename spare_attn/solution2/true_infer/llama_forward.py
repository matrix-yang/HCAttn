import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路径
MODEL_PATH = "/nfs/FM/ydq/model_zoo/Llama-2-7b-chat-hf"

class LlamaInference:
    def __init__(self, model_path=MODEL_PATH, device=None):
        """
        初始化 Llama 模型推理器
        
        Args:
            model_path: 模型路径
            device: 运行设备，默认为自动检测（优先使用 GPU）
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
            device_map="auto"  # 自动分配设备
        )
        
        # 设置模型为评估模式
        self.model.eval()
    
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
        
        # 存储生成的 token
        generated_tokens = []
        
        # 生成文本
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 调用 model.forward
                outputs = self.model(
                    input_ids=input_ids,
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
    print("Initializing Llama inference...")
    llama_inference = LlamaInference()
    
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