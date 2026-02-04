## 项目介绍
## 环境
```
# 参考README.old.md,构建transformers 4.45.2 环境,目前只安装好torch，cuda相关环境
# 只保证analysis_test正常运行
192.168.14.129:80/fm/hca_attn_env:v0
已更新环境，可以正常运行analysis_test.py和llama_forward_ki.py
192.168.14.129:80/fm/hca_attn_env:v1
```
## attn测试  
```
cd project_root
python3 ./spare_attn/solution2/true_infer/analysis_test.py
```

## llama forward测试  
目前模型会输出乱码，原因是分段后求和bf16会导致精度误差，测试效率可以先不管
```
cd project_root
python3 ./spare_attn/solution2/true_infer/llama_forward_ki.py

# 与原生的对比测试
python3 ./spare_attn/solution2/true_infer/llama_forward_compare.py
```