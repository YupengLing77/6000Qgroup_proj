# Qwen3-VL-2B Logo 检测 LoRA 微调指南

## 模型信息
- **模型**: [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- **训练方式**: LoRA 微调
- **任务**: Logo 检测 + 分类 + 定位

## 数据准备完成 ✅
- 训练集: 126,923 条样本（行业+公司名称+边界框）
- 测试集: 31,731 条样本

## 步骤 0: 下载模型

```bash
cd /home/jiahuawang/test/classVLM
bash download_model.sh
```

选择下载方式：
1. **HuggingFace** (需要 VPN)
2. **ModelScope** (国内推荐)

## 步骤 1: 转换数据格式

```bash
cd /home/jiahuawang/test/classVLM
python convert_logo_data.py
```

这会生成：
- `logo_train.json` - 训练数据（Qwen3-VL 格式）
- `logo_test.json` - 测试数据
- `logo_images/train/` - 训练图像
- `logo_images/test/` - 测试图像

## 步骤 2: 检查数据格式

```bash
head -50 logo_train.json
```

## 步骤 3: 配置已添加 ✅

数据集配置已添加到:
`/home/jiahuawang/test/classVLM/Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py`

## 步骤 4: 开始 LoRA 微调

```bash
cd /home/jiahuawang/test/classVLM
chmod +x train_logo_lora.sh
bash train_logo_lora.sh
```

## LoRA 配置说明

在 `train_logo_lora.sh` 中：

```bash
LORA_R=64           # LoRA rank (建议 8-128)
LORA_ALPHA=128      # LoRA alpha (通常是 r 的 2 倍)
LORA_DROPOUT=0.05   # Dropout 率
LR=1e-4             # 学习率 (LoRA 可以用更大的)
```

### 显存优化选项：

**如果显存不够，可以调整：**
1. 减小 `BATCH_SIZE=2`
2. 增大 `GRAD_ACCUM=8`
3. 减小 `LORA_R=32`
4. 减小 `--max_pixels 301056`

**2B 模型 + LoRA 估算显存：**
- 单卡 (BATCH_SIZE=4): ~16-20GB
- 单卡 (BATCH_SIZE=2): ~12-14GB

## 步骤 5: 监控训练

```bash
# 查看训练日志
tail -f output/qwen2.5-vl-2b-logo-lora/logs/*.log

# TensorBoard 可视化
tensorboard --logdir output/qwen2.5-vl-2b-logo-lora
```

## 步骤 6: 使用 LoRA 模型推理

训练完成后，LoRA 权重保存在:
`output/qwen3-vl-2b-logo-lora/checkpoint-XXX/`

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch
from PIL import Image

# 加载基础模型
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(
    base_model,
    "output/qwen3-vl-2b-logo-lora/checkpoint-XXX"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# 推理
image = Image.open("test_logo.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Identify the logo in this image. What is the industry and company name?"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(output[0], skip_special_tokens=True))
```

## 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `tune_mm_vision` | False | 不训练视觉编码器 |
| `tune_mm_mlp` | False | 不训练 MLP 投影层 |
| `tune_mm_llm` | True | 只用 LoRA 训练 LLM |
| `lora_enable` | True | 启用 LoRA |
| `lora_r` | 64 | LoRA 秩 |
| `learning_rate` | 1e-5 | 学习率（Qwen3-VL 建议值）|
| `max_pixels` | 50176 | 最大图像像素 (约 224x224) |
| `model_max_length` | 8192 | Qwen3-VL 推荐值 |

## 常见问题

### Q1: OOM (显存不足)
减小 batch_size 或 max_pixels

### Q2: 训练很慢
检查是否启用了 `gradient_checkpointing`

### Q3: 精度不够
- 增大 `lora_r` (如 128)
- 增加训练轮数 `EPOCHS=5`
- 调整学习率

### Q4: 如何只训练分类不训练检测
修改 `convert_logo_data.py`，只保留分类相关的对话格式
