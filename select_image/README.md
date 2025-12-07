# 智能训练数据选择工具 (阿里云API版本)

使用阿里云Qwen3-VL-Plus API智能选择最优训练子集,用于小模型LoRA微调。

## 主要优势

✅ **无需本地GPU** - 使用阿里云API,不需要占用本地显卡  
✅ **批量处理** - 一次API调用评估10张图片,大幅节省token  
✅ **智能采样** - 只评估100张样本,推断其他图片难度  
✅ **快速完成** - 几分钟内完成500张图片选择  
✅ **成本优化** - 相比逐张评估节省90%以上的token

## 功能特点

1. **智能难度评估**: 使用Qwen3-VL-Plus评估图片的识别难度
2. **类别均衡**: 保证各行业的logo都有合理的代表性
3. **难度分级**: 30%简单、40%中等、30%困难
4. **批量API调用**: 每次评估10张图片,大幅降低成本

## 使用方法

### 准备工作

1. 获取阿里云API Key: https://dashscope.console.aliyun.com/
2. 设置环境变量:
```bash
export DASHSCOPE_API_KEY='your-api-key-here'
```

### 方法1: 标准智能选择(推荐)

```bash
cd /home/jiahuawang/test/classVLM
./select_image/run_selection.sh
```

这个方法会:
- 使用Qwen3-VL-Plus API评估300张样本图片
- 批量处理,每次API调用评估20张图片
- 约15次API调用
- 输出: `train_part/train_subset_500.json`
- 时间: 约3-5分钟
- 成本: 约1-2元人民币

### 方法2: 高质量选择(评估更多样本)

```bash
cd /home/jiahuawang/test/classVLM
./select_image/run_selection_hq.sh
```

这个方法会:
- 使用API评估500张样本图片(更多样本,质量更高)
- 批量处理,每次20张
- 约25次API调用
- 输出: `train_part/train_subset_500_hq.json`
- 时间: 约5-8分钟
- 成本: 约2-3元人民币

### 方法3: 快速选择(免费,不使用API)

```bash
cd /home/jiahuawang/test/classVLM
./select_image/run_selection_fast.sh
```

这个方法会:
- 基于行业分布进行均衡采样
- 不使用API评估,完全免费
- 输出: `train_part/train_subset_500_fast.json`
- 时间: 1-2分钟

## 自定义参数

```bash
python select_image/select_image.py \
  --api_key YOUR_API_KEY \
  --input_json train_part/logo_train.json \
  --output_json train_part/train_subset_500.json \
  --target_count 500 \
  --use_model_eval \
  --eval_samples 100 \
  --seed 42
```

### 参数说明

- `--api_key`: 阿里云API密钥(也可以通过环境变量DASHSCOPE_API_KEY设置)
- `--input_json`: 输入的完整训练集JSON文件
- `--output_json`: 输出的子集JSON文件
- `--target_count`: 目标图片数量(默认500)
- `--use_model_eval`: 是否使用API评估难度
- `--eval_samples`: 使用API评估的样本数(默认100,已优化)
- `--seed`: 随机种子(默认42,保证可重复性)

## 成本估算

### 标准模式(300张样本):
- 评估300张图片
- 批量大小20张/次 = 15次API调用
- 每张图片约50KB,base64后约70KB
- 每次调用约处理20张图片 = 约8000-10000 tokens
- **总计: 约120,000-150,000 tokens**
- **预估成本: 1-2元人民币**

### 高质量模式(500张样本):
- 评估500张图片
- 批量大小20张/次 = 25次API调用
- **总计: 约200,000-250,000 tokens**
- **预估成本: 2-3元人民币**

### 成本对比
- 本方案(批量base64): 1-3元 ✓
- 如果逐张评估: 15-30元 ✗
- 如果本地部署8B模型: 需要20GB显存 ✗

## 输出格式

输出的JSON文件格式与`train_subset_3k.json`完全一致:

```json
[
  {
    "image": "logo_images/train/train_012016.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nIdentify the logo in this image..."
      },
      {
        "from": "gpt",
        "value": "Industry: Food\nCompany: 2130"
      }
    ]
  },
  ...
]
```

每张选中的图片包含3个样本:
1. Logo识别(公司名+行业)
2. Logo定位(bbox坐标)
3. 行业分类

## 选择策略

### 1. 行业均衡
- 分析完整数据集的行业分布
- 为每个行业分配合理的配额
- 保证每个行业至少3张图片

### 2. 难度分级 (使用API评估时)
- 批量评估100张样本图片的难度(1-10分)
  - 1-3分: 简单(logo清晰、大、背景简单)
  - 4-7分: 中等(存在一些挑战)
  - 8-10分: 困难(logo小、遮挡、背景复杂)
- 按30%简单、40%中等、30%困难的比例选择
- 未评估的图片根据同行业已评估图片推断
- 批量处理: 每次API调用评估10张图片

### 3. 随机性与可重复性
- 使用随机种子确保结果可重复
- 在满足约束的前提下随机选择,避免偏差
- 达到目标数量(500张)立即停止,节省时间

## 后续使用

生成的`train_subset_500.json`可以直接用于LoRA微调:

```bash
# 复制并重命名
cp train_part/train_subset_500.json train_part/train_subset_500.json

# 使用现有的训练脚本
cd train_part
./train_logo_lora.sh
```

记得修改训练脚本中的:
- `--data_path`: 指向新生成的JSON文件
- 训练轮数可能需要调整(500张图片可以训练更多轮)

## 显卡要求

- **Qwen3-VL-8B**: 需要约16-20GB显存(bfloat16精度)
- **3080显卡**: 10GB显存,可以运行但比较紧张
- 如果显存不足,可以:
  1. 使用快速模式(不评估)
  2. 减少`--sample_rate`(如0.02只评估2%)
  3. 使用更小的模型进行评估

## 预期效果

使用智能选择的500张图片子集训练小模型,相比随机选择,应该能够:
- ✅ 更好的行业覆盖
- ✅ 更均衡的难度分布
- ✅ 更高的训练效率
- ✅ 更好的泛化能力

## 故障排除

### API Key未设置
```bash
export DASHSCOPE_API_KEY='your-api-key-here'
# 或者在命令行中指定
python select_image/select_image.py --api_key 'your-api-key' ...
```

### 降低成本
```bash
# 减少评估样本数
python select_image/select_image.py --eval_samples 50 ...

# 或者不使用API评估(免费)
./select_image/run_selection_fast.sh
```

### JSON文件太大无法加载
脚本使用流式JSON加载,理论上支持任意大小的文件。如果仍有问题,请检查内存。

## API文档

阿里云DashScope API文档: https://help.aliyun.com/zh/dashscope/  
Qwen-VL-Plus模型文档: https://help.aliyun.com/zh/dashscope/developer-reference/qwen-vl-plus
