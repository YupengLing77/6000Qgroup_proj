# Qwen3-VL Logo Detection Fine-tuning

基于 Qwen3-VL-2B-Instruct 的 Logo 检测和识别任务 LoRA 微调项目。

## 📋 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [评估与分析](#评估与分析)

---

## 🔧 环境要求

### 必需依赖

```bash
pip install torch==2.6.0 torchvision==0.21.0
pip install transformers>=4.57.0
pip install deepspeed==0.17.1
pip install accelerate==1.7.0
pip install peft==0.17.1
pip install flash-attn==2.7.4.post1
pip install triton==3.2.0
pip install torchcodec==0.2
pip install datasets pillow tqdm
```

### 硬件要求

- **GPU**: 至少 1 张 24GB 显存 GPU（如 RTX 3090/4090, A100）
- **存储**: 至少 50GB 可用空间

---

## 🚀 快速开始

### ⚠️ 首次使用必须配置路径

在开始训练前，需要修改以下文件中的绝对路径：

#### 1. 训练脚本路径

**文件**: `train_part/train_logo_lora.sh`

```bash
# 第 7 行：修改为你的项目根目录
cd /home/YOUR_USERNAME/YOUR_PATH/classVLM
```

#### 2. 数据集配置路径

**文件**: `Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py`

```python
# 第 31 行和第 37 行：修改为你的项目路径
LOGO_DATASET = {
    "annotation_path": "/home/YOUR_USERNAME/YOUR_PATH/classVLM/train_subset.json",
    "data_path": "",
}

LOGO_FULL = {
    "annotation_path": "/home/YOUR_USERNAME/YOUR_PATH/classVLM/logo_train.json",
    "data_path": "",
}
```

**快速替换命令**:
```bash
# 在项目根目录执行，自动替换所有路径
PROJECT_PATH=$(pwd)  # 获取当前目录的绝对路径
sed -i "s|/home/jiahuawang/test/classVLM|${PROJECT_PATH}|g" train_part/train_logo_lora.sh
sed -i "s|/home/jiahuawang/test/classVLM|${PROJECT_PATH}|g" Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py
```

**命令解释**:
- `PROJECT_PATH=$(pwd)`: 获取当前目录的绝对路径并保存到变量
- `sed -i`: 直接修改文件（不加 `-i` 只会输出到屏幕）
- `"s|旧文本|新文本|g"`: 替换命令
  - `s` = substitute（替换）
  - `|` = 分隔符（也可用 `/`，但路径中有 `/` 所以用 `|` 更清晰）
  - `/home/jiahuawang/test/classVLM` = 要查找的旧路径
  - `${PROJECT_PATH}` = 替换成的新路径（你的实际项目路径）
  - `g` = global（替换文件中所有匹配项，不只是第一个）

**手动修改方式**（如果不想用命令）:
```bash
# 用文本编辑器打开文件，把所有 /home/jiahuawang/test/classVLM 
# 改成你的实际路径，比如 /home/yourname/projects/classVLM
vim train_part/train_logo_lora.sh
vim Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py
```

---

### 训练流程

```bash
# 1. 下载数据集
python download.py

# 2. 转换数据格式（一次性操作）
python logo_data_oral/convert_logo_data.py

# 3. 准备训练数据（10K样本）
python prepare_data.py --train 10000 --test 1000

# 4. 开始训练
bash train_part/train_logo_lora.sh

# 5. 评估模型
python comprehensive_eval.py \
  --checkpoint ./output/qwen3-vl-2b-logo-lora_YYYYMMDD_HHMMSS/checkpoint-XXX \
  --label "exp1_10k_r64" \
  --lora_rank 64 \
  --train_samples 10000 \
  --prompt v1 \
  --num_samples 30
```

---

## 📖 详细步骤

### 步骤 1: 下载数据集

从 [LogoDet-3K](https://github.com/Wangjing1551/LogoDet-3K-Dataset) 下载数据集：

```bash
python download.py
```

**输出:**
- `logo_data/` - 数据集文件夹
- 包含 126,923 训练图像 + 31,731 测试图像

---

### 步骤 2: 转换数据格式

将数据集转换为 Qwen3-VL 训练格式（**只需运行一次**）：

```bash
python logo_data_oral/convert_logo_data.py
```

**生成文件:**
- `logo_images/` - 所有图像文件
- `logo_train.json` - 完整训练数据（380,769 条样本，每张图 3 个任务）
- `logo_test.json` - 完整测试数据（95,193 条样本）

**三个训练任务:**
1. **分类**: 识别 logo 的行业和公司名称
2. **检测**: 定位 logo 的 bbox 坐标
3. **行业识别**: 仅识别行业类别

---

### 步骤 3: 准备训练子集

从完整数据集中选择指定数量的图像进行训练：

```bash
# 小规模测试（1K 图像 = 1K 样本）
python prepare_data.py --train 1000 --test 200

# 中等规模（10K 图像 = 30K 样本）
python prepare_data.py --train 10000 --test 1000

# 大规模训练（50K 图像 = 150K 样本）
python prepare_data.py --train 50000 --test 5000

# 全量训练（126K 图像 = 380K 样本）
python prepare_data.py --train 126923 --test 10000

# 自定义随机种子
python prepare_data.py --train 10000 --seed 123
```

**参数说明:**
- `--train`: 训练图像数量（默认 10000）
- `--test`: 测试图像数量（默认 1000）
- `--seed`: 随机种子（默认 42）

**生成文件:**
- `train_subset.json` - 训练子集
- `test_subset.json` - 测试子集

**注意:** 
- 无需重复转换数据集，只需运行此脚本选择不同数量即可
- 生成的文件会保存在**项目根目录**下（与 `logo_train.json` 同级）

---

### 步骤 4: 训练模型

启动 LoRA 微调训练：

```bash
bash train_part/train_logo_lora.sh
```

**训练配置（可在脚本中修改）:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LORA_R` | 64 | LoRA rank，越大效果越好但显存占用越多 |
| `LORA_ALPHA` | 128 | LoRA alpha，通常为 rank 的 2 倍 |
| `LORA_DROPOUT` | 0.05 | Dropout 比例 |
| `LR` | 1e-5 | 学习率 |
| `BATCH_SIZE` | 8 | 每个 GPU 的 batch size |
| `GRAD_ACCUM` | 4 | 梯度累积步数 |
| `EPOCHS` | 3 | 训练轮数 |

**输出目录:**
- `output/qwen3-vl-2b-logo-lora_YYYYMMDD_HHMMSS/` - 带时间戳的输出目录
- 每次训练自动保存到独立目录，不会覆盖之前的 checkpoint

**监控训练:**
```bash
# 查看 TensorBoard
tensorboard --logdir output/
```

---

## 📊 评估与分析

### 单次评估

评估指定 checkpoint 的性能：

```bash
python comprehensive_eval.py \
  --checkpoint ./output/qwen3-vl-2b-logo-lora_20251119_143052/checkpoint-500 \
  --label "exp1_10k_r64" \
  --lora_rank 64 \
  --train_samples 10000 \
  --prompt v1 \
  --num_samples 30 \
  --output_dir evaluation_results
```

**参数说明:**
- `--checkpoint`: LoRA checkpoint 路径（必需）
- `--label`: 实验标签，用于标识不同实验（必需）
- `--lora_rank`: LoRA rank 大小（默认 64）
- `--train_samples`: 训练样本数量（默认 0）
- `--prompt`: 提示词版本 `v1`/`v2`/`v3`（默认 v1）
- `--num_samples`: 评估样本数量（默认 30）
- `--test_json`: 测试数据集路径（默认 test_subset.json）
- `--output_dir`: 输出目录（默认 evaluation_results）

**生成报告:**
- `evaluation_table.csv` - CSV 格式对比表格（可直接用于论文）
- `evaluation_report.md` - Markdown 格式完整报告
- `results.json` - 详细结果（包含所有元数据）

---

### 提示词对比实验

测试不同提示词版本对模型性能的影响：

```bash
python test_prompts.py \
  --checkpoint ./output/qwen3-vl-2b-logo-lora_20251119_143052/checkpoint-500 \
  --label "exp1_10k_r64" \
  --lora_rank 64 \
  --train_samples 10000
```

**提示词版本:**

| 版本 | 说明 | 示例 |
|------|------|------|
| **v1** | 原始提示词 | "Identify the logo in this image..." |
| **v2** | 详细提示词 | "Analyze this image carefully. Identify the logo..." |
| **v3** | 简洁提示词 | "What is this logo? Industry and company?" |

**生成报告:**
- `prompt_comparison_YYYYMMDD_HHMMSS/comparison_table.csv` - 三版本对比表格
- `prompt_comparison_YYYYMMDD_HHMMSS/comparison_report.md` - 完整分析报告
- 每个版本的独立评估结果

**报告包含:**
- 三种提示词在三个任务上的性能对比
- Base 模型 vs LoRA 模型的提升对比
- 最佳提示词版本推荐

---

## 📁 项目结构

```
classVLM/
├── download.py                      # 数据集下载脚本
├── prepare_data.py                  # 训练数据准备（支持命令行参数）
├── comprehensive_eval.py            # 综合评估脚本（支持标签和提示词）
├── test_prompts.py                  # 提示词对比实验
├── logo_data_oral/
│   └── convert_logo_data.py        # 数据格式转换（一次性）
├── train_part/
│   └── train_logo_lora.sh          # 训练脚本（自动添加时间戳）
├── logo_data/                       # 原始数据集
├── logo_images/                     # 转换后的图像
├── logo_train.json                  # 完整训练数据（380K 样本）
├── logo_test.json                   # 完整测试数据（95K 样本）
├── train_subset.json                # 训练子集
├── test_subset.json                 # 测试子集
└── output/                          # 训练输出（带时间戳）
```

---

## 💡 使用建议

1. **首次使用**: 
   - 必须先配置绝对路径（见[快速开始](#-快速开始)）
   - 建议从 1K 样本开始快速验证流程
2. **正式训练**: 推荐使用 10K-50K 样本，平衡效果和时间
3. **全量训练**: 126K 样本需要较长训练时间，适合最终模型
4. **提示词优化**: 使用 `test_prompts.py` 找到最佳提示词版本
5. **实验管理**: 使用有意义的 `--label` 标识不同实验配置
6. **多机训练**: 如果在多台机器上训练，每台机器都需要修改路径配置

---

## 🎯 评估指标

| 任务 | 指标 | 说明 |
|------|------|------|
| 分类 | 准确率 | Logo 行业和公司名称识别准确率 |
| 检测 | IoU | Bbox 定位的平均 IoU |
| 行业识别 | 准确率 | 仅行业类别的识别准确率 |

---

## 📝 引用

如果使用 LogoDet-3K 数据集，请引用：

```bibtex
@article{wang2020logodet3k,
  title={LogoDet-3K: A Large-Scale Image Dataset for Logo Detection},
  author={Wang, Jing and others},
  year={2020}
}
```