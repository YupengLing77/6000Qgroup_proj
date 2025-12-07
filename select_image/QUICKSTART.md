# 快速开始指南

## 1️⃣ 获取API Key

访问阿里云DashScope控制台获取API Key:
https://dashscope.console.aliyun.com/

## 2️⃣ 设置API Key

```bash
export DASHSCOPE_API_KEY='your-api-key-here'
```

**建议**: 将此命令添加到 `~/.bashrc` 文件中,这样每次登录都会自动设置。

```bash
echo "export DASHSCOPE_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

## 3️⃣ 测试API连接

```bash
cd /home/jiahuawang/test/classVLM
python select_image/test_api.py
```

如果看到 "API connection test PASSED!",说明配置成功。

## 4️⃣ 选择运行模式

### 方案A: 智能选择 (推荐)
使用API评估100张样本,批量处理节省token
```bash
./select_image/run_selection.sh
```

**成本**: 约评估100张图片(批量处理,每次10张)  
**时间**: 5-10分钟  
**输出**: `train_part/train_subset_500.json`

### 方案B: 快速选择 (免费)
不使用API,仅基于行业分布均衡采样
```bash
./select_image/run_selection_fast.sh
```

**成本**: 免费  
**时间**: 1-2分钟  
**输出**: `train_part/train_subset_500_fast.json`

### 方案C: 交互式选择
脚本会引导你完成所有步骤
```bash
./select_image/run_interactive.sh
```

## 5️⃣ 验证输出

```bash
python -c "
import json
data = json.load(open('train_part/train_subset_500.json'))
images = set(item['image'] for item in data)
print(f'Total samples: {len(data)}')
print(f'Unique images: {len(images)}')
print(f'Samples per image: {len(data)/len(images):.1f}')
"
```

期望输出:
```
Total samples: 1500
Unique images: 500
Samples per image: 3.0
```

## 6️⃣ 使用选择的数据集训练

修改训练脚本使用新的数据集:

```bash
cd train_part

# 编辑 train_logo_lora.sh
# 修改 --data_path 为 train_subset_500.json

# 运行训练
./train_logo_lora.sh
```

## 自定义选择参数

如果需要更多控制:

```bash
python select_image/select_image.py \
  --input_json train_part/logo_train.json \
  --output_json train_part/my_custom_subset.json \
  --target_count 300 \
  --use_model_eval \
  --eval_samples 50 \
  --seed 42
```

参数说明:
- `--target_count 300`: 选择300张图片(而不是500)
- `--eval_samples 50`: 只评估50张样本(节省成本)
- `--seed 42`: 随机种子,确保结果可重复

## 故障排除

### 问题: "DASHSCOPE_API_KEY not set"
**解决**: 
```bash
export DASHSCOPE_API_KEY='your-api-key'
```

### 问题: "dashscope not installed"
**解决**: 
```bash
pip install dashscope
```

### 问题: API调用失败
**解决**: 
1. 检查API Key是否正确
2. 检查网络连接
3. 运行测试脚本: `python select_image/test_api.py`

### 问题: 想要更少的API调用
**解决**: 
```bash
# 减少评估样本数
python select_image/select_image.py --eval_samples 30 ...

# 或者使用快速模式(不评估)
./select_image/run_selection_fast.sh
```

## 预期效果

使用智能选择的500张图片:
- ✅ 行业覆盖均衡
- ✅ 难度分布合理(30%简单/40%中等/30%困难)
- ✅ 训练效率更高
- ✅ 模型泛化能力更强

相比随机选择500张,智能选择的数据集应该能让小模型在logo识别、行业分类和定位任务上有更好的表现!
