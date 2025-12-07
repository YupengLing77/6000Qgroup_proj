#!/bin/bash

# 增强版智能选择 - 更均衡的难度分布
# 评估更多样本,添加随机变化,确保难度分布更合理

cd /home/jiahuawang/test/classVLM

# 检查API Key
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: Please set DASHSCOPE_API_KEY environment variable"
    exit 1
fi

echo "Starting ENHANCED intelligent data selection with Aliyun API..."
echo "Features:"
echo "  - Evaluates 150 sample images (more diversity)"
echo "  - Adds random variation to inferred scores"
echo "  - Better difficulty distribution"
echo ""

python select_image/select_image.py \
  --input_json train_part/logo_train.json \
  --output_json train_part/train_subset_500_enhanced.json \
  --target_count 500 \
  --use_model_eval \
  --eval_samples 150 \
  --seed 42

echo ""
echo "Selection complete! Output saved to train_part/train_subset_500_enhanced.json"
echo ""
echo "Check difficulty distribution:"
python -c "
import json
from collections import Counter

# 分析难度分布
print('Analyzing generated subset...')
data = json.load(open('train_part/train_subset_500_enhanced.json'))
images = list(set(item['image'] for item in data))
print(f'Total images: {len(images)}')
print(f'Total samples: {len(data)}')
"
