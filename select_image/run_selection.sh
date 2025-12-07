#!/bin/bash

# 智能选择训练子集 - 使用阿里云API
# 使用Qwen3-VL-Plus模型通过API选择500张最优图片
# 优化token使用,只评估100张样本图片

cd /home/jiahuawang/test/classVLM

# 检查API Key
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: Please set DASHSCOPE_API_KEY environment variable"
    echo "Export your API key: export DASHSCOPE_API_KEY='your-api-key'"
    exit 1
fi

# 安装必要的包
echo "Installing dashscope package..."
pip install dashscope -q

echo "Starting intelligent data selection with Aliyun API..."
echo "Configuration:"
echo "  - Evaluating 300 sample images"
echo "  - Batch size: 20 images per API call"
echo "  - Expected API calls: ~15 times"
echo "  - Estimated cost: ~1-2 RMB"
echo ""

python select_image/select_image.py \
  --input_json train_part/logo_train.json \
  --output_json train_part/train_subset_500.json \
  --target_count 500 \
  --use_model_eval \
  --eval_samples 300 \
  --seed 42

echo "Selection complete! Output saved to train_part/train_subset_500.json"
