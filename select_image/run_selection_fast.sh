#!/bin/bash

# 快速选择训练子集 (不使用API评估)
# 基于行业分布和随机采样的均衡选择
# 不消耗API token,几分钟内完成

cd /home/jiahuawang/test/classVLM

echo "Starting fast stratified sampling without API evaluation..."
python select_image/select_image.py \
  --input_json train_part/logo_train.json \
  --output_json train_part/train_subset_500_fast.json \
  --target_count 500 \
  --seed 42

echo "Selection complete! Output saved to train_part/train_subset_500_fast.json"
