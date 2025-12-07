#!/bin/bash

# 高质量智能选择 - 评估更多样本,确保选择质量
# 使用更大的token预算获得更好的结果

cd /home/jiahuawang/test/classVLM

# 检查API Key
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Error: Please set DASHSCOPE_API_KEY environment variable"
    exit 1
fi

echo "=========================================="
echo "高质量智能数据选择"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 评估样本: 500张图片"
echo "  - 批量大小: 20张/次"
echo "  - API调用次数: ~25次"
echo "  - 预计成本: ~2-3元人民币"
echo "  - 预计时间: 3-5分钟"
echo ""
read -p "确认继续? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "开始选择..."

python select_image/select_image.py \
  --input_json train_part/logo_train.json \
  --output_json train_part/train_subset_500_hq.json \
  --target_count 500 \
  --use_model_eval \
  --eval_samples 500 \
  --seed 42

echo ""
echo "=========================================="
echo "选择完成!"
echo "=========================================="
echo "输出文件: train_part/train_subset_500_hq.json"
echo ""
echo "查看统计信息:"
python -c "
import json
from collections import Counter

data = json.load(open('train_part/train_subset_500_hq.json'))
images = list(set(item['image'] for item in data))

print(f'✓ 总样本数: {len(data)}')
print(f'✓ 唯一图片: {len(images)}')
print(f'✓ 每张图片样本数: {len(data)/len(images):.1f}')

# 统计行业分布
industries = []
for item in data:
    for conv in item['conversations']:
        if conv['from'] == 'gpt' and 'Industry:' in conv['value']:
            industry = conv['value'].split('\\n')[0].replace('Industry:', '').strip()
            industries.append(industry)
            break

industry_dist = Counter(industries)
print(f'\\n行业分布:')
for industry, count in industry_dist.most_common():
    print(f'  {industry}: {count}')
"
