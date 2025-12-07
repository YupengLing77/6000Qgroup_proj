#!/bin/bash

# 使用阿里云API选择训练子集的完整示例
# 展示如何设置API Key并运行选择

echo "====================================="
echo "智能训练数据选择 - 阿里云API版本"
echo "====================================="
echo ""

# 1. 检查并设置API Key
if [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "Step 1: 设置API Key"
    echo "请输入你的阿里云DashScope API Key:"
    read -r api_key
    export DASHSCOPE_API_KEY="$api_key"
    echo "API Key已设置"
else
    echo "Step 1: API Key已存在,使用环境变量中的值"
fi
echo ""

# 2. 安装依赖
echo "Step 2: 安装必要的包..."
pip install dashscope -q
pip install tqdm -q
echo "依赖安装完成"
echo ""

# 3. 选择模式
echo "Step 3: 选择运行模式"
echo "1) 智能选择 (使用API评估,约100张样本,推荐)"
echo "2) 快速选择 (不使用API,免费但较简单)"
read -p "请选择 (1/2): " mode
echo ""

cd /home/jiahuawang/test/classVLM

if [ "$mode" == "1" ]; then
    echo "Step 4: 开始智能选择..."
    echo "- 评估样本: 100张图片"
    echo "- 批量大小: 10张/次"
    echo "- 预计时间: 5-10分钟"
    echo ""
    
    python select_image/select_image.py \
        --input_json train_part/logo_train.json \
        --output_json train_part/train_subset_500.json \
        --target_count 500 \
        --use_model_eval \
        --eval_samples 100 \
        --seed 42
    
    echo ""
    echo "✅ 完成! 输出文件: train_part/train_subset_500.json"
    
elif [ "$mode" == "2" ]; then
    echo "Step 4: 开始快速选择..."
    echo "- 不使用API评估"
    echo "- 预计时间: 1-2分钟"
    echo ""
    
    python select_image/select_image.py \
        --input_json train_part/logo_train.json \
        --output_json train_part/train_subset_500_fast.json \
        --target_count 500 \
        --seed 42
    
    echo ""
    echo "✅ 完成! 输出文件: train_part/train_subset_500_fast.json"
else
    echo "❌ 无效的选择"
    exit 1
fi

echo ""
echo "====================================="
echo "下一步: 使用选择的数据集训练模型"
echo "cd train_part && ./train_logo_lora.sh"
echo "====================================="
