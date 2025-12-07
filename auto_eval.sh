#!/bin/bash

# 自动评估脚本
# 扫描log文件,找到训练输出路径,自动评估最新checkpoint

LOG_DIR="/home/jiahuawang/test/classVLM/train_part/log"
BASE_DIR="/home/jiahuawang/test/classVLM"
EVAL_SCRIPT="comprehensive_eval.py"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "自动评估脚本"
echo "=========================================="
echo ""

# 检查log目录是否存在
if [ ! -d "$LOG_DIR" ]; then
    echo -e "${RED}错误: Log目录不存在: $LOG_DIR${NC}"
    exit 1
fi

# 遍历所有log文件
for log_file in "$LOG_DIR"/*.log; do
    if [ ! -f "$log_file" ]; then
        continue
    fi
    
    log_name=$(basename "$log_file" .log)
    echo -e "${YELLOW}处理: $log_name${NC}"
    
    # 从log文件中提取输出路径
    # 匹配类似: output/qwen3-vl-2b-logo-lora_20251120_124305
    output_path=$(grep -oP 'output/qwen3-vl-2b-logo-lora[^/\s]+' "$log_file" | head -1)
    
    if [ -z "$output_path" ]; then
        echo -e "${RED}  ✗ 未找到输出路径,跳过${NC}"
        echo ""
        continue
    fi
    
    full_output_path="$BASE_DIR/$output_path"
    
    # 检查输出路径是否存在
    if [ ! -d "$full_output_path" ]; then
        echo -e "${RED}  ✗ 输出路径不存在: $full_output_path${NC}"
        echo ""
        continue
    fi
    
    # 找到最新的checkpoint
    # 匹配 checkpoint-数字 格式
    latest_checkpoint=$(ls -d "$full_output_path"/checkpoint-* 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$latest_checkpoint" ]; then
        echo -e "${RED}  ✗ 未找到checkpoint${NC}"
        echo ""
        continue
    fi
    
    checkpoint_name=$(basename "$latest_checkpoint")
    
    echo -e "${GREEN}  ✓ 找到输出路径: $output_path${NC}"
    echo -e "${GREEN}  ✓ 最新checkpoint: $checkpoint_name${NC}"
    
    # 从log文件名提取label
    # log_10k_vision_llm.log -> 10k_vision_llm
    label=$(echo "$log_name" | sed 's/^log_//')
    
    # 从log文件中提取训练参数
    lora_rank=$(grep -oP 'lora_rank["\s:=]+\K\d+' "$log_file" | head -1)
    train_samples=$(grep -oP '(?:dataset_size|train.*samples?)["\s:=]+\K\d+' "$log_file" | head -1)
    
    # 设置默认值
    if [ -z "$lora_rank" ]; then
        lora_rank=64
    fi
    if [ -z "$train_samples" ]; then
        # 尝试从文件名推断
        if [[ "$log_name" =~ ([0-9]+)k ]]; then
            num="${BASH_REMATCH[1]}"
            train_samples=$((num * 1000))
        else
            train_samples=30000
        fi
    fi
    
    output_dir="evaluation_results/$label"
    
    echo "  Label: $label"
    echo "  LoRA rank: $lora_rank"
    echo "  Train samples: $train_samples"
    echo "  Output dir: $output_dir"
    
    # 检查评估结果是否已存在
    if [ -d "$BASE_DIR/$output_dir" ] && [ -f "$BASE_DIR/$output_dir/results.json" ]; then
        echo -e "${YELLOW}  ⚠ 评估结果已存在,跳过${NC}"
        echo ""
        continue
    fi
    
    # 构建评估命令
    eval_cmd="cd $BASE_DIR && CUDA_VISIBLE_DEVICES=0 python $EVAL_SCRIPT \
  --checkpoint $latest_checkpoint \
  --label \"$label\" \
  --lora_rank $lora_rank \
  --train_samples $train_samples \
  --prompt v1 \
  --num_samples 300 \
  --output_dir $output_dir"
    
    echo ""
    echo -e "${GREEN}执行评估命令:${NC}"
    echo "$eval_cmd"
    echo ""
    
    # 执行评估
    eval "$eval_cmd"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ 评估完成!${NC}"
    else
        echo -e "${RED}  ✗ 评估失败!${NC}"
    fi
    
    echo ""
    echo "=========================================="
    echo ""
done

echo -e "${GREEN}所有评估完成!${NC}"
