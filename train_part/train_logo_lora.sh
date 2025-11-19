#!/bin/bash
# Qwen3-VL-2B LoRA 微调脚本 - Logo 检测任务

# 抑制 multiprocess 警告（Python 3.12 兼容性问题）
export PYTHONWARNINGS="ignore::UserWarning"

cd /home/jiahuawang/test/classVLM
# GPU 配置
export CUDA_VISIBLE_DEVICES=0  # 单卡训练，多卡改为 0,1,2,3
NPROC_PER_NODE=1  # GPU 数量
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20001-29999 -n 1)

# 模型路径 - Qwen3-VL-2B-Instruct
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"  # HuggingFace 路径，或本地路径

# 数据配置
DATASETS="logo_dataset%100"  # 使用 100% 的 logo 数据

# 输出配置 - 使用时间戳避免覆盖
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./output/qwen3-vl-2b-logo-lora_${TIMESTAMP}"
RUN_NAME="qwen3-vl-2b-logo-lora_${TIMESTAMP}"

# LoRA 配置
LORA_R=64  # LoRA rank，越大效果越好但显存占用越多
LORA_ALPHA=128  # LoRA alpha，通常设为 r 的 2 倍
LORA_DROPOUT=0.05

# 训练超参数
LR=1e-5  # Qwen3-VL 建议学习率 1e-5 到 2e-5
BATCH_SIZE=8  # 每个 GPU 的 batch size
GRAD_ACCUM=4  # 梯度累积，实际 batch size = BATCH_SIZE * GRAD_ACCUM * GPU数
EPOCHS=3  # 训练轮数

# DeepSpeed 配置（单卡用 zero2，多卡可用 zero3）
DEEPSPEED_CONFIG="./Qwen3-VL/qwen-vl-finetune/scripts/zero2.json"

# 训练脚本路径
TRAIN_SCRIPT="./Qwen3-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py"

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $TRAIN_SCRIPT \
         --deepspeed $DEEPSPEED_CONFIG \
         --model_name_or_path $MODEL_PATH \
         --dataset_use $DATASETS \
         --data_flatten True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --tune_mm_llm True \
         --lora_enable True \
         --lora_r $LORA_R \
         --lora_alpha $LORA_ALPHA \
         --lora_dropout $LORA_DROPOUT \
         --bf16 \
         --output_dir $OUTPUT_DIR \
         --num_train_epochs $EPOCHS \
         --per_device_train_batch_size $BATCH_SIZE \
         --gradient_accumulation_steps $GRAD_ACCUM \
         --learning_rate $LR \
         --weight_decay 0.01 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --logging_steps 10 \
         --save_steps 500 \
         --save_total_limit 2 \
         --max_pixels 50176 \
         --min_pixels 784 \
         --model_max_length 8192 \
         --gradient_checkpointing True \
         --dataloader_num_workers 4 \
         --run_name $RUN_NAME \
         --report_to tensorboard

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo "模型保存在: $OUTPUT_DIR"
echo "LoRA 权重可以直接用于推理"
echo ""
echo "使用方法："
echo "1. 基础模型: Qwen/Qwen3-VL-2B-Instruct"
echo "2. LoRA 权重: $OUTPUT_DIR/checkpoint-XXX"
echo "3. 推理: python inference_logo.py <图像路径>"