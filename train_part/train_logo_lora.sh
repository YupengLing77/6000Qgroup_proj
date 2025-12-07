#!/bin/bash
# Qwen3-VL-2B LoRA 微调脚本 - Logo 检测任务

# 抑制 multiprocess 警告（Python 3.12 兼容性问题）
export PYTHONWARNINGS="ignore::UserWarning"

cd /home/jiahuawang/test/classVLM

# ============================================
# 数据集配置 - 在这里修改你要使用的数据文件
# ============================================
TRAIN_JSON="train_subset_3k.json"  # 训练数据，可选: train_subset.json 或 logo_train.json
TEST_JSON="test_subset_1k.json"    # 测试数据，可选: test_subset.json 或 logo_test.json

# GPU 配置
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

# ============================================
# LoRA 配置
# ============================================
LORA_R=64  # LoRA rank，越大效果越好但显存占用越多
LORA_ALPHA=128  # LoRA alpha，通常设为 r 的 2 倍
LORA_DROPOUT=0.05

# ============================================
# 训练策略选择（调整下面三个参数实现不同训练方式）
# ============================================
# 策略1: 标准 LoRA（当前配置）
#   - tune_mm_vision=False  不训练视觉编码器
#   - tune_mm_mlp=False     不训练多模态投影层
#   - tune_mm_llm=True      只用 LoRA 训练语言模型
#   特点: 参数少，训练快，适合数据量中等
#
# 策略2: 视觉+LoRA（增强视觉理解）
#   - tune_mm_vision=True   全参数微调视觉编码器
#   - tune_mm_mlp=False     不训练投影层
#   - tune_mm_llm=True      LoRA 训练语言模型
#   特点: 提升视觉特征提取，适合图像复杂任务
#
# 策略3: 投影层+LoRA（平衡方案）
#   - tune_mm_vision=False  冻住视觉编码器
#   - tune_mm_mlp=True      全参数微调投影层
#   - tune_mm_llm=True      LoRA 训练语言模型
#   特点: 调整模态对齐，参数量适中
#
# 策略4: 全量微调（最强但最慢）
#   - tune_mm_vision=True   训练视觉编码器
#   - tune_mm_mlp=True      训练投影层
#   - tune_mm_llm=True      LoRA 训练语言模型
#   特点: 效果最好，但显存和时间消耗大
#
# 策略5: 纯 LoRA（最轻量）
#   - lora_enable=True
#   - 其他全 False
#   特点: 最省显存，适合快速实验
# ============================================
export CUDA_VISIBLE_DEVICES=2  # 单卡训练，多卡改为 0,1,2,3
TUNE_VISION=False  # 是否训练视觉编码器
TUNE_MLP=False     # 是否训练多模态投影层
TUNE_LLM=True      # 是否用 LoRA 训练语言模型

# 训练超参数
LR=1e-5  # Qwen3-VL 建议学习率 1e-5 到 2e-5
BATCH_SIZE=64  # 每个 GPU 的 batch size
GRAD_ACCUM=4  # 梯度累积，实际 batch size = BATCH_SIZE * GRAD_ACCUM * GPU数

# ============================================
# Epoch 配置建议（根据数据集大小调整）
# ============================================
# 数据规模          建议 Epoch    说明
# 1K 样本 (3K)      5-10         小数据需要更多轮数，避免欠拟合
# 10K 样本 (30K)    3-5          中等数据，平衡训练时间和效果
# 50K 样本 (150K)   2-3          大数据集，少量轮数即可
# 126K 样本 (380K)  1-2          全量数据，1-2 轮足够
#
# 原则：数据越多，epoch 越少；数据越少，epoch 越多
# ============================================
EPOCHS=50  # 训练轮数 - 根据上述建议修改

# DeepSpeed 配置（单卡用 zero2，多卡可用 zero3）
DEEPSPEED_CONFIG="./Qwen3-VL/qwen-vl-finetune/scripts/zero2.json"

# 训练脚本路径
TRAIN_SCRIPT="./Qwen3-VL/qwen-vl-finetune/qwenvl/train/train_qwen.py"

# 更新数据集配置文件中的路径
echo "=========================================="
echo "训练配置"
echo "=========================================="
echo "正在配置数据集路径..."
PROJECT_ROOT=$(pwd)
# sed -i "s|\"annotation_path\": \".*train_subset.json\"|\"annotation_path\": \"${PROJECT_ROOT}/train_part/${TRAIN_JSON}\"|g" ./Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py
# sed -i "s|\"annotation_path\": \".*logo_train.json\"|\"annotation_path\": \"${PROJECT_ROOT}/train_part/${TRAIN_JSON}\"|g" ./Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py
echo "✅ 训练数据: ${PROJECT_ROOT}/train_part/${TRAIN_JSON}"
echo "✅ 测试数据: ${PROJECT_ROOT}/train_part/${TEST_JSON}"

# 统计样本数量并给出建议
if [ -f "${PROJECT_ROOT}/train_part/${TRAIN_JSON}" ]; then
    SAMPLE_COUNT=$(grep -o '"image"' "${PROJECT_ROOT}/train_part/${TRAIN_JSON}" | wc -l)
    echo "📊 训练样本数: ${SAMPLE_COUNT}"
    
    # 根据样本数给出 Epoch 建议
    if [ $SAMPLE_COUNT -lt 5000 ]; then
        SUGGESTED_EPOCHS="5-10"
        REASON="小数据集，需要更多轮数"
    elif [ $SAMPLE_COUNT -lt 50000 ]; then
        SUGGESTED_EPOCHS="3-5"
        REASON="中等数据集，平衡效果与时间"
    elif [ $SAMPLE_COUNT -lt 200000 ]; then
        SUGGESTED_EPOCHS="2-3"
        REASON="大数据集，少量轮数即可"
    else
        SUGGESTED_EPOCHS="1-2"
        REASON="超大数据集，1-2轮足够"
    fi
    
    echo "💡 建议 Epoch: ${SUGGESTED_EPOCHS} (${REASON})"
    echo "⚙️  当前 Epoch: ${EPOCHS}"
    
    if [ $EPOCHS -lt $(echo $SUGGESTED_EPOCHS | cut -d'-' -f1) ] || [ $EPOCHS -gt $(echo $SUGGESTED_EPOCHS | cut -d'-' -f2) ]; then
        echo "⚠️  警告: 当前 Epoch 可能不在最佳范围内"
    fi
fi

echo "🎯 训练策略:"
echo "  - 视觉编码器: $([ "$TUNE_VISION" = "True" ] && echo "训练 ✅" || echo "冻结 ❄️")"
echo "  - 投影层: $([ "$TUNE_MLP" = "True" ] && echo "训练 ✅" || echo "冻结 ❄️")"
echo "  - 语言模型: $([ "$TUNE_LLM" = "True" ] && echo "LoRA ✅" || echo "冻结 ❄️")"
echo "=========================================="
echo ""

# 启动训练
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $TRAIN_SCRIPT \
         --deepspeed $DEEPSPEED_CONFIG \
         --model_name_or_path $MODEL_PATH \
         --dataset_use $DATASETS \
         --data_flatten True \
         --tune_mm_vision $TUNE_VISION \
         --tune_mm_mlp $TUNE_MLP \
         --tune_mm_llm $TUNE_LLM \
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