# 运行脚本：
python prepare_data.py --train 10000 --test 1000
bash train_part/train_logo_lora.sh

评估：
python comprehensive_eval.py \
  --checkpoint ./output/qwen3-vl-2b-logo-lora_20251119_143052/checkpoint-500 \
  --label "exp1_10k_r64" \
  --lora_rank 64 \
  --train_samples 10000 \
  --prompt v1 \
  --num_samples 30

提示词对比：
python test_prompts.py \
  --checkpoint ./output/qwen3-vl-2b-logo-lora_20251119_143052/checkpoint-500 \
  --label "exp1_10k_r64" \
  --lora_rank 64 \
  --train_samples 10000
# 第一轮测试：
训练数据量： 1k 图片 lora 微调
训练参数：
LORA_R = 64              # LoRA 秩
LORA_ALPHA = 128         # LoRA alpha（2倍r）
LORA_DROPOUT = 0.05      # Dropout率


学习率 (LR) = 1e-5       # Qwen3-VL 推荐值
Batch Size = 4           # 每GPU batch size
梯度累积 = 4              # 实际batch = 4×4 = 16
训练轮数 = 3 epochs
总步数 = 564 steps       # (3000样本 / 16) × 3轮

优化器 = AdamW
调度器 = Cosine
Warmup = 3%
权重衰减 = 0.01


分类准确率: 30.0% → 100.0%
检测 IoU: 3.6% → 20.5%
行业识别: 30.0% → 60.0%


