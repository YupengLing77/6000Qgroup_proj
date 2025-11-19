1 下载数据集
 https://github.com/Wangjing1551/LogoDet-3K-Dataset 到 logodata文件夹
2  用 convert_logo_data.py 转换成qwen3 训练格式
    会生成 logo_images 和 logo_test.json logo_train.json
3  运行prepare_data.py
    python prepare_data.py --train 10000 --test 1000
    可以调整数据集大小
    生成 train_subset.json test_subset.json
    修改数据集路径：
4 运行训练脚本
    bash train_part/train_logo_lora.sh
    训练脚本里有超参数可以修改
5 一些评测
    python comprehensive_eval.py \
    --checkpoint ./output/qwen3-vl-2b-logo-lora_20251119_143052/checkpoint-500 \
    --label "exp1_10k_r64" \
    --lora_rank 64 \
    --train_samples 10000 \
    --prompt v1 \
    --num_samples 30