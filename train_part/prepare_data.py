"""
训练配置：从已转换的数据集中选择指定数量进行训练
不需要重复转换数据，只需修改这里的配置
"""
import json
import random
import argparse

def prepare_training_data(
    full_train_json="logo_train.json",
    full_test_json="logo_test.json",
    output_train_json="train_subset.json",
    output_test_json="test_subset.json",
    num_train_images=10000,  # 修改这里选择训练图像数量
    num_test_images=1000,    # 修改这里选择测试图像数量
    seed=42
):
    """
    从全量数据中选择指定数量的图像
    
    Args:
        num_train_images: 训练图像数量（1000, 10000, 126923全量）
        num_test_images: 测试图像数量
    """
    random.seed(seed)
    
    # 加载全量数据
    print("加载全量数据...")
    with open(full_train_json, 'r') as f:
        full_train = json.load(f)
    with open(full_test_json, 'r') as f:
        full_test = json.load(f)
    
    print(f"全量训练样本: {len(full_train)} 条")
    print(f"全量测试样本: {len(full_test)} 条")
    
    # 按图像分组（每张图有3个任务：分类、检测、行业）
    train_by_image = {}
    for sample in full_train:
        img = sample['image']
        if img not in train_by_image:
            train_by_image[img] = []
        train_by_image[img].append(sample)
    
    test_by_image = {}
    for sample in full_test:
        img = sample['image']
        if img not in test_by_image:
            test_by_image[img] = []
        test_by_image[img].append(sample)
    
    print(f"\n全量训练图像: {len(train_by_image)} 张")
    print(f"全量测试图像: {len(test_by_image)} 张")
    
    # 随机选择指定数量的图像
    train_images = list(train_by_image.keys())
    test_images = list(test_by_image.keys())
    
    random.shuffle(train_images)
    random.shuffle(test_images)
    
    selected_train_images = train_images[:num_train_images]
    selected_test_images = test_images[:num_test_images]
    
    # 收集选中图像的所有样本
    train_subset = []
    for img in selected_train_images:
        train_subset.extend(train_by_image[img])
    
    test_subset = []
    for img in selected_test_images:
        test_subset.extend(test_by_image[img])
    
    # 保存子集
    with open(output_train_json, 'w', encoding='utf-8') as f:
        json.dump(train_subset, f, ensure_ascii=False, indent=2)
    
    with open(output_test_json, 'w', encoding='utf-8') as f:
        json.dump(test_subset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 数据准备完成!")
    print(f"训练图像: {len(selected_train_images)} 张")
    print(f"训练样本: {len(train_subset)} 条")
    print(f"测试图像: {len(selected_test_images)} 张")
    print(f"测试样本: {len(test_subset)} 条")
    print(f"\n保存为:")
    print(f"  - {output_train_json}")
    print(f"  - {output_test_json}")
    
    return {
        'train_images': len(selected_train_images),
        'train_samples': len(train_subset),
        'test_images': len(selected_test_images),
        'test_samples': len(test_subset)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备训练数据子集')
    parser.add_argument('--train', type=int, default=10000, 
                        help='训练图像数量 (默认: 10000, 全量: 126923)')
    parser.add_argument('--test', type=int, default=1000,
                        help='测试图像数量 (默认: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"准备训练数据")
    print(f"{'='*60}")
    print(f"训练图像: {args.train:,} 张")
    print(f"测试图像: {args.test:,} 张")
    print(f"随机种子: {args.seed}")
    print(f"{'='*60}\n")
    
    # 准备数据
    result = prepare_training_data(
        num_train_images=args.train,
        num_test_images=args.test,
        seed=args.seed
    )
    
    print(f"\n{'='*60}")
    print("✅ 数据准备完成！")
    print(f"{'='*60}")
    print(f"训练样本: {result['train_samples']:,} 条 ({result['train_images']:,} 张图)")
    print(f"测试样本: {result['test_samples']:,} 条 ({result['test_images']:,} 张图)")
    print(f"\n下一步: bash train_logo_lora.sh")
