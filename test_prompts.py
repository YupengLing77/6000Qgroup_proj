"""
测试不同提示词对模型性能的影响
比较 v1(原始)、v2(详细)、v3(简洁) 三种提示词版本
"""
import subprocess
import os
from datetime import datetime

def run_evaluation(checkpoint, label, lora_rank, train_samples, prompt_version, output_base="prompt_comparison"):
    """运行单次评估"""
    output_dir = f"{output_base}/{label}_prompt_{prompt_version}"
    
    cmd = [
        "python", "comprehensive_eval.py",
        "--checkpoint", checkpoint,
        "--label", f"{label}_prompt_{prompt_version}",
        "--lora_rank", str(lora_rank),
        "--train_samples", str(train_samples),
        "--prompt", prompt_version,
        "--test_json", "test_subset.json",
        "--num_samples", "30",
        "--output_dir", output_dir
    ]
    
    print(f"\n{'='*70}")
    print(f"运行评估: {label} - 提示词版本: {prompt_version}")
    print(f"{'='*70}")
    
    subprocess.run(cmd, check=True)
    return output_dir

def compare_prompt_versions(checkpoint, label, lora_rank, train_samples):
    """比较三种提示词版本的性能"""
    import json
    import csv
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"prompt_comparison_{timestamp}"
    os.makedirs(output_base, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"开始提示词对比实验")
    print(f"{'='*70}")
    print(f"实验标签: {label}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"训练样本: {train_samples:,}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*70}\n")
    
    # 运行三个版本的评估
    results_all = {}
    for version in ['v1', 'v2', 'v3']:
        output_dir = run_evaluation(checkpoint, label, lora_rank, train_samples, version, output_base)
        
        # 读取结果
        result_file = os.path.join(output_dir, 'results.json')
        with open(result_file, 'r', encoding='utf-8') as f:
            results_all[version] = json.load(f)
    
    # 生成对比报告
    generate_comparison_report(results_all, output_base, label, lora_rank, train_samples)
    
    print(f"\n{'='*70}")
    print(f"✅ 提示词对比实验完成！")
    print(f"{'='*70}")
    print(f"结果保存在: {output_base}/")
    print(f"  - comparison_table.csv")
    print(f"  - comparison_report.md")
    print(f"{'='*70}\n")

def generate_comparison_report(results_all, output_base, label, lora_rank, train_samples):
    """生成对比报告"""
    import csv
    
    # CSV 报告
    csv_path = os.path.join(output_base, 'comparison_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 实验配置
        writer.writerow(['实验配置'])
        writer.writerow(['标签', label])
        writer.writerow(['LoRA Rank', lora_rank])
        writer.writerow(['训练样本', train_samples])
        writer.writerow([])
        
        # 表头
        writer.writerow(['任务', '指标', 'v1_原始_base', 'v1_原始_lora', 
                        'v2_详细_base', 'v2_详细_lora', 
                        'v3_简洁_base', 'v3_简洁_lora'])
        
        # 分类任务
        v1_cls_b = results_all['v1']['stats']['classification_accuracy']['base'] * 100
        v1_cls_l = results_all['v1']['stats']['classification_accuracy']['lora'] * 100
        v2_cls_b = results_all['v2']['stats']['classification_accuracy']['base'] * 100
        v2_cls_l = results_all['v2']['stats']['classification_accuracy']['lora'] * 100
        v3_cls_b = results_all['v3']['stats']['classification_accuracy']['base'] * 100
        v3_cls_l = results_all['v3']['stats']['classification_accuracy']['lora'] * 100
        
        writer.writerow(['分类', '准确率(%)', 
                        f'{v1_cls_b:.1f}', f'{v1_cls_l:.1f}',
                        f'{v2_cls_b:.1f}', f'{v2_cls_l:.1f}',
                        f'{v3_cls_b:.1f}', f'{v3_cls_l:.1f}'])
        
        # 检测任务
        v1_det_b = results_all['v1']['stats']['detection_iou']['base'] * 100
        v1_det_l = results_all['v1']['stats']['detection_iou']['lora'] * 100
        v2_det_b = results_all['v2']['stats']['detection_iou']['base'] * 100
        v2_det_l = results_all['v2']['stats']['detection_iou']['lora'] * 100
        v3_det_b = results_all['v3']['stats']['detection_iou']['base'] * 100
        v3_det_l = results_all['v3']['stats']['detection_iou']['lora'] * 100
        
        writer.writerow(['检测', 'IoU(%)', 
                        f'{v1_det_b:.1f}', f'{v1_det_l:.1f}',
                        f'{v2_det_b:.1f}', f'{v2_det_l:.1f}',
                        f'{v3_det_b:.1f}', f'{v3_det_l:.1f}'])
        
        # 行业识别
        v1_ind_b = results_all['v1']['stats']['industry_accuracy']['base'] * 100
        v1_ind_l = results_all['v1']['stats']['industry_accuracy']['lora'] * 100
        v2_ind_b = results_all['v2']['stats']['industry_accuracy']['base'] * 100
        v2_ind_l = results_all['v2']['stats']['industry_accuracy']['lora'] * 100
        v3_ind_b = results_all['v3']['stats']['industry_accuracy']['base'] * 100
        v3_ind_l = results_all['v3']['stats']['industry_accuracy']['lora'] * 100
        
        writer.writerow(['行业识别', '准确率(%)', 
                        f'{v1_ind_b:.1f}', f'{v1_ind_l:.1f}',
                        f'{v2_ind_b:.1f}', f'{v2_ind_l:.1f}',
                        f'{v3_ind_b:.1f}', f'{v3_ind_l:.1f}'])
        
        writer.writerow([])
        writer.writerow(['提升对比 (LoRA vs Base)'])
        writer.writerow(['任务', '指标', 'v1_提升', 'v2_提升', 'v3_提升', '最佳'])
        
        cls_v1_gain = v1_cls_l - v1_cls_b
        cls_v2_gain = v2_cls_l - v2_cls_b
        cls_v3_gain = v3_cls_l - v3_cls_b
        best_cls = max([('v1', cls_v1_gain), ('v2', cls_v2_gain), ('v3', cls_v3_gain)], key=lambda x: x[1])
        
        writer.writerow(['分类', '准确率提升(%)', 
                        f'{cls_v1_gain:+.1f}', f'{cls_v2_gain:+.1f}', f'{cls_v3_gain:+.1f}',
                        f'{best_cls[0]} ({best_cls[1]:+.1f}%)'])
        
        det_v1_gain = v1_det_l - v1_det_b
        det_v2_gain = v2_det_l - v2_det_b
        det_v3_gain = v3_det_l - v3_det_b
        best_det = max([('v1', det_v1_gain), ('v2', det_v2_gain), ('v3', det_v3_gain)], key=lambda x: x[1])
        
        writer.writerow(['检测', 'IoU提升(%)', 
                        f'{det_v1_gain:+.1f}', f'{det_v2_gain:+.1f}', f'{det_v3_gain:+.1f}',
                        f'{best_det[0]} ({best_det[1]:+.1f}%)'])
        
        ind_v1_gain = v1_ind_l - v1_ind_b
        ind_v2_gain = v2_ind_l - v2_ind_b
        ind_v3_gain = v3_ind_l - v3_ind_b
        best_ind = max([('v1', ind_v1_gain), ('v2', ind_v2_gain), ('v3', ind_v3_gain)], key=lambda x: x[1])
        
        writer.writerow(['行业识别', '准确率提升(%)', 
                        f'{ind_v1_gain:+.1f}', f'{ind_v2_gain:+.1f}', f'{ind_v3_gain:+.1f}',
                        f'{best_ind[0]} ({best_ind[1]:+.1f}%)'])
    
    # Markdown 报告
    md_path = os.path.join(output_base, 'comparison_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 提示词版本对比实验报告\n\n")
        f.write("## 实验配置\n\n")
        f.write(f"- **实验标签**: {label}\n")
        f.write(f"- **LoRA Rank**: {lora_rank}\n")
        f.write(f"- **训练样本数**: {train_samples:,}\n\n")
        
        f.write("## 提示词版本说明\n\n")
        f.write("- **v1 (原始)**: 使用数据集原始提示词\n")
        f.write("- **v2 (详细)**: 更详细的指令描述\n")
        f.write("- **v3 (简洁)**: 极简化的提示词\n\n")
        
        f.write("## 性能对比表\n\n")
        f.write("| 任务 | 指标 | v1_base | v1_lora | v2_base | v2_lora | v3_base | v3_lora |\n")
        f.write("|------|------|---------|---------|---------|---------|---------|----------|\n")
        f.write(f"| 分类 | 准确率(%) | {v1_cls_b:.1f} | {v1_cls_l:.1f} | {v2_cls_b:.1f} | {v2_cls_l:.1f} | {v3_cls_b:.1f} | {v3_cls_l:.1f} |\n")
        f.write(f"| 检测 | IoU(%) | {v1_det_b:.1f} | {v1_det_l:.1f} | {v2_det_b:.1f} | {v2_det_l:.1f} | {v3_det_b:.1f} | {v3_det_l:.1f} |\n")
        f.write(f"| 行业 | 准确率(%) | {v1_ind_b:.1f} | {v1_ind_l:.1f} | {v2_ind_b:.1f} | {v2_ind_l:.1f} | {v3_ind_b:.1f} | {v3_ind_l:.1f} |\n\n")
        
        f.write("## LoRA 提升对比\n\n")
        f.write("| 任务 | v1提升 | v2提升 | v3提升 | 最佳版本 |\n")
        f.write("|------|--------|--------|--------|----------|\n")
        f.write(f"| 分类 | {cls_v1_gain:+.1f}% | {cls_v2_gain:+.1f}% | {cls_v3_gain:+.1f}% | **{best_cls[0]}** ({best_cls[1]:+.1f}%) |\n")
        f.write(f"| 检测 | {det_v1_gain:+.1f}% | {det_v2_gain:+.1f}% | {det_v3_gain:+.1f}% | **{best_det[0]}** ({best_det[1]:+.1f}%) |\n")
        f.write(f"| 行业 | {ind_v1_gain:+.1f}% | {ind_v2_gain:+.1f}% | {ind_v3_gain:+.1f}% | **{best_ind[0]}** ({best_ind[1]:+.1f}%) |\n\n")
        
        f.write("## 结论\n\n")
        f.write("提示词工程对模型性能有显著影响。根据上述结果：\n\n")
        f.write(f"- **分类任务**: {best_cls[0]} 版本表现最佳，提升 {best_cls[1]:.1f}%\n")
        f.write(f"- **检测任务**: {best_det[0]} 版本表现最佳，提升 {best_det[1]:.1f}%\n")
        f.write(f"- **行业识别**: {best_ind[0]} 版本表现最佳，提升 {best_ind[1]:.1f}%\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试不同提示词对性能的影响')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='LoRA checkpoint 路径')
    parser.add_argument('--label', type=str, required=True, 
                        help='实验标签，如 exp1_10k_r64')
    parser.add_argument('--lora_rank', type=int, default=64, 
                        help='LoRA rank')
    parser.add_argument('--train_samples', type=int, default=10000, 
                        help='训练样本数量')
    
    args = parser.parse_args()
    
    compare_prompt_versions(
        checkpoint=args.checkpoint,
        label=args.label,
        lora_rank=args.lora_rank,
        train_samples=args.train_samples
    )
