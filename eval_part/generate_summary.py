"""
整合评估结果和训练信息
从evaluation_results目录读取所有评估结果，从log文件读取训练时间
生成综合CSV报告
"""

import json
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

def extract_training_info(log_file):
    """从log文件提取训练信息"""
    info = {
        'train_runtime': None,
        'train_samples': None,
        'epochs': None,
        'steps': None
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 提取训练时间
            match = re.search(r"'train_runtime':\s*([0-9.]+)", content)
            if match:
                info['train_runtime'] = float(match.group(1))
            
            # 提取训练样本数
            match = re.search(r"'num_train_epochs':\s*([0-9.]+)", content)
            if match:
                info['epochs'] = float(match.group(1))
            
            # 提取总步数
            match = re.search(r"'max_steps':\s*([0-9]+)", content)
            if match:
                info['steps'] = int(match.group(1))
            
            # 尝试从其他地方提取样本数
            match = re.search(r"dataset_size['\"]?\s*[:=]\s*([0-9]+)", content)
            if match:
                info['train_samples'] = int(match.group(1))
    except Exception as e:
        print(f"  警告: 读取{log_file}失败: {e}")
    
    return info

def format_time(seconds):
    """格式化时间"""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def main():
    base_dir = Path("/home/jiahuawang/test/classVLM")
    eval_dir = base_dir / "evaluation_results"
    log_dir = base_dir / "train_part" / "log"
    output_dir = base_dir / "eval_part"
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("整合评估结果和训练信息")
    print("=" * 60)
    print()
    
    # 收集所有评估结果
    results = []
    
    for result_folder in sorted(eval_dir.iterdir()):
        if not result_folder.is_dir():
            continue
        
        result_file = result_folder / "results.json"
        if not result_file.exists():
            continue
        
        print(f"处理: {result_folder.name}")
        
        try:
            # 读取评估结果
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            stats = data.get('stats', {})
            
            # 提取关键信息
            label = metadata.get('experiment_label', result_folder.name)
            lora_rank = metadata.get('lora_rank', 'N/A')
            
            # 从label中正确提取训练样本数
            train_samples = 'N/A'
            label_lower = label.lower()
            if '100k' in label_lower:
                train_samples = 100000
            elif '10k' in label_lower:
                train_samples = 10000
            elif '3k' in label_lower:
                train_samples = 3000
            elif '1k' in label_lower:
                train_samples = 1000
            elif '500' in label_lower:
                train_samples = 500
            
            # 提取性能指标
            cls_base = stats.get('classification_accuracy', {}).get('base', 0) * 100
            cls_lora = stats.get('classification_accuracy', {}).get('lora', 0) * 100
            cls_improve = cls_lora - cls_base
            
            det_base = stats.get('detection_iou', {}).get('base', 0) * 100
            det_lora = stats.get('detection_iou', {}).get('lora', 0) * 100
            det_improve = det_lora - det_base
            
            ind_base = stats.get('industry_accuracy', {}).get('base', 0) * 100
            ind_lora = stats.get('industry_accuracy', {}).get('lora', 0) * 100
            ind_improve = ind_lora - ind_base
            
            # 尝试从log文件读取训练信息
            log_name = f"log_{label}.log"
            log_file = log_dir / log_name
            
            # 推断batch size和lora target
            batch_size = 'N/A'
            lora_target = 'N/A'
            
            if 'batch4' in label.lower():
                batch_size = 4
            elif 'batch8' in label.lower():
                batch_size = 8
            elif '10k' in label.lower() and 'batch' not in label.lower():
                batch_size = 2  # 默认
            
            if 'mlp_llm' in label.lower():
                lora_target = 'MLP+LLM'
            elif 'vision_llm' in label.lower():
                lora_target = 'Vision+LLM'
            elif 'all' in label.lower():
                lora_target = 'All Layers'
            elif label == '10k':
                lora_target = 'LLM Only'
            
            train_info = {'train_runtime': None}
            if log_file.exists():
                train_info = extract_training_info(log_file)
                print(f"  ✓ 找到训练日志: {log_name}")
            else:
                # 尝试其他可能的log文件名
                possible_logs = list(log_dir.glob(f"*{label}*.log"))
                if possible_logs:
                    train_info = extract_training_info(possible_logs[0])
                    print(f"  ✓ 找到训练日志: {possible_logs[0].name}")
                else:
                    print(f"  ⚠ 未找到训练日志")
            
            results.append({
                'label': label,
                'lora_rank': lora_rank,
                'train_samples': train_samples,
                'train_runtime': train_info['train_runtime'],
                'train_runtime_formatted': format_time(train_info['train_runtime']),
                'epochs': train_info.get('epochs', 'N/A'),
                'steps': train_info.get('steps', 'N/A'),
                'cls_base': cls_base,
                'cls_lora': cls_lora,
                'cls_improve': cls_improve,
                'det_base': det_base,
                'det_lora': det_lora,
                'det_improve': det_improve,
                'ind_base': ind_base,
                'ind_lora': ind_lora,
                'ind_improve': ind_improve,
                'batch_size': batch_size,
                'lora_target': lora_target
            })
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
        
        print()
    
    if not results:
        print("未找到评估结果!")
        return
    
    # 保存到CSV
    csv_file = output_dir / "evaluation_summary.csv"
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            'Experiment',
            'LoRA Rank',
            'Train Samples',
            'Train Time (s)',
            'Train Time',
            'Epochs',
            'Steps',
            'Classification Baseline (%)',
            'Classification LoRA (%)',
            'Classification Improvement (%)',
            'Detection Baseline (%)',
            'Detection LoRA (%)',
            'Detection Improvement (%)',
            'Industry Baseline (%)',
            'Industry LoRA (%)',
            'Industry Improvement (%)',
            'Batch Size',
            'LoRA Target'
        ])
        
        # 写入数据
        for r in results:
            writer.writerow([
                r['label'],
                r['lora_rank'],
                r['train_samples'],
                r['train_runtime'] if r['train_runtime'] else 'N/A',
                r['train_runtime_formatted'],
                r['epochs'],
                r['steps'],
                f"{r['cls_base']:.2f}",
                f"{r['cls_lora']:.2f}",
                f"{r['cls_improve']:+.2f}",
                f"{r['det_base']:.2f}",
                f"{r['det_lora']:.2f}",
                f"{r['det_improve']:+.2f}",
                f"{r['ind_base']:.2f}",
                f"{r['ind_lora']:.2f}",
                f"{r['ind_improve']:+.2f}",
                r['batch_size'],
                r['lora_target']
            ])
    
    print("=" * 60)
    print(f"✓ CSV报告已生成: {csv_file}")
    print(f"✓ 共整合 {len(results)} 个实验结果")
    print("=" * 60)
    print()
    
    # 打印简要统计
    print("Summary Statistics:")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:20s} | "
              f"Cls:{r['cls_improve']:+6.2f}% | "
              f"Det:{r['det_improve']:+6.2f}% | "
              f"Ind:{r['ind_improve']:+6.2f}% | "
              f"Time:{r['train_runtime_formatted']}")
    print("-" * 60)

if __name__ == "__main__":
    main()
