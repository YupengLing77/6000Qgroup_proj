"""
综合评估：测试分类、检测、行业识别三个任务
生成完整的对比报告
"""
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
import json
import re
import os
from tqdm import tqdm
import base64
from io import BytesIO

def parse_bbox(response_text):
    """从模型回答中提取 bbox 坐标"""
    json_match = re.search(r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
    if json_match:
        return [int(x) for x in json_match.groups()]
    array_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', response_text)
    if array_match:
        return [int(x) for x in array_match.groups()]
    return None

def calculate_iou(box1, box2):
    """计算 IoU"""
    if not box1 or not box2:
        return 0.0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def check_classification_match(pred, gt):
    """检查分类是否正确（模糊匹配）"""
    pred_lower = pred.lower()
    gt_lower = gt.lower()
    
    # 提取行业和公司名
    industry_match = "industry" in gt_lower and any(word in pred_lower for word in gt_lower.split() if len(word) > 3)
    company_match = "company" in gt_lower and any(word in pred_lower for word in gt_lower.split() if len(word) > 3)
    
    return industry_match or company_match or (gt_lower in pred_lower)

def comprehensive_evaluation(
    test_json,
    base_model_path,
    lora_checkpoint,
    num_samples=20,
    output_dir="comprehensive_eval",
    experiment_label="",
    lora_rank=64,
    train_samples=0,
    prompt_version="v1"
):
    """综合评估三个任务
    
    Args:
        experiment_label: 实验标签，如 "exp1_10k_r64"
        lora_rank: LoRA rank 大小
        train_samples: 训练样本数量
        prompt_version: 提示词版本 (v1, v2, v3)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验元数据
    metadata = {
        "experiment_label": experiment_label,
        "lora_rank": lora_rank,
        "train_samples": train_samples,
        "prompt_version": prompt_version,
        "base_model": base_model_path,
        "lora_checkpoint": lora_checkpoint,
        "test_samples": num_samples
    }
    
    # 加载数据
    with open(test_json, 'r') as f:
        all_data = json.load(f)
    
    # 按任务类型分组
    tasks = {
        'classification': [],  # 包含 "Identify"
        'detection': [],       # 包含 "Locate"
        'industry': []         # 包含 "industry"
    }
    
    for sample in all_data:
        question = sample['conversations'][0]['value']
        if 'Identify' in question and 'industry and company' in question:
            tasks['classification'].append(sample)
        elif 'Locate' in question or 'bbox' in question.lower():
            tasks['detection'].append(sample)
        elif 'industry' in question.lower() and 'What' in question:
            tasks['industry'].append(sample)
    
    print(f"任务分布:")
    print(f"  分类任务: {len(tasks['classification'])} 条")
    print(f"  检测任务: {len(tasks['detection'])} 条")
    print(f"  行业识别: {len(tasks['industry'])} 条")
    
    # 每个任务取相同数量样本
    samples_per_task = num_samples // 3
    test_samples = (
        tasks['classification'][:samples_per_task] +
        tasks['detection'][:samples_per_task] +
        tasks['industry'][:samples_per_task]
    )
    
    # 加载模型
    print("\n加载基础模型...")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    base_processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    base_model.eval()
    
    print("加载 LoRA 模型...")
    lora_base = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(lora_base, lora_checkpoint)
    lora_processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    lora_model.eval()
    
    # 评估
    results = []
    task_stats = {
        'classification': {'base_correct': 0, 'lora_correct': 0, 'total': 0},
        'detection': {'base_iou': [], 'lora_iou': [], 'total': 0},
        'industry': {'base_correct': 0, 'lora_correct': 0, 'total': 0}
    }
    
    for i, sample in enumerate(tqdm(test_samples, desc="评估中")):
        image_path = sample['image']
        question = sample['conversations'][0]['value'].replace('<image>\n', '')
        gt_answer = sample['conversations'][1]['value']
        
        # 判断任务类型
        if 'Identify' in question:
            task_type = 'classification'
        elif 'Locate' in question:
            task_type = 'detection'
        else:
            task_type = 'industry'
        
        image = Image.open(image_path).convert("RGB")
        
        # 根据提示词版本生成问题
        if prompt_version == "v1":
            prompt_question = question  # 原始问题
        elif prompt_version == "v2":
            # 更详细的提示词
            if task_type == 'classification':
                prompt_question = "Analyze this image carefully. Identify the logo and provide the industry category and company name."
            elif task_type == 'detection':
                prompt_question = question.replace("Locate", "Please carefully locate").replace("output", "provide precise")
            else:
                prompt_question = "What is the specific industry category of this logo? Please analyze and answer."
        elif prompt_version == "v3":
            # 简洁提示词
            if task_type == 'classification':
                prompt_question = "What is this logo? Industry and company?"
            elif task_type == 'detection':
                prompt_question = question.replace("Locate the", "Find").replace("in this image and output the bbox coordinates in JSON format", "bbox")
            else:
                prompt_question = "Industry?"
        else:
            prompt_question = question
        
        # 预测
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_question}]}]
        
        # 基础模型
        text = base_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = base_processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            base_output = base_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        base_response = base_processor.batch_decode(base_output, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        
        # LoRA 模型
        text = lora_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = lora_processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            lora_output = lora_model.generate(**inputs, max_new_tokens=256, do_sample=False)
        lora_response = lora_processor.batch_decode(lora_output, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        
        # 评估
        if task_type == 'detection':
            gt_bbox = parse_bbox(gt_answer)
            base_bbox = parse_bbox(base_response)
            lora_bbox = parse_bbox(lora_response)
            base_iou = calculate_iou(gt_bbox, base_bbox)
            lora_iou = calculate_iou(gt_bbox, lora_bbox)
            task_stats['detection']['base_iou'].append(base_iou)
            task_stats['detection']['lora_iou'].append(lora_iou)
            task_stats['detection']['total'] += 1
            metric = {'base_iou': base_iou, 'lora_iou': lora_iou}
        else:
            base_correct = check_classification_match(base_response, gt_answer)
            lora_correct = check_classification_match(lora_response, gt_answer)
            if base_correct:
                task_stats[task_type]['base_correct'] += 1
            if lora_correct:
                task_stats[task_type]['lora_correct'] += 1
            task_stats[task_type]['total'] += 1
            metric = {'base_correct': base_correct, 'lora_correct': lora_correct}
        
        results.append({
            'id': i,
            'task': task_type,
            'image': image_path,
            'question': question,
            'gt_answer': gt_answer,
            'base_response': base_response,
            'lora_response': lora_response,
            'metric': metric
        })
    
    # 计算统计
    stats_summary = {
        'classification_accuracy': {
            'base': task_stats['classification']['base_correct'] / max(task_stats['classification']['total'], 1),
            'lora': task_stats['classification']['lora_correct'] / max(task_stats['classification']['total'], 1)
        },
        'detection_iou': {
            'base': sum(task_stats['detection']['base_iou']) / max(len(task_stats['detection']['base_iou']), 1),
            'lora': sum(task_stats['detection']['lora_iou']) / max(len(task_stats['detection']['lora_iou']), 1)
        },
        'industry_accuracy': {
            'base': task_stats['industry']['base_correct'] / max(task_stats['industry']['total'], 1),
            'lora': task_stats['industry']['lora_correct'] / max(task_stats['industry']['total'], 1)
        }
    }
    
    # 保存结果
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'stats': stats_summary,
            'details': results
        }, f, ensure_ascii=False, indent=2)
    
    # 生成表格报告
    generate_table_report(stats_summary, results, output_dir, metadata)
    
    return stats_summary, results

def generate_table_report(stats, results, output_dir, metadata=None):
    """生成表格报告（CSV + Markdown）"""
    import csv
    
    # CSV 报告
    csv_path = os.path.join(output_dir, 'evaluation_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入实验配置
        if metadata:
            writer.writerow(['实验配置'])
            writer.writerow(['标签', metadata.get('experiment_label', 'N/A')])
            writer.writerow(['LoRA Rank', metadata.get('lora_rank', 'N/A')])
            writer.writerow(['训练样本', metadata.get('train_samples', 'N/A')])
            writer.writerow(['提示词版本', metadata.get('prompt_version', 'v1')])
            writer.writerow([])  # 空行
        
        writer.writerow(['任务', '指标', '基础模型', 'LoRA模型', '提升'])
        
        # 分类
        base_cls = stats['classification_accuracy']['base'] * 100
        lora_cls = stats['classification_accuracy']['lora'] * 100
        writer.writerow(['分类', '准确率(%)', f'{base_cls:.1f}', f'{lora_cls:.1f}', f'{lora_cls - base_cls:+.1f}'])
        
        # 检测
        base_iou = stats['detection_iou']['base'] * 100
        lora_iou = stats['detection_iou']['lora'] * 100
        writer.writerow(['检测', 'IoU(%)', f'{base_iou:.1f}', f'{lora_iou:.1f}', f'{lora_iou - base_iou:+.1f}'])
        
        # 行业
        base_ind = stats['industry_accuracy']['base'] * 100
        lora_ind = stats['industry_accuracy']['lora'] * 100
        writer.writerow(['行业识别', '准确率(%)', f'{base_ind:.1f}', f'{lora_ind:.1f}', f'{lora_ind - base_ind:+.1f}'])
    
    # Markdown 报告
    md_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Qwen3-VL-2B Logo 识别 LoRA 微调评估报告\n\n")
        
        # 写入实验配置
        if metadata:
            f.write("## 实验配置\n\n")
            f.write(f"- **实验标签**: {metadata.get('experiment_label', 'N/A')}\n")
            f.write(f"- **LoRA Rank**: {metadata.get('lora_rank', 'N/A')}\n")
            f.write(f"- **训练样本数**: {metadata.get('train_samples', 'N/A'):,}\n")
            f.write(f"- **提示词版本**: {metadata.get('prompt_version', 'v1')}\n")
            f.write(f"- **Checkpoint**: {metadata.get('lora_checkpoint', 'N/A')}\n\n")
        
        f.write("## 总体性能对比\n\n")
        f.write("| 任务 | 指标 | 基础模型 | LoRA模型 | 提升 |\n")
        f.write("|------|------|---------|---------|------|\n")
        f.write(f"| 分类 | 准确率 | {base_cls:.1f}% | {lora_cls:.1f}% | **{lora_cls - base_cls:+.1f}%** |\n")
        f.write(f"| 检测 | IoU | {base_iou:.1f}% | {lora_iou:.1f}% | **{lora_iou - base_iou:+.1f}%** |\n")
        f.write(f"| 行业识别 | 准确率 | {base_ind:.1f}% | {lora_ind:.1f}% | **{lora_ind - base_ind:+.1f}%** |\n")
    
    print(f"\n✅ 报告已生成:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {os.path.join(output_dir, 'results.json')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='综合评估 LoRA 模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='LoRA checkpoint 路径')
    parser.add_argument('--label', type=str, default='', help='实验标签，如 exp1_10k_r64')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument('--train_samples', type=int, default=0, help='训练样本数量')
    parser.add_argument('--prompt', type=str, default='v1', choices=['v1', 'v2', 'v3'],
                        help='提示词版本: v1=原始, v2=详细, v3=简洁')
    parser.add_argument('--test_json', type=str, default='test_subset.json', help='测试数据集')
    parser.add_argument('--num_samples', type=int, default=30, help='测试样本数')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"实验配置")
    print(f"{'='*60}")
    print(f"标签: {args.label}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"训练样本: {args.train_samples:,}")
    print(f"提示词版本: {args.prompt}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")
    
    stats, results = comprehensive_evaluation(
        test_json=args.test_json,
        base_model_path="Qwen/Qwen3-VL-2B-Instruct",
        lora_checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        experiment_label=args.label,
        lora_rank=args.lora_rank,
        train_samples=args.train_samples,
        prompt_version=args.prompt
    )
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)
    print(f"分类准确率: {stats['classification_accuracy']['base']:.1%} → {stats['classification_accuracy']['lora']:.1%}")
    print(f"检测 IoU: {stats['detection_iou']['base']:.1%} → {stats['detection_iou']['lora']:.1%}")
    print(f"行业识别: {stats['industry_accuracy']['base']:.1%} → {stats['industry_accuracy']['lora']:.1%}")
