#!/usr/bin/env python3
"""
自动评估脚本 - Python版本
扫描log文件,找到训练输出路径和最新checkpoint,自动运行评估
"""

import os
import re
import subprocess
import glob
from pathlib import Path
from typing import Optional, Tuple

# 配置
LOG_DIR = "/home/jiahuawang/test/classVLM/train_part/log"
BASE_DIR = "/home/jiahuawang/test/classVLM"
EVAL_SCRIPT = "comprehensive_eval.py"
GPU_ID = 0  # 使用的GPU ID

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

def extract_output_path(log_file: str) -> Optional[str]:
    """从log文件中提取输出路径"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 匹配 output/qwen3-vl-2b-logo-lora_日期_时间
            match = re.search(r'output/qwen3-vl-2b-logo-lora[^\s/]+', content)
            if match:
                return match.group(0)
    except Exception as e:
        print(f"  {Colors.RED}✗ 读取log文件失败: {e}{Colors.NC}")
    return None

def find_latest_checkpoint(output_path: str) -> Optional[str]:
    """找到最新的checkpoint"""
    checkpoints = glob.glob(os.path.join(output_path, "checkpoint-*"))
    if not checkpoints:
        return None
    
    # 按checkpoint编号排序
    def get_checkpoint_num(path):
        match = re.search(r'checkpoint-(\d+)', path)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_checkpoint_num)
    return checkpoints[-1]

def extract_training_params(log_file: str) -> Tuple[int, int]:
    """从log文件中提取训练参数"""
    lora_rank = 64  # 默认值
    train_samples = 30000  # 默认值
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # 提取lora_rank
            match = re.search(r'lora_rank["\s:=]+(\d+)', content)
            if match:
                lora_rank = int(match.group(1))
            
            # 提取训练样本数
            match = re.search(r'(?:dataset_size|train.*samples?)["\s:=]+(\d+)', content, re.IGNORECASE)
            if match:
                train_samples = int(match.group(1))
            else:
                # 从文件名推断
                log_name = os.path.basename(log_file)
                match = re.search(r'(\d+)k', log_name)
                if match:
                    num = int(match.group(1))
                    train_samples = num * 1000
    except Exception as e:
        print(f"  {Colors.YELLOW}⚠ 提取参数失败,使用默认值: {e}{Colors.NC}")
    
    return lora_rank, train_samples

def run_evaluation(checkpoint: str, label: str, lora_rank: int, 
                   train_samples: int, output_dir: str) -> bool:
    """运行评估"""
    cmd = [
        "python", EVAL_SCRIPT,
        "--checkpoint", checkpoint,
        "--label", label,
        "--lora_rank", str(lora_rank),
        "--train_samples", str(train_samples),
        "--prompt", "v1",
        "--num_samples", "300",
        "--output_dir", output_dir
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    
    print(f"\n{Colors.GREEN}执行评估命令:{Colors.NC}")
    print(f"CUDA_VISIBLE_DEVICES={GPU_ID} " + " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            env=env,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}✗ 评估失败: {e}{Colors.NC}")
        return False

def main():
    print("=" * 50)
    print("自动评估脚本 (Python版)")
    print("=" * 50)
    print()
    
    # 检查log目录
    if not os.path.isdir(LOG_DIR):
        print(f"{Colors.RED}错误: Log目录不存在: {LOG_DIR}{Colors.NC}")
        return
    
    # 获取所有log文件
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    
    if not log_files:
        print(f"{Colors.YELLOW}未找到log文件{Colors.NC}")
        return
    
    print(f"找到 {len(log_files)} 个log文件\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for log_file in sorted(log_files):
        log_name = os.path.basename(log_file)
        log_name_no_ext = os.path.splitext(log_name)[0]
        
        print(f"{Colors.YELLOW}处理: {log_name}{Colors.NC}")
        
        # 提取输出路径
        output_path = extract_output_path(log_file)
        if not output_path:
            print(f"  {Colors.RED}✗ 未找到输出路径,跳过{Colors.NC}\n")
            skip_count += 1
            continue
        
        full_output_path = os.path.join(BASE_DIR, output_path)
        
        # 检查路径是否存在
        if not os.path.isdir(full_output_path):
            print(f"  {Colors.RED}✗ 输出路径不存在: {full_output_path}{Colors.NC}\n")
            skip_count += 1
            continue
        
        # 找到最新checkpoint
        latest_checkpoint = find_latest_checkpoint(full_output_path)
        if not latest_checkpoint:
            print(f"  {Colors.RED}✗ 未找到checkpoint{Colors.NC}\n")
            skip_count += 1
            continue
        
        checkpoint_name = os.path.basename(latest_checkpoint)
        print(f"  {Colors.GREEN}✓ 找到输出路径: {output_path}{Colors.NC}")
        print(f"  {Colors.GREEN}✓ 最新checkpoint: {checkpoint_name}{Colors.NC}")
        
        # 提取label (去掉log_前缀)
        label = log_name_no_ext
        if label.startswith('log_'):
            label = label[4:]
        
        # 提取训练参数
        lora_rank, train_samples = extract_training_params(log_file)
        
        output_dir = os.path.join("evaluation_results", label)
        full_output_dir = os.path.join(BASE_DIR, output_dir)
        
        print(f"  Label: {label}")
        print(f"  LoRA rank: {lora_rank}")
        print(f"  Train samples: {train_samples}")
        print(f"  Output dir: {output_dir}")
        
        # 检查评估结果是否已存在
        if os.path.isdir(full_output_dir) and os.path.isfile(os.path.join(full_output_dir, "results.json")):
            print(f"  {Colors.YELLOW}⚠ 评估结果已存在,跳过{Colors.NC}\n")
            skip_count += 1
            continue
        
        # 运行评估
        if run_evaluation(latest_checkpoint, label, lora_rank, train_samples, output_dir):
            print(f"  {Colors.GREEN}✓ 评估完成!{Colors.NC}\n")
            success_count += 1
        else:
            print(f"  {Colors.RED}✗ 评估失败!{Colors.NC}\n")
            fail_count += 1
        
        print("=" * 50)
        print()
    
    # 总结
    print(f"\n{Colors.BLUE}评估总结:{Colors.NC}")
    print(f"  成功: {Colors.GREEN}{success_count}{Colors.NC}")
    print(f"  跳过: {Colors.YELLOW}{skip_count}{Colors.NC}")
    print(f"  失败: {Colors.RED}{fail_count}{Colors.NC}")
    print(f"  总计: {len(log_files)}")

if __name__ == "__main__":
    main()
