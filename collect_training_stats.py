"""
收集训练统计信息：训练时长、loss、checkpoint信息等
从 TensorBoard events 和 checkpoint 文件中提取
"""
import os
import json
import glob
from datetime import datetime, timedelta
import csv

def parse_timestamp_from_dirname(dirname):
    """从目录名解析时间戳"""
    # 格式: qwen3-vl-2b-logo-lora_20251120_124437
    try:
        timestamp_str = dirname.split('_')[-2] + '_' + dirname.split('_')[-1]
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except:
        return None

def get_checkpoint_info(output_dir):
    """获取 checkpoint 信息"""
    trainer_state_file = os.path.join(output_dir, 'trainer_state.json')
    
    if not os.path.exists(trainer_state_file):
        return None
    
    with open(trainer_state_file, 'r') as f:
        trainer_state = json.load(f)
    
    # 提取关键信息
    info = {
        'total_steps': trainer_state.get('global_step', 0),
        'num_train_epochs': trainer_state.get('epoch', 0),
        'training_time_hours': trainer_state.get('total_flos', 0) / 1e12,  # 粗略估计
        'best_metric': trainer_state.get('best_metric'),
        'best_model_checkpoint': trainer_state.get('best_model_checkpoint'),
    }
    
    # 从 log_history 获取详细信息
    if 'log_history' in trainer_state:
        log_history = trainer_state['log_history']
        
        # 初始和最终 loss
        train_logs = [log for log in log_history if 'loss' in log]
        if train_logs:
            info['initial_loss'] = train_logs[0].get('loss')
            info['final_loss'] = train_logs[-1].get('loss')
        
        # 学习率
        if train_logs:
            info['final_learning_rate'] = train_logs[-1].get('learning_rate')
        
        # 训练开始和结束时间
        if log_history:
            first_log = log_history[0]
            last_log = log_history[-1]
            
            # 计算实际训练时间（从 step 时间估算）
            if 'epoch' in last_log:
                total_epochs = last_log['epoch']
                total_steps = last_log.get('step', info['total_steps'])
                
                # 如果有时间戳，计算实际时长
                if len(log_history) > 1:
                    # 假设每个 log 间隔代表训练进度
                    info['estimated_time'] = True
    
    # 查找所有 checkpoint
    checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    checkpoint_steps = []
    for cp in checkpoints:
        try:
            step = int(os.path.basename(cp).split('-')[1])
            checkpoint_steps.append(step)
        except:
            pass
    
    info['checkpoints'] = sorted(checkpoint_steps)
    info['num_checkpoints'] = len(checkpoint_steps)
    
    return info

def parse_tensorboard_events(output_dir):
    """解析 TensorBoard events 文件"""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # 查找 events 文件
        runs_dir = os.path.join(output_dir, 'runs')
        if not os.path.exists(runs_dir):
            runs_dir = output_dir
        
        event_files = glob.glob(os.path.join(runs_dir, '**', 'events.out.tfevents.*'), recursive=True)
        
        if not event_files:
            return None
        
        # 加载最新的 event 文件
        latest_event = max(event_files, key=os.path.getctime)
        
        ea = event_accumulator.EventAccumulator(os.path.dirname(latest_event))
        ea.Reload()
        
        # 提取训练时间
        scalars = ea.Tags()['scalars']
        
        info = {}
        
        # 获取 loss
        if 'train/loss' in scalars or 'loss' in scalars:
            loss_tag = 'train/loss' if 'train/loss' in scalars else 'loss'
            loss_events = ea.Scalars(loss_tag)
            
            if loss_events:
                first_event = loss_events[0]
                last_event = loss_events[-1]
                
                # 计算训练时长（秒）
                training_duration = last_event.wall_time - first_event.wall_time
                info['training_duration_seconds'] = training_duration
                info['training_duration_hours'] = training_duration / 3600
                info['training_duration_formatted'] = str(timedelta(seconds=int(training_duration)))
                
                info['initial_loss'] = first_event.value
                info['final_loss'] = last_event.value
                info['total_steps'] = last_event.step
        
        return info
    
    except ImportError:
        print("⚠️  需要安装 tensorboard: pip install tensorboard")
        return None
    except Exception as e:
        print(f"⚠️  解析 TensorBoard 失败: {e}")
        return None

def collect_all_experiments(base_output_dir='output'):
    """收集所有实验的统计信息"""
    experiments = []
    
    # 遍历所有输出目录
    for exp_dir in glob.glob(os.path.join(base_output_dir, 'qwen3-vl-2b-logo-lora_*')):
        if not os.path.isdir(exp_dir):
            continue
        
        exp_name = os.path.basename(exp_dir)
        print(f"\n处理实验: {exp_name}")
        
        # 从目录名提取时间戳
        start_time = parse_timestamp_from_dirname(exp_name)
        
        # 获取 checkpoint 信息
        checkpoint_info = get_checkpoint_info(exp_dir)
        
        # 解析 TensorBoard
        tb_info = parse_tensorboard_events(exp_dir)
        
        if checkpoint_info or tb_info:
            exp_data = {
                'experiment_name': exp_name,
                'output_dir': exp_dir,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'Unknown',
            }
            
            # 合并信息
            if checkpoint_info:
                exp_data.update(checkpoint_info)
            
            if tb_info:
                exp_data.update(tb_info)
            
            experiments.append(exp_data)
            
            print(f"  ✅ 收集成功")
            if 'training_duration_formatted' in exp_data:
                print(f"     训练时长: {exp_data['training_duration_formatted']}")
            if 'total_steps' in exp_data:
                print(f"     总步数: {exp_data['total_steps']}")
            if 'final_loss' in exp_data:
                print(f"     最终Loss: {exp_data['final_loss']:.4f}")
    
    return experiments

def save_to_csv(experiments, output_file='training_stats.csv'):
    """保存为 CSV 表格"""
    if not experiments:
        print("没有找到实验数据")
        return
    
    # 定义字段
    fieldnames = [
        'experiment_name',
        'start_time',
        'training_duration_formatted',
        'training_duration_hours',
        'total_steps',
        'num_train_epochs',
        'initial_loss',
        'final_loss',
        'num_checkpoints',
        'checkpoints',
        'output_dir'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for exp in experiments:
            # 格式化 checkpoints 列表
            if 'checkpoints' in exp:
                exp['checkpoints'] = ', '.join(map(str, exp['checkpoints']))
            
            # 格式化训练时长
            if 'training_duration_hours' in exp:
                exp['training_duration_hours'] = f"{exp['training_duration_hours']:.2f}"
            
            # 格式化 loss
            if 'initial_loss' in exp:
                exp['initial_loss'] = f"{exp['initial_loss']:.4f}"
            if 'final_loss' in exp:
                exp['final_loss'] = f"{exp['final_loss']:.4f}"
            
            writer.writerow(exp)
    
    print(f"\n✅ CSV 已保存: {output_file}")

def save_to_markdown(experiments, output_file='training_stats.md'):
    """保存为 Markdown 表格"""
    if not experiments:
        print("没有找到实验数据")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 训练统计报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验概览\n\n")
        f.write("| 实验名称 | 开始时间 | 训练时长 | 总步数 | Epoch | 初始Loss | 最终Loss | Checkpoints |\n")
        f.write("|----------|---------|---------|--------|-------|----------|----------|-------------|\n")
        
        for exp in experiments:
            name = exp.get('experiment_name', 'Unknown')
            start = exp.get('start_time', 'Unknown')
            duration = exp.get('training_duration_formatted', 'Unknown')
            steps = exp.get('total_steps', 'N/A')
            epochs = exp.get('num_train_epochs', 'N/A')
            
            # 安全处理 loss 值（可能是字符串）
            init_loss = 'N/A'
            if 'initial_loss' in exp:
                try:
                    init_loss = f"{float(exp['initial_loss']):.4f}"
                except (ValueError, TypeError):
                    init_loss = str(exp['initial_loss'])
            
            final_loss = 'N/A'
            if 'final_loss' in exp:
                try:
                    final_loss = f"{float(exp['final_loss']):.4f}"
                except (ValueError, TypeError):
                    final_loss = str(exp['final_loss'])
            
            num_ckpt = exp.get('num_checkpoints', 0)
            
            # 安全处理 epochs 值
            epochs_str = 'N/A'
            if epochs != 'N/A':
                try:
                    epochs_str = f"{float(epochs):.1f}"
                except (ValueError, TypeError):
                    epochs_str = str(epochs)
            
            f.write(f"| {name} | {start} | {duration} | {steps} | {epochs_str} | {init_loss} | {final_loss} | {num_ckpt} |\n")
        
        f.write("\n## 详细信息\n\n")
        for exp in experiments:
            f.write(f"### {exp.get('experiment_name', 'Unknown')}\n\n")
            f.write(f"- **输出目录**: `{exp.get('output_dir', 'N/A')}`\n")
            f.write(f"- **开始时间**: {exp.get('start_time', 'Unknown')}\n")
            f.write(f"- **训练时长**: {exp.get('training_duration_formatted', 'Unknown')}\n")
            f.write(f"- **总步数**: {exp.get('total_steps', 'N/A')}\n")
            
            # 安全处理 epochs
            epochs = exp.get('num_train_epochs', 'N/A')
            if epochs != 'N/A':
                try:
                    epochs_str = f"{float(epochs):.2f}"
                except (ValueError, TypeError):
                    epochs_str = str(epochs)
            else:
                epochs_str = 'N/A'
            f.write(f"- **训练轮数**: {epochs_str}\n")
            
            if 'checkpoints' in exp and isinstance(exp['checkpoints'], list):
                ckpts = ', '.join(map(str, exp['checkpoints']))
                f.write(f"- **Checkpoints**: {ckpts}\n")
            
            f.write("\n")
    
    print(f"✅ Markdown 已保存: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='收集训练统计信息')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录路径')
    parser.add_argument('--csv', type=str, default='training_stats.csv', help='CSV 输出文件')
    parser.add_argument('--markdown', type=str, default='training_stats.md', help='Markdown 输出文件')
    
    args = parser.parse_args()
    
    print("="*60)
    print("开始收集训练统计信息")
    print("="*60)
    
    # 收集所有实验
    experiments = collect_all_experiments(args.output_dir)
    
    if experiments:
        print(f"\n找到 {len(experiments)} 个实验")
        
        # 保存为 CSV
        save_to_csv(experiments, args.csv)
        
        # 保存为 Markdown
        save_to_markdown(experiments, args.markdown)
        
        print("\n" + "="*60)
        print("✅ 统计信息收集完成！")
        print("="*60)
    else:
        print("\n❌ 未找到任何实验数据")
