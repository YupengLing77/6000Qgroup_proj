#!/usr/bin/env python3
"""
Comprehensive Visualization of Evaluation Results (English Version)
Generates multiple comparison charts with English labels
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create output directory
output_dir = Path("visualization")
output_dir.mkdir(exist_ok=True)

# Read data
df = pd.read_csv("eval_part/evaluation_summary.csv")

print(f"Loaded {len(df)} experiments")
print("\nData Overview:")
print(df.to_string())

# Map experiment labels to readable names
label_map = {
    '100k': '100k samples',
    '10k': '10k samples',
    '10k_all': '10k (All Layers)',
    '10k_batch4': '10k (Batch 4)',
    '10k_mlp_llm': '10k (MLP+LLM)',
    '10k_vision_llm': '10k (Vision+LLM)',
    '3k': '3k samples',
    '500hq': '500HQ samples',
    '500hq_all': '500HQ (All Layers)'
}


def plot_dataset_size_comparison():
    """1. Dataset Size Comparison"""
    print("\nGenerating Chart 1: Dataset Size Comparison...")
    
    # Filter experiments by dataset size (use 500hq_v2 as representative for 500 samples)
    size_data = df[df['Experiment'].isin(['500hq_v2', '3k', '10k', '100k'])].copy()
    size_data = size_data.sort_values('Train Samples')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Dataset Size Comparison', fontsize=16, fontweight='bold')
    
    # Subplot 1: Three tasks accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(len(size_data))
    width = 0.25
    
    ax1.bar(x - width, size_data['Classification LoRA (%)'], width, label='Classification', alpha=0.8)
    ax1.bar(x, size_data['Detection LoRA (%)'], width, label='Logo Detection', alpha=0.8)
    ax1.bar(x + width, size_data['Industry LoRA (%)'], width, label='Industry Recognition', alpha=0.8)
    
    ax1.set_xlabel('Training Samples', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('LoRA Model Performance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map.get(s, s) for s in size_data['Experiment']])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Performance improvement
    ax2 = axes[0, 1]
    ax2.bar(x - width, size_data['Classification Improvement (%)'], 
            width, label='Classification', alpha=0.8)
    ax2.bar(x, size_data['Detection Improvement (%)'], 
            width, label='Logo Detection', alpha=0.8)
    ax2.bar(x + width, size_data['Industry Improvement (%)'], 
            width, label='Industry Recognition', alpha=0.8)
    
    ax2.set_xlabel('Training Samples', fontsize=11)
    ax2.set_ylabel('Improvement (%)', fontsize=11)
    ax2.set_title('Performance Improvement over Baseline', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([label_map.get(s, s) for s in size_data['Experiment']])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Training time
    ax3 = axes[1, 0]
    colors = sns.color_palette("husl", len(size_data))
    train_times = size_data['Train Time (s)'].replace('N/A', np.nan).astype(float) / 3600
    bars = ax3.bar(x, train_times, color=colors, alpha=0.8)
    
    ax3.set_xlabel('Training Samples', fontsize=11)
    ax3.set_ylabel('Training Time (hours)', fontsize=11)
    ax3.set_title('Training Time Cost', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([label_map.get(s, s) for s in size_data['Experiment']])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}h',
                    ha='center', va='bottom', fontsize=10)
    
    # Subplot 4: Efficiency (performance/time)
    ax4 = axes[1, 1]
    efficiency = []
    for _, row in size_data.iterrows():
        avg_improvement = (
            row['Classification Improvement (%)'] + 
            row['Detection Improvement (%)'] + 
            row['Industry Improvement (%)']
        ) / 3
        time_val = row['Train Time (s)']
        if time_val != 'N/A' and not pd.isna(time_val):
            time_hours = float(time_val) / 3600
            efficiency.append(avg_improvement / time_hours if time_hours > 0 else 0)
        else:
            efficiency.append(0)
    
    bars = ax4.bar(x, efficiency, color=colors, alpha=0.8)
    ax4.set_xlabel('Training Samples', fontsize=11)
    ax4.set_ylabel('Efficiency (Improvement %/hour)', fontsize=11)
    ax4.set_title('Training Efficiency', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([label_map.get(s, s) for s in size_data['Experiment']])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_dataset_size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '1_dataset_size_comparison.png'}")
    plt.close()


def plot_batch_size_comparison():
    """2. Batch Size Comparison"""
    print("\nGenerating Chart 2: Batch Size Comparison...")
    
    # Filter batch comparison experiments (10k samples)
    batch_data = df[df['Experiment'].isin(['10k_all', '10k_batch4', '10k'])].copy()
    
    # Add batch size info
    batch_info = {'10k_all': 2, '10k_batch4': 4, '10k': 8}
    batch_data['BatchSize'] = batch_data['Experiment'].map(batch_info)
    batch_data = batch_data.sort_values('BatchSize')
    
    if len(batch_data) < 2:
        print("⚠ Insufficient data, skipping batch size comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Batch Size Comparison (10k samples)', fontsize=16, fontweight='bold')
    
    tasks = ['Classification', 'Detection', 'Industry']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (task, color) in enumerate(zip(tasks, colors)):
        ax = axes[idx]
        
        # Prepare data
        x = np.arange(len(batch_data))
        baseline = batch_data[f'{task} Baseline (%)'].values
        lora = batch_data[f'{task} LoRA (%)'].values
        improvement = batch_data[f'{task} Improvement (%)'].values
        
        # Plot grouped bars
        width = 0.35
        ax.bar(x - width/2, baseline, width, label='Baseline', alpha=0.7, color='gray')
        ax.bar(x + width/2, lora, width, label='LoRA', alpha=0.8, color=color)
        
        # Add improvement labels
        for i, imp in enumerate(improvement):
            ax.text(i + width/2, lora[i] + 2, f'+{imp:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)
        
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{task} Performance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Batch {b}' for b in batch_data['BatchSize']], fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, max(lora) * 1.15])
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_batch_size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '2_batch_size_comparison.png'}")
    plt.close()


def plot_lora_method_comparison():
    """3. LoRA Method Comparison"""
    print("\nGenerating Chart 3: LoRA Method Comparison...")
    
    # Filter LoRA method experiments (10k samples)
    lora_methods = ['10k', '10k_all', '10k_mlp_llm', '10k_vision_llm']
    method_data = df[df['Experiment'].isin(lora_methods)].copy()
    
    if len(method_data) < 2:
        print("⚠ Insufficient data, skipping LoRA method comparison")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('LoRA Fine-tuning Strategy Comparison (10k samples)', fontsize=16, fontweight='bold')
    
    # Method labels
    method_labels = {
        '10k': 'LLM Only',
        '10k_all': 'All Layers',
        '10k_mlp_llm': 'MLP+LLM',
        '10k_vision_llm': 'Vision+LLM'
    }
    method_data['Method'] = method_data['Experiment'].map(method_labels)
    
    # Subplot 1: Radar chart
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    
    categories = ['Classification', 'Detection', 'Industry']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    colors_radar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, (_, row) in enumerate(method_data.iterrows()):
        values = [
            row['Classification LoRA (%)'],
            row['Detection LoRA (%)'],
            row['Industry LoRA (%)']
        ]
        values += values[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=row['Method'], 
                color=colors_radar[idx % len(colors_radar)], markersize=8)
        ax1.fill(angles, values, alpha=0.15, color=colors_radar[idx % len(colors_radar)])
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_title('Multi-task Performance Radar', fontsize=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)
    
    # Subplot 2: Improvement comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(method_data))
    width = 0.25
    
    improvements = {
        'Classification': method_data['Classification Improvement (%)'],
        'Detection': method_data['Detection Improvement (%)'],
        'Industry': method_data['Industry Improvement (%)']
    }
    
    for i, (task, imp) in enumerate(improvements.items()):
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, imp, width, label=task, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('LoRA Method', fontsize=11)
    ax2.set_ylabel('Improvement (%)', fontsize=11)
    ax2.set_title('Task Performance Improvement', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_data['Method'], fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Training time
    ax3 = fig.add_subplot(gs[0, 2])
    time_values = method_data['Train Time (s)'].replace('N/A', np.nan).astype(float) / 60
    bars = ax3.barh(method_data['Method'], time_values, alpha=0.8, color=colors_radar[:len(method_data)])
    
    ax3.set_xlabel('Training Time (minutes)', fontsize=11)
    ax3.set_title('Training Time Cost', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar in bars:
        width_val = bar.get_width()
        if not np.isnan(width_val):
            ax3.text(width_val, bar.get_y() + bar.get_height()/2.,
                    f'{width_val:.0f}min',
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Subplots 4-6: Detailed task comparison
    tasks = ['Classification', 'Detection', 'Industry']
    for idx, task in enumerate(tasks):
        ax = fig.add_subplot(gs[1, idx])
        
        baseline = method_data[f'{task} Baseline (%)'].values
        lora = method_data[f'{task} LoRA (%)'].values
        
        x = np.arange(len(method_data))
        width = 0.35
        
        ax.bar(x - width/2, baseline, width, label='Baseline', alpha=0.7, color='gray')
        ax.bar(x + width/2, lora, width, label='LoRA', alpha=0.8, color=colors_radar[idx])
        
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{task} Detailed Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_data['Method'], fontsize=9, rotation=15)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(output_dir / '3_lora_method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '3_lora_method_comparison.png'}")
    plt.close()


def plot_comprehensive_heatmap():
    """4. Comprehensive Performance Heatmap"""
    print("\nGenerating Chart 4: Comprehensive Heatmap...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Performance Heatmap', fontsize=16, fontweight='bold')
    
    # Prepare data
    tasks = ['Classification', 'Detection', 'Industry']
    
    # Heatmap 1: LoRA accuracy
    lora_performance = []
    for task in tasks:
        lora_performance.append(df[f'{task} LoRA (%)'].values)
    lora_performance = np.array(lora_performance)
    
    im1 = ax1.imshow(lora_performance, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(np.arange(len(df)))
    ax1.set_yticks(np.arange(len(tasks)))
    ax1.set_xticklabels([label_map.get(e, e) for e in df['Experiment']], rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(tasks)
    ax1.set_title('LoRA Model Accuracy (%)', fontsize=12, fontweight='bold')
    
    # Add value labels
    for i in range(len(tasks)):
        for j in range(len(df)):
            text = ax1.text(j, i, f'{lora_performance[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=ax1, label='Accuracy (%)')
    
    # Heatmap 2: Improvement
    improvement = []
    for task in tasks:
        improvement.append(df[f'{task} Improvement (%)'].values)
    improvement = np.array(improvement)
    
    im2 = ax2.imshow(improvement, cmap='YlOrRd', aspect='auto', vmin=0, vmax=80)
    ax2.set_xticks(np.arange(len(df)))
    ax2.set_yticks(np.arange(len(tasks)))
    ax2.set_xticklabels([label_map.get(e, e) for e in df['Experiment']], rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(tasks)
    ax2.set_title('Performance Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    
    # Add value labels
    for i in range(len(tasks)):
        for j in range(len(df)):
            text = ax2.text(j, i, f'+{improvement[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=ax2, label='Improvement (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '4_comprehensive_heatmap.png'}")
    plt.close()


def plot_training_cost_analysis():
    """5. Training Cost and Benefit Analysis"""
    print("\nGenerating Chart 5: Training Cost Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Cost and Benefit Analysis', fontsize=16, fontweight='bold')
    
    # Calculate average improvement
    df['Avg Improvement'] = df[['Classification Improvement (%)', 'Detection Improvement (%)', 'Industry Improvement (%)']].mean(axis=1)
    
    # Filter valid training time data
    df_valid = df[df['Train Time (s)'] != 'N/A'].copy()
    df_valid['Train Time (s)'] = df_valid['Train Time (s)'].astype(float)
    
    # Subplot 1: Training time vs average improvement (scatter)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df_valid['Train Time (s)'] / 3600, df_valid['Avg Improvement'], 
                         s=df_valid['Train Samples'] / 100, alpha=0.6, c=range(len(df_valid)), 
                         cmap='viridis', edgecolors='black', linewidth=1)
    
    # Add labels
    for idx, row in df_valid.iterrows():
        ax1.annotate(label_map.get(row['Experiment'], row['Experiment']), 
                    (row['Train Time (s)'] / 3600, row['Avg Improvement']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Training Time (hours)', fontsize=11)
    ax1.set_ylabel('Average Improvement (%)', fontsize=11)
    ax1.set_title('Training Time vs Performance Improvement', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(df_valid) > 1:
        z = np.polyfit(df_valid['Train Time (s)'] / 3600, df_valid['Avg Improvement'], 1)
        p = np.poly1d(z)
        ax1.plot(df_valid['Train Time (s)'] / 3600, p(df_valid['Train Time (s)'] / 3600), 
                "r--", alpha=0.5, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        ax1.legend()
    
    # Subplot 2: Sample size vs improvement
    ax2 = axes[0, 1]
    colors_scatter = plt.cm.Set3(range(len(df)))
    
    ax2.scatter(df['Train Samples'], df['Avg Improvement'], s=200, alpha=0.7, c=colors_scatter, edgecolors='black')
    
    for idx, row in df.iterrows():
        ax2.annotate(label_map.get(row['Experiment'], row['Experiment']), 
                    (row['Train Samples'], row['Avg Improvement']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Training Samples', fontsize=11)
    ax2.set_ylabel('Average Improvement (%)', fontsize=11)
    ax2.set_title('Sample Size vs Performance Improvement', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Cost-effectiveness ranking
    ax3 = axes[1, 0]
    df_valid['Efficiency'] = df_valid['Avg Improvement'] / (df_valid['Train Time (s)'] / 3600)
    df_sorted = df_valid.sort_values('Efficiency', ascending=True)
    
    bars = ax3.barh(range(len(df_sorted)), df_sorted['Efficiency'], alpha=0.8)
    
    # Color mapping
    colors_map = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors_map):
        bar.set_color(color)
    
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels([label_map.get(e, e) for e in df_sorted['Experiment']])
    ax3.set_xlabel('Cost-Effectiveness (Improvement %/hour)', fontsize=11)
    ax3.set_title('Training Cost-Effectiveness Ranking', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax3.text(row['Efficiency'], i, f"  {row['Efficiency']:.2f}",
                va='center', fontsize=9, fontweight='bold')
    
    # Subplot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for _, row in df.iterrows():
        time_str = row['Train Time'] if row['Train Time (s)'] != 'N/A' else 'N/A'
        table_data.append([
            label_map.get(row['Experiment'], row['Experiment']),
            f"{row['Train Samples']:,}",
            time_str,
            f"{row['Avg Improvement']:.1f}%",
            f"{row.get('Efficiency', 0):.2f}" if 'Efficiency' in row and not pd.isna(row.get('Efficiency')) else 'N/A'
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Experiment', 'Samples', 'Time', 'Avg Imp.', 'Efficiency'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.15, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    
    # Header style
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Training Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_training_cost_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '5_training_cost_analysis.png'}")
    plt.close()


def plot_task_specific_analysis():
    """6. Task-Specific Analysis"""
    print("\nGenerating Chart 6: Task-Specific Analysis...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Three Tasks In-Depth Analysis', fontsize=16, fontweight='bold')
    
    tasks = ['Classification', 'Detection', 'Industry']
    colors_task = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (task, color) in enumerate(zip(tasks, colors_task)):
        # Top row: Baseline vs LoRA
        ax_top = axes[0, idx]
        
        x = np.arange(len(df))
        width = 0.35
        
        baseline = df[f'{task} Baseline (%)']
        lora = df[f'{task} LoRA (%)']
        
        bars1 = ax_top.bar(x - width/2, baseline, width, label='Baseline', alpha=0.7, color='gray')
        bars2 = ax_top.bar(x + width/2, lora, width, label='LoRA', alpha=0.8, color=color)
        
        ax_top.set_ylabel('Accuracy (%)', fontsize=10)
        ax_top.set_title(f'{task}: Baseline vs LoRA', fontsize=11, fontweight='bold')
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([label_map.get(e, e) for e in df['Experiment']], rotation=45, ha='right', fontsize=7)
        ax_top.legend(loc='upper left')
        ax_top.grid(True, alpha=0.3, axis='y')
        
        # Bottom row: Improvement magnitude
        ax_bottom = axes[1, idx]
        
        improvement = df[f'{task} Improvement (%)']
        bars = ax_bottom.bar(x, improvement, alpha=0.8, color=color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax_bottom.text(bar.get_x() + bar.get_width()/2., height,
                          f'+{height:.1f}%',
                          ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax_bottom.set_ylabel('Improvement (%)', fontsize=10)
        ax_bottom.set_title(f'{task}: Improvement Magnitude', fontsize=11, fontweight='bold')
        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels([label_map.get(e, e) for e in df['Experiment']], rotation=45, ha='right', fontsize=7)
        ax_bottom.grid(True, alpha=0.3, axis='y')
        
        # Highlight best
        max_idx = improvement.idxmax()
        bars[max_idx].set_edgecolor('gold')
        bars[max_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_task_specific_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '6_task_specific_analysis.png'}")
    plt.close()


def plot_performance_summary():
    """7. Performance Summary Dashboard"""
    print("\nGenerating Chart 7: Performance Summary...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Model Performance Summary Dashboard', fontsize=18, fontweight='bold')
    
    # 1. Best performance ranking
    ax1 = fig.add_subplot(gs[0, :2])
    
    best_scores = []
    for _, row in df.iterrows():
        best_scores.append({
            'label': row['Experiment'],
            'Classification': row['Classification LoRA (%)'],
            'Detection': row['Detection LoRA (%)'],
            'Industry': row['Industry LoRA (%)'],
            'Average': (row['Classification LoRA (%)'] + row['Detection LoRA (%)'] + row['Industry LoRA (%)']) / 3
        })
    
    best_df = pd.DataFrame(best_scores).sort_values('Average', ascending=False)
    
    x = np.arange(len(best_df))
    width = 0.2
    
    ax1.bar(x - 1.5*width, best_df['Classification'], width, label='Classification', alpha=0.8, color='#FF6B6B')
    ax1.bar(x - 0.5*width, best_df['Detection'], width, label='Detection', alpha=0.8, color='#4ECDC4')
    ax1.bar(x + 0.5*width, best_df['Industry'], width, label='Industry', alpha=0.8, color='#45B7D1')
    ax1.bar(x + 1.5*width, best_df['Average'], width, label='Average', alpha=0.8, color='#95E1D3')
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Overall Performance Ranking (Sorted by Average)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([label_map.get(s, s) for s in best_df['label']], rotation=30, ha='right', fontsize=9)
    ax1.legend(loc='upper right', ncol=4)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # 2. Key metrics card
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # Calculate key metrics
    best_classification = df.loc[df['Classification LoRA (%)'].idxmax()]
    best_detection = df.loc[df['Detection LoRA (%)'].idxmax()]
    best_industry = df.loc[df['Industry LoRA (%)'].idxmax()]
    
    df_time_valid = df[df['Train Time (s)'] != 'N/A'].copy()
    if len(df_time_valid) > 0:
        df_time_valid['Train Time (s)'] = df_time_valid['Train Time (s)'].astype(float)
        fastest = df_time_valid.loc[df_time_valid['Train Time (s)'].idxmin()]
        fastest_label = label_map.get(fastest['Experiment'], fastest['Experiment'])
        fastest_time = fastest['Train Time']
    else:
        fastest_label = 'N/A'
        fastest_time = 'N/A'
    
    metrics_text = f"""
    ━━━ Key Metrics ━━━
    
    🏆 Best Classification:
       {label_map.get(best_classification['Experiment'], best_classification['Experiment'])}
       Accuracy: {best_classification['Classification LoRA (%)']}%
    
    🏆 Best Detection:
       {label_map.get(best_detection['Experiment'], best_detection['Experiment'])}
       Accuracy: {best_detection['Detection LoRA (%)']}%
    
    🏆 Best Industry:
       {label_map.get(best_industry['Experiment'], best_industry['Experiment'])}
       Accuracy: {best_industry['Industry LoRA (%)']}%
    
    ⚡ Fastest Training:
       {fastest_label}
       Time: {fastest_time}
    
    📊 Total Experiments: {len(df)}
    """
    
    ax2.text(0.1, 0.5, metrics_text, fontsize=9, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    # 3-5. Task improvement trends
    tasks = ['Classification', 'Detection', 'Industry']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (task, color) in enumerate(zip(tasks, colors)):
        ax = fig.add_subplot(gs[1, idx])
        
        improvement = df[f'{task} Improvement (%)']
        
        # Plot line chart
        ax.plot(range(len(df)), improvement, marker='o', linewidth=2, 
               markersize=8, color=color, alpha=0.8)
        ax.fill_between(range(len(df)), improvement, alpha=0.3, color=color)
        
        ax.set_xlabel('Experiment', fontsize=10)
        ax.set_ylabel('Improvement (%)', fontsize=10)
        ax.set_title(f'{task} Improvement Trend', fontsize=11, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([label_map.get(e, e) for e in df['Experiment']], rotation=45, ha='right', fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # Highlight maximum
        max_idx = improvement.idxmax()
        max_val = improvement.max()
        ax.scatter(max_idx, max_val, s=200, color='red', zorder=5, 
                  edgecolors='darkred', linewidth=2)
        ax.annotate(f'Max: +{max_val:.1f}%', 
                   xy=(max_idx, max_val),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 6. Sample size vs performance
    ax6 = fig.add_subplot(gs[2, :])
    
    # Prepare data
    samples = df['Train Samples'].values
    class_perf = df['Classification LoRA (%)'].values
    detect_perf = df['Detection LoRA (%)'].values
    industry_perf = df['Industry LoRA (%)'].values
    
    # Plot three curves
    ax6.scatter(samples, class_perf, s=150, alpha=0.7, color='#FF6B6B', 
               label='Classification', edgecolors='black', linewidth=1)
    ax6.plot(samples, class_perf, '--', alpha=0.5, color='#FF6B6B')
    
    ax6.scatter(samples, detect_perf, s=150, alpha=0.7, color='#4ECDC4', 
               label='Logo Detection', edgecolors='black', linewidth=1)
    ax6.plot(samples, detect_perf, '--', alpha=0.5, color='#4ECDC4')
    
    ax6.scatter(samples, industry_perf, s=150, alpha=0.7, color='#45B7D1', 
               label='Industry Recognition', edgecolors='black', linewidth=1)
    ax6.plot(samples, industry_perf, '--', alpha=0.5, color='#45B7D1')
    
    # Add labels
    for i, row in df.iterrows():
        ax6.annotate(label_map.get(row['Experiment'], row['Experiment']), 
                    (row['Train Samples'], row['Classification LoRA (%)']),
                    xytext=(0, 10), textcoords='offset points', 
                    fontsize=7, ha='center', alpha=0.7)
    
    ax6.set_xlabel('Training Samples', fontsize=11)
    ax6.set_ylabel('Accuracy (%)', fontsize=11)
    ax6.set_title('Sample Size vs Performance', fontsize=13, fontweight='bold')
    ax6.set_xscale('log')
    ax6.legend(loc='lower right', fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / '7_performance_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '7_performance_summary.png'}")
    plt.close()


def generate_all_visualizations():
    """Generate all visualization charts"""
    print("=" * 60)
    print("Starting visualization generation...")
    print("=" * 60)
    
    # Generate all charts
    plot_dataset_size_comparison()           # 1. Dataset size comparison
    plot_batch_size_comparison()             # 2. Batch size comparison
    plot_lora_method_comparison()            # 3. LoRA method comparison
    plot_comprehensive_heatmap()             # 4. Comprehensive heatmap
    plot_training_cost_analysis()            # 5. Cost-benefit analysis
    plot_task_specific_analysis()            # 6. Task-specific analysis
    plot_performance_summary()               # 7. Performance summary
    
    print("\n" + "=" * 60)
    print(f"✓ All charts generated! Saved in: {output_dir.absolute()}")
    print("=" * 60)
    
    # Generate README
    from datetime import datetime
    readme_content = f"""# Model Evaluation Visualization Results

## Generated Charts

### 1. Dataset Size Comparison (1_dataset_size_comparison.png)
- Compares 3k, 10k, 100k, 500HQ dataset scales
- Includes performance, improvement, training time, and efficiency metrics

### 2. Batch Size Comparison (2_batch_size_comparison.png)
- Compares batch 2 vs batch 4 vs batch 8
- Shows detailed performance differences across three tasks

### 3. LoRA Method Comparison (3_lora_method_comparison.png)
- Compares LLM Only, All Layers, MLP+LLM, Vision+LLM strategies
- Multi-dimensional analysis including radar chart, improvements, and training time

### 4. Comprehensive Performance Heatmap (4_comprehensive_heatmap.png)
- Performance matrix for all experiments
- Left: LoRA accuracy heatmap
- Right: Performance improvement heatmap

### 5. Training Cost and Benefit Analysis (5_training_cost_analysis.png)
- Training time vs performance improvement scatter plot
- Sample size vs performance relationship
- Cost-effectiveness ranking
- Training statistics summary table

### 6. Task-Specific Analysis (6_task_specific_analysis.png)
- Independent analysis for three tasks (Classification, Detection, Industry)
- Baseline comparison and improvement magnitude for each task

### 7. Performance Summary Dashboard (7_performance_summary.png)
- Overall ranking
- Key metrics card
- Task improvement trends
- Sample size vs performance relationship

## Data Source
- Evaluation Results CSV: `eval_part/evaluation_summary.csv`
- Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage
```bash
# Regenerate all charts
python eval_part/visualize_results_en.py
```

## Experiments Overview
- Total: {len(df)} experiments
- Sample sizes: 500, 3k, 10k, 100k
- LoRA methods: LLM Only, All Layers, MLP+LLM, Vision+LLM
- Batch sizes: 2, 4, 8
"""
    
    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n📝 Generated README: {output_dir / 'README.md'}")


if __name__ == "__main__":
    generate_all_visualizations()
