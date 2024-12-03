import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_model_files(results_dir, model_prefix):
    """Find all CSV files for a specific model type in the results directory"""
    pattern = os.path.join(results_dir, f"{model_prefix}*_evaluation_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No {model_prefix} model evaluation file found in {results_dir}")
    return files

def load_teacher_student_data(results_dir):
    """Load teacher and student model data"""
    teacher_df = pd.read_csv(find_model_files(results_dir, "teacher")[0])
    student_df = pd.read_csv(find_model_files(results_dir, "kd")[0])
    return teacher_df, student_df

def load_pruning_data(results_dir):
    """Load pruning model data and sort by pruning level"""
    pr_files = find_model_files(results_dir, "pr_kd")
    data = []
    
    for file in pr_files:
        df = pd.read_csv(file)
        subtype = df['subtype'].iloc[0]
        data.append({
            'level': subtype,
            'model_size_mb': df['model_size_mb'].iloc[0],
            'flops': df['total_flops'].iloc[0] / 1e6,
            'inference_time': float(df['100_inference_time_mean_std'].iloc[0].split('±')[0])
        })
    
    return pd.DataFrame(data)

def load_quantization_data(results_dir):
    """Load quantization model data and sort by bit width"""
    ptq_files = find_model_files(results_dir, "ptq_kd")
    data = []
    
    for file in ptq_files:
        df = pd.read_csv(file)
        subtype = df['subtype'].iloc[0]
        data.append({
            'bits': int(subtype),
            'model_size_mb': df['model_size_mb'].iloc[0],
            'flops': df['total_flops'].iloc[0] / 1e6,
            'inference_time': float(df['100_inference_time_mean_std'].iloc[0].split('±')[0])
        })
    
    return pd.DataFrame(data)

def load_compression_data(results_dir):
    """Load both pruning and quantization data for total comparison"""
    # Load pruning data
    pr_files = find_model_files(results_dir, "pr_kd")
    pruning_data = []
    pruning_order = ['low', 'medium', 'high']
    
    for file in pr_files:
        df = pd.read_csv(file)
        subtype = df['subtype'].iloc[0]
        pruning_data.append({
            'method': 'Pruning',
            'variant': subtype,
            'model_size_mb': df['model_size_mb'].iloc[0],
            'flops': df['total_flops'].iloc[0] / 1e6,
            'inference_time': float(df['100_inference_time_mean_std'].iloc[0].split('±')[0])
        })
    
    pruning_df = pd.DataFrame(pruning_data)
    pruning_df['sort_idx'] = pruning_df['variant'].map({v: i for i, v in enumerate(pruning_order)})
    pruning_df = pruning_df.sort_values('sort_idx').drop('sort_idx', axis=1)
    
    # Load quantization data
    ptq_files = find_model_files(results_dir, "ptq_kd")
    quant_data = []
    
    for file in ptq_files:
        df = pd.read_csv(file)
        bit_width = int(df['subtype'].iloc[0])
        quant_data.append({
            'method': 'Quantization',
            'variant': str(bit_width),
            'model_size_mb': df['model_size_mb'].iloc[0],
            'flops': df['total_flops'].iloc[0] / 1e6,
            'inference_time': float(df['100_inference_time_mean_std'].iloc[0].split('±')[0])
        })
    
    quant_df = pd.DataFrame(quant_data)
    quant_df['sort_idx'] = quant_df['variant'].astype(int)
    quant_df = quant_df.sort_values('sort_idx', ascending=False).drop('sort_idx', axis=1)
    
    return pruning_df, quant_df

def visualize_teacher_student_comparison(teacher_data, student_data, save_path='visualize/teacher_student_comparison.png'):
    """Visualize the comparison between teacher and student models"""
    # Prepare data for plotting
    model_sizes = {
        'Teacher': teacher_data['model_size_mb'].iloc[0],
        'Student': student_data['model_size_mb'].iloc[0]
    }
    
    flops = {
        'Teacher': teacher_data['total_flops'].iloc[0] / 1e6,
        'Student': student_data['total_flops'].iloc[0] / 1e6
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set style
    plt.style.use('seaborn')
    colors = ['#82ca9d', '#8884d8']
    
    # Plot Model Size
    sns.barplot(x=list(model_sizes.keys()), y=list(model_sizes.values()), ax=ax1, palette=colors)
    ax1.set_title('Model Size Comparison')
    ax1.set_ylabel('Size (MB)')
    
    # Plot FLOPs
    sns.barplot(x=list(flops.keys()), y=list(flops.values()), ax=ax2, palette=colors)
    ax2.set_title('FLOPs Comparison')
    ax2.set_ylabel('FLOPs (M)')
    
    # Calculate and display compression rates
    size_reduction = ((model_sizes['Teacher'] - model_sizes['Student']) / model_sizes['Teacher'] * 100)
    flops_reduction = ((flops['Teacher'] - flops['Student']) / flops['Teacher'] * 100)
    
    plt.figtext(0.02, -0.05, f'Compression Rates:\nModel Size: {size_reduction:.2f}% reduction\nFLOPs: {flops_reduction:.2f}% reduction',
                ha='left', va='center')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_pruning_comparison(pruning_df, student_data, save_path='visualize/pruning_comparison.png'):
    """Create line plots comparing different pruning levels with student model"""
    # Set style
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Create subplot for graphs
    top_gs = gs[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(top_gs[0])
    ax2 = fig.add_subplot(top_gs[1])
    ax3 = fig.add_subplot(top_gs[2])
    
    # Sort levels for consistent x-axis
    level_order = ['low', 'medium', 'high']
    pruning_df['level_num'] = pruning_df['level'].map({level: i for i, level in enumerate(level_order)})
    pruning_df = pruning_df.sort_values('level_num')
    
    # Colors and styles
    student_color = '#6B46C1'
    main_color = '#e74c3c'
    
    student_line_style = {
        'color': student_color,
        'alpha': 0.8,
        'linestyle': '--',
        'linewidth': 2,
        'label': 'Student'
    }
    
    # Plot 1: Model Size
    ax1.plot(level_order, pruning_df['model_size_mb'], marker='o', color=main_color, linewidth=2, label='Pruning')
    student_size = student_data['model_size_mb'].iloc[0]
    ax1.axhline(y=student_size, **student_line_style)
    
    for idx, size in enumerate(pruning_df['model_size_mb']):
        reduction = ((student_size - size) / student_size) * 100
        ax1.annotate(f'{reduction:.1f}%', 
                    xy=(idx, size), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    ax1.set_title('Model Size Comparison')
    ax1.set_ylabel('Size (MB)')
    ax1.legend()
    
    # Plot 2: FLOPs
    ax2.plot(level_order, pruning_df['flops'], marker='o', color=main_color, linewidth=2, label='Pruning')
    student_flops = student_data['total_flops'].iloc[0] / 1e6
    ax2.axhline(y=student_flops, **student_line_style)
    
    for idx, flops in enumerate(pruning_df['flops']):
        reduction = ((student_flops - flops) / student_flops) * 100
        ax2.annotate(f'{reduction:.1f}%', 
                    xy=(idx, flops), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    ax2.set_title('FLOPs Comparison')
    ax2.set_ylabel('FLOPs (M)')
    ax2.legend()
    
    # Plot 3: Inference Time
    ax3.plot(level_order, pruning_df['inference_time'], marker='o', color=main_color, linewidth=2, label='Pruning')
    student_time = float(student_data['100_inference_time_mean_std'].iloc[0].split('±')[0])
    ax3.axhline(y=student_time, **student_line_style)
    
    for idx, time in enumerate(pruning_df['inference_time']):
        change = ((time - student_time) / student_time) * 100
        sign = '+' if change > 0 else ''
        ax3.annotate(f'{sign}{change:.1f}%', 
                    xy=(idx, time), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    ax3.set_title('Inference Time Comparison')
    ax3.set_ylabel('Time (ms)')
    ax3.legend()
    
    # Add pruning ratio table
    pruning_configs = {
        'low': {'conv1': 0.2, 'conv2': 0.3, 'conv3': 0.4},
        'medium': {'conv1': 0.3, 'conv2': 0.4, 'conv3': 0.5},
        'high': {'conv1': 0.4, 'conv2': 0.5, 'conv3': 0.6}
    }
    
    table_data = []
    columns = ['Level'] + [f'Conv{i}' for i in range(1, 4)]
    for level in ['low', 'medium', 'high']:
        ratios = pruning_configs[level]
        row = [level.capitalize()] + [f'{ratios[f"conv{i}"]*100}%' for i in range(1, 4)]
        table_data.append(row)
    
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data,
                          colLabels=columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.3, 0.0, 0.4, 1.0])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_quantization_comparison(quant_df, student_data, save_path='visualize/quantization_comparison.png'):
    """Create line plots comparing different quantization levels with student model"""
    # Set style
    plt.style.use('seaborn')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Sort by bit width in descending order
    quant_df = quant_df.sort_values('bits', ascending=False)
    bit_labels = [f'{int(b)}-bit' for b in quant_df['bits']]
    
    # Colors and styles
    main_color = '#2ecc71'
    student_color = '#6B46C1'
    student_line_style = {
        'color': student_color,
        'alpha': 0.8,
        'linestyle': '--',
        'linewidth': 2,
        'label': 'Student'
    }
    
    # Plot 1: Model Size
    ax1.plot(bit_labels, quant_df['model_size_mb'], marker='o', color=main_color, linewidth=2, label='Quantization')
    student_size = student_data['model_size_mb'].iloc[0]
    ax1.axhline(y=student_size, **student_line_style)
    
    for idx, size in enumerate(quant_df['model_size_mb']):
        change = ((size - student_size) / student_size) * 100
        sign = '+' if change > 0 else ''
        ax1.annotate(f'{sign}{change:.1f}%', 
                    xy=(idx, size), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    ax1.set_title('Model Size Comparison')
    ax1.set_xlabel('Quantization Bit Width')
    ax1.set_ylabel('Size (MB)')
    ax1.legend()
    
    # Plot 2: FLOPs
    ax2.plot(bit_labels, quant_df['flops'], marker='o', color=main_color, linewidth=2, label='Quantization')
    student_flops = student_data['total_flops'].iloc[0] / 1e6
    ax2.axhline(y=student_flops, **student_line_style)
    
    for idx, flops in enumerate(quant_df['flops']):
        change = ((flops - student_flops) / student_flops) * 100
        sign = '+' if change > 0 else ''
        ax2.annotate(f'{sign}{change:.1f}%', 
                    xy=(idx, flops), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    ax2.set_title('FLOPs Comparison')
    ax2.set_xlabel('Quantization Bit Width')
    ax2.set_ylabel('FLOPs (M)')
    ax2.legend()
    
    # Plot 3: Inference Time
    ax3.plot(bit_labels, quant_df['inference_time'], marker='o', color=main_color, linewidth=2, label='Quantization')
    student_time = float(student_data['100_inference_time_mean_std'].iloc[0].split('±')[0])
    ax3.axhline(y=student_time, **student_line_style)
    
    for idx, time in enumerate(quant_df['inference_time']):
        change = ((time - student_time) / student_time) * 100
        sign = '+' if change > 0 else ''
        ax3.annotate(f'{sign}{change:.1f}%', 
                    xy=(idx, time), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom')
    
    ax3.set_title('Inference Time Comparison')
    ax3.set_xlabel('Quantization Bit Width')
    ax3.set_ylabel('Time (ms)')
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_total_comparison(pruning_df, quant_df, student_data, save_path='visualize/total_comparison.png'):
    """Create line plots comparing different compression methods with separate x-axes"""
    plt.style.use('seaborn')
    
    # Create figure with space for table
    fig = plt.figure(figsize=(20, 8))
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    
    # Create subplot for graphs (top row)
    top_gs = gs[0].subgridspec(1, 3)
    ax1 = fig.add_subplot(top_gs[0])
    ax2 = fig.add_subplot(top_gs[1])
    ax3 = fig.add_subplot(top_gs[2])
    
    # Colors and styles
    quant_color = '#2ecc71'    # Green for quantization
    pruning_color = '#e74c3c'  # Red for pruning
    student_color = '#6B46C1'  # Purple for student
    
    student_line_style = {
        'color': student_color,
        'alpha': 0.8,
        'linestyle': '--',
        'linewidth': 2,
        'label': 'Student'
    }
    
    def plot_with_dual_x(ax, pruning_df, quant_df, metric, ylabel, student_val):
        # Create twin axis
        ax2 = ax.twiny()
        
        # Plot pruning data on bottom x-axis
        pruning_line = ax.plot(range(len(pruning_df)), pruning_df[metric], 
                             marker='o', color=pruning_color, linewidth=2, label='Pruning')
        ax.set_xticks(range(len(pruning_df)))
        ax.set_xticklabels(pruning_df['variant'], rotation=0)
        ax.set_xlabel('Pruning Level')
        
        # Plot quantization data on top x-axis
        quant_line = ax2.plot(range(len(quant_df)), quant_df[metric], 
                            marker='o', color=quant_color, linewidth=2, label='Quantization')
        ax2.set_xticks(range(len(quant_df)))
        ax2.set_xticklabels([f'{bit}-bit' for bit in quant_df['variant']], rotation=0)
        ax2.set_xlabel('Post Training Quantization(PTQ) Bit Width')
        
        # Add student reference line
        ax.axhline(y=student_val, **student_line_style)
        
        # Add percentage labels for pruning
        for idx, val in enumerate(pruning_df[metric]):
            change = ((val - student_val) / student_val) * 100
            sign = '+' if change > 0 else ''
            ax.annotate(f'{sign}{change:.1f}%', 
                       xy=(idx, val),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       va='bottom',
                       color=pruning_color)
        
        # Add percentage labels for quantization
        for idx, val in enumerate(quant_df[metric]):
            change = ((val - student_val) / student_val) * 100
            sign = '+' if change > 0 else ''
            ax.annotate(f'{sign}{change:.1f}%', 
                       xy=(idx, val),
                       xytext=(0, -20),
                       textcoords='offset points',
                       ha='center',
                       va='top',
                       color=quant_color)
        
        ax.set_ylabel(ylabel)
        
        # Combine legends
        lines = pruning_line + quant_line + [ax.get_lines()[-1]]
        labels = ['Pruning', 'Quantization', 'Student']
        ax.legend(lines, labels)
    
    # Plot all metrics
    student_size = student_data['model_size_mb'].iloc[0]
    student_flops = student_data['total_flops'].iloc[0] / 1e6
    student_time = float(student_data['100_inference_time_mean_std'].iloc[0].split('±')[0])
    
    plot_with_dual_x(ax1, pruning_df, quant_df, 'model_size_mb', 'Size (MB)', student_size)
    plot_with_dual_x(ax2, pruning_df, quant_df, 'flops', 'FLOPs (M)', student_flops)
    plot_with_dual_x(ax3, pruning_df, quant_df, 'inference_time', 'Time (ms)', student_time)
    
    ax1.set_title('Model Size Comparison')
    ax2.set_title('FLOPs Comparison')
    ax3.set_title('Inference Time Comparison')
    
    # Add pruning ratio table
    pruning_configs = {
        'low': {'conv1': 0.2, 'conv2': 0.3, 'conv3': 0.4},
        'medium': {'conv1': 0.3, 'conv2': 0.4, 'conv3': 0.5},
        'high': {'conv1': 0.4, 'conv2': 0.5, 'conv3': 0.6}
    }
    
    table_data = []
    columns = ['Level'] + [f'Conv{i}' for i in range(1, 4)]
    for level in ['low', 'medium', 'high']:
        ratios = pruning_configs[level]
        row = [level.capitalize()] + [f'{ratios[f"conv{i}"]*100}%' for i in range(1, 4)]
        table_data.append(row)
    
    # Add table subplot
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data,
                          colLabels=columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.3, 0.0, 0.4, 0.8])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add a title for the table
    ax_table.text(0.45, 0.9, 'Pruning Ratios Per Layer', fontsize=12, weight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Main function to generate all visualizations"""
    results_dir = 'results'
    os.makedirs('visualize', exist_ok=True)
    
    # Load all required data
    teacher_df, student_df = load_teacher_student_data(results_dir)
    pruning_df = load_pruning_data(results_dir)
    quant_df = load_quantization_data(results_dir)
    pruning_df_total, quant_df_total = load_compression_data(results_dir)
    
    # Generate all visualizations
    visualize_teacher_student_comparison(teacher_df, student_df)
    plot_pruning_comparison(pruning_df, student_df)
    plot_quantization_comparison(quant_df, student_df)
    plot_total_comparison(pruning_df_total, quant_df_total, student_df)
    
    print("All visualizations have been generated in the 'visualize' directory")

if __name__ == "__main__":
    main()