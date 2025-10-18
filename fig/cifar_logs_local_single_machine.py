import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib as mpl
import os, sys
import math
import random

from datetime import datetime

# Apply same style settings as fig3.py
mpl.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'sans-serif'

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['xtick.color'] = 'Grey'
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['ytick.color'] = 'Grey'
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['hatch.color'] = 'Red'
mpl.rcParams['hatch.linewidth'] = 1.5

mpl.rcParams['grid.color'] = 'gainsboro'
mpl.rc('axes', edgecolor='darkgrey')
plt.rc('pdf', fonttype=42)

# Colors for different workers
colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']

def parse_log_file(log_file_path):
    """
    Parse a single worker log file.
    
    Returns:
        tuple: (global_indices, worker_indices) 
               where global_indices = file_id * 10000 + sample_index
               and worker_indices = line numbers (0-based)
    """
    global_indices = []
    worker_indices = []
    
    try:
        with open(log_file_path, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if '-' in line:
                    parts = line.split('-')
                    if len(parts) == 2:
                        file_id = int(parts[0])
                        sample_index = int(parts[1])
                        
                        # Calculate global index: file_id * 10000 + sample_index
                        global_index = file_id * 10000 + sample_index
                        
                        global_indices.append(global_index)
                        worker_indices.append(line_num)  # Line number as worker internal index
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    
    return global_indices, worker_indices

def plot_real_logs_distribution(logs_dir, title="CorgiPile Real Logs Distribution", output_file=None):
    """
    Plot the distribution of real log data showing the effect of dual-layer shuffle.
    
    Args:
        logs_dir (str): Directory containing worker log files
        title (str): Plot title
        output_file (str): Output file path (optional)
    """
    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(111)
    
    # Apply same layout adjustments as fig3.py
    plt.subplots_adjust(left=0.198, bottom=0.17, right=0.946, top=0.967,
                       wspace=0.205, hspace=0.2)
    
    # Set labels
    ax.set_ylabel("Global tuple id", color='black')
    ax.set_xlabel("Worker internal processing order")
    
    # Process each worker's log file
    worker_files = [
        "worker_0_samples.txt",
        "worker_1_samples.txt", 
        "worker_2_samples.txt"
    ]
    
    max_worker_index = 0
    max_global_index = 0
    
    for worker_id, worker_file in enumerate(worker_files):
        log_file_path = os.path.join(logs_dir, worker_file)
        
        if os.path.exists(log_file_path):
            print(f"Processing {worker_file}...")
            global_indices, worker_indices = parse_log_file(log_file_path)
            
            if global_indices:
                # Plot with different color for each worker
                ax.scatter(worker_indices, global_indices, 
                          s=1, 
                          color=colors[worker_id], 
                          label=f'Worker {worker_id}',
                          alpha=0.6)
                
                max_worker_index = max(max_worker_index, max(worker_indices))
                max_global_index = max(max_global_index, max(global_indices))
                
                print(f"Worker {worker_id}: {len(global_indices)} samples")
                print(f"  Global index range: [{min(global_indices)}, {max(global_indices)}]")
                print(f"  Worker index range: [{min(worker_indices)}, {max(worker_indices)}]")
        else:
            print(f"Warning: {log_file_path} not found")
    
    # Set axis limits (adjust based on data)
    ax.set_xlim(xmin=0, xmax=max_worker_index + 1000)
    ax.set_ylim(ymin=0, ymax=max_global_index + 5000)
    
    # Add grid and legend
    ax.grid(True)
    ax.legend(loc='upper right')
    
    # Save the plot
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        # Default save location if no output file specified
        default_output = os.path.join(os.path.dirname(__file__), "corgipile_logs_distribution.png")
        fig.savefig(default_output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_output}")

    plt.close()  # Close figure to free memory

def analyze_shuffle_quality(logs_dir):
    """
    Analyze the quality of shuffle by checking the distribution patterns.
    """
    print("\n=== Shuffle Quality Analysis ===")
    
    worker_files = [
        "worker_0_samples.txt",
        "worker_1_samples.txt", 
        "worker_2_samples.txt"
    ]
    
    total_samples = 0
    file_distribution = {}
    
    for worker_id, worker_file in enumerate(worker_files):
        log_file_path = os.path.join(logs_dir, worker_file)
        
        if os.path.exists(log_file_path):
            global_indices, worker_indices = parse_log_file(log_file_path)
            total_samples += len(global_indices)
            
            # Count samples per file for this worker
            worker_file_dist = {}
            for global_idx in global_indices:
                file_id = global_idx // 10000
                worker_file_dist[file_id] = worker_file_dist.get(file_id, 0) + 1
            
            print(f"Worker {worker_id} file distribution: {worker_file_dist}")
            
            # Aggregate global file distribution
            for file_id, count in worker_file_dist.items():
                file_distribution[file_id] = file_distribution.get(file_id, 0) + count
    
    print(f"Total samples processed: {total_samples}")
    print(f"Global file distribution: {file_distribution}")
    
    # Check if shuffle is working (samples should not be in order)
    for worker_id, worker_file in enumerate(worker_files):
        log_file_path = os.path.join(logs_dir, worker_file)
        if os.path.exists(log_file_path):
            global_indices, worker_indices = parse_log_file(log_file_path)
            
            # Check first few samples to see if they're shuffled
            first_10_global = global_indices[:10] if len(global_indices) >= 10 else global_indices
            print(f"Worker {worker_id} first 10 global indices: {first_10_global}")
            
            # Simple shuffle check: are consecutive samples from the same file in order?
            is_shuffled = False
            if len(global_indices) >= 5:
                # Check if first 5 samples are NOT consecutive
                first_5 = global_indices[:5]
                for i in range(1, len(first_5)):
                    if abs(first_5[i] - first_5[i-1]) != 1:
                        is_shuffled = True
                        break
            
            print(f"Worker {worker_id} appears shuffled: {is_shuffled}")

if __name__ == '__main__':
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration with relative paths
    logs_directory = os.path.join(project_root, "examples", "logs", "local_test_3_workers")
    plot_title = "CorgiPile Dual-Layer Shuffle: 3 Workers Real Distribution"
    output_file_path = os.path.join(script_dir, "cifar_logs_local_single_machine.png")
    
    # Check if logs directory exists
    if not os.path.exists(logs_directory):
        print(f"Error: Logs directory '{logs_directory}' not found!")
        print("Please make sure the path is correct.")
        sys.exit(1)
    
    print("=== CorgiPile Real Logs Visualization ===")
    print(f"Logs directory: {logs_directory}")
    
    # Analyze shuffle quality
    analyze_shuffle_quality(logs_directory)
    
    # Generate the plot
    print(f"\nGenerating plot...")
    plot_real_logs_distribution(logs_directory, plot_title, output_file_path)
