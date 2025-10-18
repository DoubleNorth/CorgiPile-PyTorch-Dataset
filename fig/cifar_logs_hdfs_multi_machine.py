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

# Colors for different nodes and workers
# Node 0: Blue tones, Node 1: Red tones
colors = {
    'node0': ['tab:blue', 'lightblue', 'steelblue'],
    'node1': ['tab:red', 'lightcoral', 'crimson']
}

# Markers for different workers
markers = ['o', 's', '^']  # circle, square, triangle

def parse_distributed_log_file(log_file_path):
    """
    Parse a distributed worker log file.
    
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

def plot_distributed_logs_distribution(logs_dir, title="CorgiPile Distributed Logs Distribution", output_file=None):
    """
    Plot the distribution of distributed log data showing multi-node processing.
    
    Args:
        logs_dir (str): Directory containing node subdirectories with worker log files
        title (str): Plot title
        output_file (str): Output file path (optional)
    """
    fig = plt.figure(figsize=(4.5, 3.5))  # Slightly larger for distributed data
    ax = fig.add_subplot(111)
    
    # Apply same layout adjustments as fig3.py
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92,
                       wspace=0.205, hspace=0.2)
    
    # Set labels
    ax.set_ylabel("Global tuple id", color='black')
    ax.set_xlabel("Worker internal processing order")
    
    # Process each node's log files
    nodes = ['node0', 'node1']
    worker_files = [
        "worker_0_samples.txt",
        "worker_1_samples.txt", 
        "worker_2_samples.txt"
    ]
    
    max_worker_index = 0
    max_global_index = 0
    total_samples = 0
    
    legend_elements = []
    
    for node_id, node_name in enumerate(nodes):
        node_dir = os.path.join(logs_dir, node_name)
        
        if not os.path.exists(node_dir):
            print(f"Warning: Node directory {node_dir} not found")
            continue
            
        print(f"Processing {node_name}...")
        
        for worker_id, worker_file in enumerate(worker_files):
            log_file_path = os.path.join(node_dir, worker_file)
            
            if os.path.exists(log_file_path):
                print(f"  Processing {worker_file}...")
                global_indices, worker_indices = parse_distributed_log_file(log_file_path)
                
                if global_indices:
                    # Plot with different color/marker for each node/worker combination
                    color = colors[node_name][worker_id]
                    marker = markers[worker_id]
                    
                    # Offset worker_indices for different nodes to avoid overlap
                    offset_worker_indices = [idx + node_id * 15000 for idx in worker_indices]
                    
                    ax.scatter(offset_worker_indices, global_indices, 
                              s=1.5, 
                              color=color, 
                              marker=marker,
                              alpha=0.7,
                              label=f'{node_name.capitalize()} W{worker_id}')
                    
                    max_worker_index = max(max_worker_index, max(offset_worker_indices))
                    max_global_index = max(max_global_index, max(global_indices))
                    total_samples += len(global_indices)
                    
                    print(f"    {len(global_indices)} samples")
                    print(f"    Global index range: [{min(global_indices)}, {max(global_indices)}]")
                    print(f"    Worker index range: [{min(worker_indices)}, {max(worker_indices)}]")
            else:
                print(f"  Warning: {log_file_path} not found")
    
    print(f"Total samples across all nodes: {total_samples}")
    
    # Set axis limits (adjust based on data)
    ax.set_xlim(xmin=0, xmax=max_worker_index + 2000)
    ax.set_ylim(ymin=0, ymax=max_global_index + 5000)
    
    # Add grid and legend
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9)
    
    # Add title
    ax.set_title(title, fontsize=12, pad=10)
    
    # Save the plot
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        # Default save location if no output file specified
        default_output = os.path.join(os.path.dirname(__file__), "distributed_logs_distribution.png")
        fig.savefig(default_output, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {default_output}")
    
    plt.close()  # Close figure to free memory

def analyze_distributed_shuffle_quality(logs_dir):
    """
    Analyze the quality of shuffle in distributed setting.
    """
    print("\n=== Distributed Shuffle Quality Analysis ===")
    
    nodes = ['node0', 'node1']
    worker_files = [
        "worker_0_samples.txt",
        "worker_1_samples.txt", 
        "worker_2_samples.txt"
    ]
    
    total_samples = 0
    global_file_distribution = {}
    node_stats = {}
    
    for node_id, node_name in enumerate(nodes):
        node_dir = os.path.join(logs_dir, node_name)
        node_stats[node_name] = {'total_samples': 0, 'file_distribution': {}}
        
        if not os.path.exists(node_dir):
            continue
            
        print(f"\n--- {node_name.upper()} Statistics ---")
        
        for worker_id, worker_file in enumerate(worker_files):
            log_file_path = os.path.join(node_dir, worker_file)
            
            if os.path.exists(log_file_path):
                global_indices, worker_indices = parse_distributed_log_file(log_file_path)
                worker_samples = len(global_indices)
                total_samples += worker_samples
                node_stats[node_name]['total_samples'] += worker_samples
                
                # Count samples per file for this worker
                worker_file_dist = {}
                for global_idx in global_indices:
                    file_id = global_idx // 10000
                    worker_file_dist[file_id] = worker_file_dist.get(file_id, 0) + 1
                
                print(f"  Worker {worker_id}: {worker_samples} samples, files: {worker_file_dist}")
                
                # Update node and global file distribution
                for file_id, count in worker_file_dist.items():
                    node_stats[node_name]['file_distribution'][file_id] = \
                        node_stats[node_name]['file_distribution'].get(file_id, 0) + count
                    global_file_distribution[file_id] = \
                        global_file_distribution.get(file_id, 0) + count
                
                # Check shuffle quality for first few samples
                if len(global_indices) >= 5:
                    first_5_global = global_indices[:5]
                    print(f"    First 5 global indices: {first_5_global}")
        
        print(f"  {node_name} total: {node_stats[node_name]['total_samples']} samples")
        print(f"  {node_name} file distribution: {node_stats[node_name]['file_distribution']}")
    
    print(f"\n=== Global Statistics ===")
    print(f"Total samples across all nodes: {total_samples}")
    print(f"Global file distribution: {global_file_distribution}")
    
    # Check for proper distributed partitioning
    print(f"\n=== Distributed Partitioning Analysis ===")
    node0_files = set(node_stats['node0']['file_distribution'].keys())
    node1_files = set(node_stats['node1']['file_distribution'].keys())
    
    print(f"Node0 processes files: {sorted(node0_files)}")
    print(f"Node1 processes files: {sorted(node1_files)}")
    
    overlap = node0_files.intersection(node1_files)
    if overlap:
        print(f"⚠️  File overlap detected: {sorted(overlap)} (This might be expected for some datasets)")
    else:
        print(f"✅ Perfect file partitioning: No overlap between nodes")

if __name__ == '__main__':
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Configuration with relative paths for HDFS distributed logs
    logs_directory = os.path.join(project_root, "examples", "distributed_logs_hdfs")
    plot_title = "CorgiPile HDFS Distributed Processing: 2 Nodes × 3 Workers"
    output_file_path = os.path.join(script_dir, "cifar_logs_hdfs_multi_machine.png")
    
    # Check if logs directory exists
    if not os.path.exists(logs_directory):
        print(f"Error: Distributed logs directory '{logs_directory}' not found!")
        print("Please make sure the path is correct.")
        sys.exit(1)
    
    print("=== CorgiPile HDFS Distributed Logs Visualization ===")
    print(f"HDFS logs directory: {logs_directory}")
    
    # Analyze distributed shuffle quality
    analyze_distributed_shuffle_quality(logs_directory)
    
    # Generate the plot
    print(f"\nGenerating HDFS distributed plot...")
    plot_distributed_logs_distribution(logs_directory, plot_title, output_file_path)
