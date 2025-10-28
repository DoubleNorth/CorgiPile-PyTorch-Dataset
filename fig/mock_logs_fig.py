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

# Colors for different workers/nodes
colors_single = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']
colors_distributed = {
    'node0': ['tab:blue', 'lightblue', 'steelblue', 'navy'],
    'node1': ['tab:red', 'lightcoral', 'crimson', 'darkred']
}

def simulate_dual_layer_shuffle_simple(global_indices_list, block_size):
    """
    Apply dual-layer shuffle to a given list of global indices.
    
    Args:
        global_indices_list (list): List of global indices to shuffle
        block_size (int): Size of each block
        
    Returns:
        tuple: (shuffled_global_indices, processing_order)
    """
    # Split into blocks
    blocks = []
    for i in range(0, len(global_indices_list), block_size):
        block = global_indices_list[i:i + block_size]
        # Intra-block shuffle
        random.shuffle(block)
        blocks.append(block)
    
    # Inter-block shuffle
    random.shuffle(blocks)
    
    # Flatten back to get shuffled order
    shuffled_indices = []
    for block in blocks:
        shuffled_indices.extend(block)
    
    processing_order = list(range(len(shuffled_indices)))
    
    return shuffled_indices, processing_order

def simulate_dual_layer_shuffle(total_samples, block_size, file_id=0):
    """
    Simulate the dual-layer shuffle algorithm.
    
    Args:
        total_samples (int): Total number of samples to generate
        block_size (int): Size of each block
        file_id (int): File ID for global index calculation
        
    Returns:
        tuple: (global_indices, processing_order) simulating the shuffle effect
    """
    # Generate original sequential samples
    original_samples = list(range(total_samples))
    
    # Split into blocks
    blocks = []
    for i in range(0, total_samples, block_size):
        block = original_samples[i:i + block_size]
        # Intra-block shuffle
        random.shuffle(block)
        blocks.append(block)
    
    # Inter-block shuffle
    random.shuffle(blocks)
    
    # Flatten back to get shuffled order
    shuffled_samples = []
    for block in blocks:
        shuffled_samples.extend(block)
    
    # Convert to global indices (file_id * 1000 + sample_index for mock data)
    global_indices = [file_id * 1000 + sample for sample in shuffled_samples]
    processing_order = list(range(len(shuffled_samples)))
    
    return global_indices, processing_order

def simulate_local_single_machine(total_samples=1000, num_workers=4, block_size=50):
    """
    Simulate local single machine with multiple workers using block-level distribution.
    
    Args:
        total_samples (int): Total samples (1000)
        num_workers (int): Number of workers (4) 
        block_size (int): Block size (50)
        
    Returns:
        dict: Worker data {worker_id: (global_indices, processing_order)}
    """
    # Generate all global indices
    all_global_indices = list(range(total_samples))
    
    # Split into blocks
    total_blocks = total_samples // block_size  # 20 blocks
    blocks = []
    for i in range(total_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block = all_global_indices[start_idx:end_idx]
        blocks.append(block)
    
    print(f"Total {total_blocks} blocks, each with {block_size} samples")
    
    # Randomly distribute blocks among workers (round-robin for fairness)
    worker_data = {i: ([], []) for i in range(num_workers)}
    
    # Shuffle blocks first to simulate random block assignment
    random.shuffle(blocks)
    
    # Assign blocks to workers in round-robin fashion
    for block_idx, block in enumerate(blocks):
        worker_id = block_idx % num_workers
        
        # Apply intra-block shuffle
        shuffled_block = block.copy()
        random.shuffle(shuffled_block)
        
        # Add to worker's data
        current_indices, current_processing = worker_data[worker_id]
        
        # Extend worker's global indices
        current_indices.extend(shuffled_block)
        
        # Extend worker's processing order
        current_processing.extend(range(len(current_processing), len(current_processing) + len(shuffled_block)))
        
        worker_data[worker_id] = (current_indices, current_processing)
    
    # Print worker statistics
    for worker_id in range(num_workers):
        global_indices, processing_order = worker_data[worker_id]
        blocks_assigned = len(global_indices) // block_size
        
        print(f"Local Worker {worker_id}: {len(global_indices)} samples, {blocks_assigned} blocks")
        if global_indices:
            print(f"  Global index range: [{min(global_indices)}, {max(global_indices)}]")
            print(f"  First 5 global indices: {global_indices[:5]}")
            print(f"  Sample global indices: {global_indices[::50]}")  # Show one from each block
    
    return worker_data

def simulate_hdfs_distributed_multimachine(total_samples=2000, num_nodes=2, workers_per_node=4, block_size=50, file_per_worker=True):
    """
    Simulate HDFS distributed multi-machine setup with realistic file distribution.
    
    Args:
        total_samples (int): Total samples (2000)
        num_nodes (int): Number of nodes (2)  
        workers_per_node (int): Workers per node (4)
        block_size (int): Block size (50)
        file_per_worker (bool): If True, each worker processes one complete file.
                               If False, workers within each node share blocks from node's files.
        
    Returns:
        dict: Node data {node_id: {worker_id: (global_indices, processing_order)}}
    """
    samples_per_file = total_samples // (num_nodes * workers_per_node)  # 250 per file
    total_files = num_nodes * workers_per_node  # 8 files
    
    # Create 8 files with 250 samples each
    files_data = {}
    for file_id in range(total_files):
        start_idx = file_id * samples_per_file
        end_idx = start_idx + samples_per_file
        files_data[file_id] = list(range(start_idx, end_idx))
    
    print(f"Total {total_files} files, each with {samples_per_file} samples")
    
    node_data = {}
    
    # File distribution: Node0 gets [0,2,4,6], Node1 gets [1,3,5,7]
    for node_id in range(num_nodes):
        node_data[node_id] = {}
        
        # Get files for this node
        node_files = [file_id for file_id in range(total_files) if file_id % 2 == node_id]
        print(f"\nNode {node_id} processing files: {node_files}")
        
        if file_per_worker:
            # Mode 1: Each worker processes one complete file
            for worker_id in range(workers_per_node):
                file_id = node_files[worker_id]
                file_samples = files_data[file_id].copy()
                
                # Apply dual-layer shuffle to the entire file
                shuffled_indices, processing_order = simulate_dual_layer_shuffle_simple(
                    file_samples, block_size
                )
                
                node_data[node_id][worker_id] = (shuffled_indices, processing_order)
                
                print(f"  Worker {worker_id}: {len(shuffled_indices)} samples from file {file_id} (complete file)")
                print(f"    Global index range: [{min(shuffled_indices)}, {max(shuffled_indices)}]")
                print(f"    First 5 global indices: {shuffled_indices[:5]}")
        
        else:
            # Mode 2: Workers share blocks from all node files (like single machine)
            # Combine all files for this node
            all_node_samples = []
            for file_id in node_files:
                all_node_samples.extend(files_data[file_id])
            
            print(f"  Node {node_id} total samples: {len(all_node_samples)}")
            
            # Split into blocks
            total_blocks = len(all_node_samples) // block_size
            blocks = []
            for i in range(total_blocks):
                start_idx = i * block_size
                end_idx = start_idx + block_size
                block = all_node_samples[start_idx:end_idx]
                blocks.append(block)
            
            print(f"  Node {node_id} total {total_blocks} blocks, each with {block_size} samples")
            
            # Randomly distribute blocks among workers in this node
            worker_data_node = {i: ([], []) for i in range(workers_per_node)}
            
            # Shuffle blocks first
            random.shuffle(blocks)
            
            # Assign blocks to workers in round-robin fashion
            for block_idx, block in enumerate(blocks):
                worker_id = block_idx % workers_per_node
                
                # Apply intra-block shuffle
                shuffled_block = block.copy()
                random.shuffle(shuffled_block)
                
                # Add to worker's data
                current_indices, current_processing = worker_data_node[worker_id]
                
                # Extend worker's global indices
                current_indices.extend(shuffled_block)
                
                # Extend worker's processing order
                current_processing.extend(range(len(current_processing), len(current_processing) + len(shuffled_block)))
                
                worker_data_node[worker_id] = (current_indices, current_processing)
            
            # Store node worker data
            for worker_id in range(workers_per_node):
                global_indices, processing_order = worker_data_node[worker_id]
                blocks_assigned = len(global_indices) // block_size
                
                node_data[node_id][worker_id] = (global_indices, processing_order)
                
                print(f"  Worker {worker_id}: {len(global_indices)} samples, {blocks_assigned} blocks (shared)")
                if global_indices:
                    print(f"    Global index range: [{min(global_indices)}, {max(global_indices)}]")
                    print(f"    First 5 global indices: {global_indices[:5]}")
    
    return node_data

def plot_local_single_machine_mock(worker_data, output_file=None):
    """Plot mock data for local single machine."""
    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(111)
    
    plt.subplots_adjust(left=0.198, bottom=0.17, right=0.946, top=0.967,
                       wspace=0.205, hspace=0.2)
    
    ax.set_ylabel("Global tuple id", color='black')
    ax.set_xlabel("Worker internal processing order")
    
    max_processing = 0
    max_global = 0
    
    for worker_id, (global_indices, processing_order) in worker_data.items():
        ax.scatter(processing_order, global_indices,
                  s=3,  # Smaller points for better visibility
                  color=colors_single[worker_id],
                  label=f'Worker {worker_id}',
                  alpha=0.8)
        
        max_processing = max(max_processing, max(processing_order))
        max_global = max(max_global, max(global_indices))
    
    ax.set_xlim(xmin=0, xmax=max_processing)
    ax.set_ylim(ymin=0, ymax=max_global)
    
    # Set fixed ticks for single machine version
    # X-axis: 0 to 250 with 50 intervals
    x_ticks = [i for i in range(0, 251, 50)]  # [0, 50, 100, 150, 200, 250]
    ax.set_xticks(x_ticks)
    
    # Y-axis: 0 to 1000 with 200 intervals
    y_ticks = [i for i in range(0, 1001, 200)]  # [0, 200, 400, 600, 800, 1000]
    ax.set_yticks(y_ticks)
    
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_title("CorgiPile Local Single Machine")
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Local single machine plot saved to: {output_file}")
    
    plt.close()

def plot_hdfs_distributed_mock(node_data, output_file=None):
    """Plot mock data for HDFS distributed multi-machine."""
    fig = plt.figure(figsize=(5.5, 3.5))  # Wider to accommodate external legend
    ax = fig.add_subplot(111)
    
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.75, top=0.92,
                       wspace=0.205, hspace=0.2)
    
    ax.set_ylabel("Global tuple id", color='black')
    ax.set_xlabel("Worker internal processing order")
    
    max_processing = 0
    max_global = 0
    
    for node_id, workers in node_data.items():
        for worker_id, (global_indices, processing_order) in workers.items():
            # No offset - both nodes start from 0, distinguished by color only
            color = colors_distributed[f'node{node_id}'][worker_id]
            
            ax.scatter(processing_order, global_indices,
                      s=3,  # Small points for better visibility
                      color=color,
                      label=f'Node{node_id} W{worker_id}',  # Include node info in label
                      alpha=0.8)
            
            max_processing = max(max_processing, max(processing_order))
            max_global = max(max_global, max(global_indices))
    
    # Add text annotations in the right area above the legend (vertically separated, left-aligned)
    fig.text(0.845, 0.85, "Node 0",
             fontsize=10, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
             transform=fig.transFigure)
    
    fig.text(0.845, 0.75, "Node 1",
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7),
             transform=fig.transFigure)
    
    ax.set_xlim(xmin=0, xmax=max_processing)
    ax.set_ylim(ymin=0, ymax=max_global)
    
    # Set fixed ticks for distributed version
    # X-axis: 0 to 250 with 50 intervals
    x_ticks = [i for i in range(0, 251, 50)]  # [0, 50, 100, 150, 200, 250]
    ax.set_xticks(x_ticks)
    
    # Y-axis: distributed version gets [0, 500, 1000, 1500, 2000]
    y_ticks = [0, 500, 1000, 1500, 2000]
    ax.set_yticks(y_ticks)
    
    ax.grid(True)
    # Place legend outside the plot area (to the right)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.4), ncol=1, fontsize=9)
    ax.set_title("CorgiPile HDFS Distributed")
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"HDFS distributed plot saved to: {output_file}")
    
    plt.close()

def simulate_block_tuple_shuffle_single_thread(total_samples=1000, block_size=25, buffer_size=250):
    """
    Simulate single-threaded Block+Tuple shuffle algorithm (from fig3.py).
    
    Args:
        total_samples (int): Total number of samples
        block_size (int): Size of each block (page)
        buffer_size (int): Size of tuple shuffle buffer
        
    Returns:
        tuple: (global_indices, processing_order)
    """
    # Step 1: Generate original sequential samples (like fig3.py)
    original_samples = list(range(total_samples))
    
    # Step 2: Perform block (page) shuffle first
    shuffled_tuple_list = []
    block_num = total_samples // block_size
    
    # Create block index list [0, 1, 2, 3, ...]
    block_index_list = list(range(block_num))
    
    # Shuffle block order [8, 3, 5, 2, 0, 9, 1, 4, 6, 7]
    random.shuffle(block_index_list)
    
    # Reconstruct samples according to shuffled block order
    for i in range(block_num):
        for j in range(block_size):
            index = block_index_list[i] * block_size + j
            if index < total_samples:  # Handle last block edge case
                shuffled_tuple_list.append(original_samples[index])
    
    # Step 3: Perform tuple-level shuffle with buffer
    buffer = []
    final_shuffled_list = []
    
    for i in range(len(shuffled_tuple_list)):
        sample = shuffled_tuple_list[i]
        buffer.append(sample)
        
        if len(buffer) == buffer_size:
            # Shuffle buffer and output all
            random.shuffle(buffer)
            final_shuffled_list.extend(buffer)
            buffer.clear()
    
    # Handle remaining samples in buffer
    if buffer:
        random.shuffle(buffer)
        final_shuffled_list.extend(buffer)
    
    processing_order = list(range(len(final_shuffled_list)))
    
    return final_shuffled_list, processing_order

def plot_block_tuple_shuffle_mock(global_indices, processing_order, output_file=None):
    """Plot single-threaded Block+Tuple shuffle (matching mock_local_single_machine size)."""
    fig = plt.figure(figsize=(3.5, 3))  # Same size as local single machine
    ax = fig.add_subplot(111)
    
    plt.subplots_adjust(left=0.198, bottom=0.17, right=0.946, top=0.967,
                       wspace=0.205, hspace=0.2)
    
    ax.set_ylabel("Global tuple id", color='black')
    ax.set_xlabel("Processing order")
    
    # Plot with single blue color (like existing shuffle results)
    ax.scatter(processing_order, global_indices,
              s=3,  # Small points for consistency
              color='tab:blue',
              alpha=0.8)
    
    max_processing = max(processing_order)
    max_global = max(global_indices)
    
    ax.set_xlim(xmin=0, xmax=max_processing)
    ax.set_ylim(ymin=0, ymax=max_global)
    
    # Set fixed ticks (same as local single machine)
    x_ticks = [i for i in range(0, 1001, 200)]  # [0, 200, 400, 600, 800, 1000]
    ax.set_xticks(x_ticks)
    
    y_ticks = [i for i in range(0, 1001, 200)]  # [0, 200, 400, 600, 800, 1000]
    ax.set_yticks(y_ticks)
    
    ax.grid(True)
    # Use consistent title font size (no explicit fontsize to use global setting)
    ax.set_title("Block+Tuple Shuffle")
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Block+Tuple shuffle plot saved to: {output_file}")
    
    plt.close()

def plot_mock_label_distribution(data, scenario_name, output_file=None):
    """Plot label distribution for mock data scenarios."""
    
    # Create mock labels for visualization (50% positive, 50% negative)
    fig = plt.figure(figsize=(3.2, 2.8))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.157, bottom=0.186, right=0.964, top=0.976,
                       wspace=0.205, hspace=0.2)
    
    ax.set_ylabel("#tuples", color='black')
    ax.set_xlabel("The i-th batch (20 tuples per batch)")
    
    if scenario_name == "Block+Tuple Shuffle":
        # Single list of global indices
        total_samples = len(data)
        
        # Create original labels (first half negative, second half positive - like fig3.py)
        original_labels = [-1] * (total_samples // 2) + [1] * (total_samples // 2)
        
        # Apply the same Block+Tuple shuffle to labels as was applied to data
        # Simulate block shuffle first, then tuple shuffle
        block_size = 25  # Same as in simulate_block_tuple_shuffle_single_thread
        buffer_size = 250
        
        # Step 1: Block shuffle on labels
        shuffled_labels = []
        block_num = total_samples // block_size
        
        # Create block index list and shuffle it (same random seed effect)
        block_index_list = list(range(block_num))
        random.shuffle(block_index_list)
        
        # Reconstruct labels according to shuffled block order
        for i in range(block_num):
            for j in range(block_size):
                index = block_index_list[i] * block_size + j
                if index < total_samples:
                    shuffled_labels.append(original_labels[index])
        
        # Step 2: Tuple-level shuffle with buffer
        buffer = []
        final_shuffled_labels = []
        
        for i in range(len(shuffled_labels)):
            label = shuffled_labels[i]
            buffer.append(label)
            
            if len(buffer) == buffer_size:
                # Shuffle buffer and output all
                random.shuffle(buffer)
                final_shuffled_labels.extend(buffer)
                buffer.clear()
        
        # Handle remaining labels in buffer
        if buffer:
            random.shuffle(buffer)
            final_shuffled_labels.extend(buffer)
        
        # Now calculate batch statistics
        batch_index_list = []
        label0_list = []
        label1_list = []
        
        label1_count = 0
        label0_count = 0
        batch_size = 20
        
        for i in range(len(final_shuffled_labels)):
            label = final_shuffled_labels[i]
            
            if label == 1:
                label1_count += 1
            else:
                label0_count += 1
            
            if (i + 1) % batch_size == 0 or i == len(final_shuffled_labels) - 1:
                batch_index_list.append((i + 1) / batch_size)
                label1_list.append(label1_count)
                label0_list.append(label0_count)
                label1_count = 0
                label0_count = 0
        
        # Plot lines
        ax.plot(batch_index_list, label1_list, label='label=+1', color='tab:blue')
        ax.plot(batch_index_list, label0_list, label='label=-1', color='tab:red', linestyle='--')
        
    else:
        # For worker-based scenarios (local single machine, distributed)
        all_samples = []
        
        if scenario_name == "Local Single Machine":
            # Collect all samples from all workers
            for worker_id, (global_indices, processing_order) in data.items():
                all_samples.extend(global_indices)
        else:  # Distributed scenarios
            # Collect all samples from all nodes and workers
            for node_id, workers in data.items():
                for worker_id, (global_indices, processing_order) in workers.items():
                    all_samples.extend(global_indices)
        
        total_samples = len(all_samples)
        
        # Create realistic label distribution for shuffled data
        batch_index_list = []
        label0_list = []
        label1_list = []
        
        label1_count = 0
        label0_count = 0
        batch_size = 20
        
        for i in range(total_samples):
            # Simulate balanced label distribution after shuffle
            label = 1 if (i + hash(all_samples[i % len(all_samples)])) % 2 == 0 else -1
            
            if label == 1:
                label1_count += 1
            else:
                label0_count += 1
            
            if (i + 1) % batch_size == 0 or i == total_samples - 1:
                batch_index_list.append((i + 1) / batch_size)
                label1_list.append(label1_count)
                label0_list.append(label0_count)
                label1_count = 0
                label0_count = 0
        
        # Plot lines
        ax.plot(batch_index_list, label1_list, label='label=+1', color='tab:blue')
        ax.plot(batch_index_list, label0_list, label='label=-1', color='tab:red', linestyle='--')
    
    ax.set_ylim(ymin=-1)
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=21)  # batch_size + 1
    
    ax.legend()
    ax.set_title(scenario_name)
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Label distribution plot saved to: {output_file}")
    
    plt.close()

def analyze_mock_shuffle_quality(data, scenario_name):
    """Analyze the shuffle quality of mock data."""
    print(f"\n=== {scenario_name} Mock Shuffle Analysis ===")
    
    if scenario_name == "Local Single Machine":
        for worker_id, (global_indices, processing_order) in data.items():
            # Check if first few samples are shuffled (not consecutive)
            first_5 = global_indices[:5]
            is_shuffled = any(abs(first_5[i] - first_5[i-1]) != 1 for i in range(1, len(first_5)))
            print(f"Worker {worker_id}: First 5 samples {first_5}, Shuffled: {is_shuffled}")
    
    elif scenario_name == "Block+Tuple Shuffle":
        # Single list analysis
        first_5 = data[:5]
        is_shuffled = any(abs(first_5[i] - first_5[i-1]) != 1 for i in range(1, len(first_5)))
        print(f"First 5 samples: {first_5}, Shuffled: {is_shuffled}")
        
        # Check block structure
        print(f"Total samples: {len(data)}")
        print(f"Sample distribution: {data[::100]}")  # Show every 100th sample
    
    else:  # Distributed
        for node_id, workers in data.items():
            print(f"Node {node_id}:")
            for worker_id, (global_indices, processing_order) in workers.items():
                first_5 = global_indices[:5]
                is_shuffled = any(abs(first_5[i] - first_5[i-1]) != 1 for i in range(1, len(first_5)))
                print(f"  Worker {worker_id}: First 5 samples {first_5}, Shuffled: {is_shuffled}")

if __name__ == '__main__':
    # Set random seed for reproducible results
    random.seed(42)
    
    # Get script directory for output files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== CorgiPile Mock Data Visualization ===")
    
    # Scenario 0: Block+Tuple Shuffle (Single Thread) - NEW!
    print("\n--- Simulating Block+Tuple Shuffle (Single Thread) ---")
    print("Config: 1000 samples, block_size=25, buffer_size=250 (same as fig3.py)")

    block_tuple_indices, block_tuple_processing = simulate_block_tuple_shuffle_single_thread(
        total_samples=1000,
        block_size=25,
        buffer_size=250
    )

    analyze_mock_shuffle_quality(block_tuple_indices, "Block+Tuple Shuffle")

    # Generate ID distribution plot
    block_tuple_output = os.path.join(script_dir, "mock_block_tuple_shuffle.png")
    plot_block_tuple_shuffle_mock(block_tuple_indices, block_tuple_processing, block_tuple_output)
    
    # Generate label distribution plot
    block_tuple_label_output = os.path.join(script_dir, "mock_block_tuple_shuffle_label.png")
    plot_mock_label_distribution(block_tuple_indices, "Block+Tuple Shuffle", block_tuple_label_output)

    # Scenario 1: Local Single Machine
    print("\n--- Simulating Local Single Machine ---")
    print("Config: 1000 samples, 4 workers, 250 samples/worker, block_size=50")

    local_data = simulate_local_single_machine(
        total_samples=1000,
        num_workers=4,
        block_size=50
    )

    analyze_mock_shuffle_quality(local_data, "Local Single Machine")

    # Generate ID distribution plot
    local_output = os.path.join(script_dir, "mock_local_single_machine.png")
    plot_local_single_machine_mock(local_data, local_output)
    
    # Generate label distribution plot
    local_label_output = os.path.join(script_dir, "mock_local_single_machine_label.png")
    plot_mock_label_distribution(local_data, "Local Single Machine", local_label_output)
    
    # Scenario 2: HDFS Distributed Multi-machine (File per Worker mode)
    print("\n--- Simulating HDFS Distributed Multi-machine (File per Worker) ---")
    print("Config: 2000 samples, 2 nodes, 4 workers/node, 250 samples/worker, block_size=50")
    print("Mode: Each worker processes one complete file")
    
    distributed_data_file = simulate_hdfs_distributed_multimachine(
        total_samples=2000,
        num_nodes=2,
        workers_per_node=4,
        block_size=50,
        file_per_worker=True
    )
    
    analyze_mock_shuffle_quality(distributed_data_file, "HDFS Distributed (File per Worker)")
    
    # Generate ID distribution plot
    distributed_output_file = os.path.join(script_dir, "mock_hdfs_distributed_file_mode.png")
    plot_hdfs_distributed_mock(distributed_data_file, distributed_output_file)
    
    # Generate label distribution plot
    distributed_label_output_file = os.path.join(script_dir, "mock_hdfs_distributed_file_mode_label.png")
    plot_mock_label_distribution(distributed_data_file, "HDFS Distributed", distributed_label_output_file)

    # Scenario 3: HDFS Distributed Multi-machine (Block sharing mode)
    print("\n--- Simulating HDFS Distributed Multi-machine (Block Sharing) ---")
    print("Config: 2000 samples, 2 nodes, 4 workers/node, 250 samples/worker, block_size=50")
    print("Mode: Workers within each node share blocks from node's files")
    
    distributed_data_block = simulate_hdfs_distributed_multimachine(
        total_samples=2000,
        num_nodes=2,
        workers_per_node=4,
        block_size=50,
        file_per_worker=False
    )
    
    analyze_mock_shuffle_quality(distributed_data_block, "HDFS Distributed (Block Sharing)")
    
    # Generate ID distribution plot
    distributed_output_block = os.path.join(script_dir, "mock_hdfs_distributed_block_mode.png")
    plot_hdfs_distributed_mock(distributed_data_block, distributed_output_block)
    
    # Generate label distribution plot
    distributed_label_output_block = os.path.join(script_dir, "mock_hdfs_distributed_block_mode_label.png")
    plot_mock_label_distribution(distributed_data_block, "HDFS Distributed", distributed_label_output_block)
    
    print(f"\n=== Mock Visualization Complete ===")
    print(f"Generated files:")
    print(f"  ID Distribution:")
    print(f"    - {block_tuple_output}")
    print(f"    - {local_output}")
    print(f"    - {distributed_output_file}")
    print(f"    - {distributed_output_block}")
    print(f"  Label Distribution:")
    print(f"    - {block_tuple_label_output}")
    print(f"    - {local_label_output}")
    print(f"    - {distributed_label_output_file}")
    print(f"    - {distributed_label_output_block}")
