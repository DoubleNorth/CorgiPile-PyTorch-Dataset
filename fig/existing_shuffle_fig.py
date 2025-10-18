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

# Apply same style settings as fig3.py and mock_logs_fig.py
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

# Use single blue color for all plots
shuffle_color = 'tab:blue'

class Tuple:
    def __init__(self, id, label):
        self.id = id
        self.label = label

    def get_id(self):
        return self.id

    def get_label(self):
        return self.label

class Table:
    def __init__(self, tuple_num, label_1_ratio):
        self.index_list = []
        self.tuple_num = tuple_num
        self.tuple_list = []

        label_1_num = tuple_num * label_1_ratio
        label__1_num = tuple_num - label_1_num

        for i in range(0, tuple_num):
            label = -1
            if i >= label__1_num:
                label = 1
            tuple = Tuple(i, label)
            self.tuple_list.append(tuple)

            self.index_list.append(i)

    def get_index_list(self):
        return self.index_list

    def perform_no_shuffle(self):
        return self.tuple_list

    def perform_fully_shuffle(self):
        random.shuffle(self.tuple_list)
        return self.tuple_list

    def perform_sliding_window_shuffle(self):
        original_list = self.tuple_list
        total_size = len(original_list)
        window_size = int(0.1 * total_size)
        window = []
        new_list = []

        for i in range(0, window_size):
            window.append(original_list[i])

        for i in range(window_size, total_size):
            index = random.randint(0, window_size - 1)
            new_list.append(window[index])
            window[index] = original_list[i]
        
        random.shuffle(window)
        for t in window:
            new_list.append(t)
        assert(len(new_list) == total_size)
        self.tuple_list = new_list

        return self.tuple_list
    
    def perform_mrs_shuffle(self):
        original_list = self.tuple_list
        total_size = len(original_list)

        window_size = int(0.1 * total_size)
        window = []
        new_list = []

        sample_list = random.sample(original_list, window_size)
        final_list = []

        for i in range(0, window_size):
            window.append(original_list[i])
        
        for i in range(window_size, total_size):
            index = random.randint(0, window_size - 1)
            new_list.append(window[index])
            window[index] = original_list[i]
        
        for t in window:
            new_list.append(t)

        sample_index = 0
        new_list_index = 0
        for i in range(0, total_size):
            if random.random() <= 0.35:
                final_list.append(sample_list[sample_index])
                sample_index =  (sample_index + 1) % window_size
            else:
                final_list.append(new_list[new_list_index])
                new_list_index += 1

        self.tuple_list = final_list
           
        return self.tuple_list

    def perform_only_page_shuffle(self, block_tuple_num):
        shuffled_tuple_list = []

        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        block_index_list = []

        block_num = int(self.tuple_num / block_tuple_num)
        for i in range(0, block_num):
            block_index_list.append(i)
        
        # [8, 3, 5, 2, 0, 9, 1, 4, 6, 7]
        random.shuffle(block_index_list)

        for i in range(0, block_num):
            for j in range(0, block_tuple_num):
                index = block_index_list[i] * block_tuple_num + j
                tuple = self.tuple_list[index]
                shuffled_tuple_list.append(tuple)
        
        return shuffled_tuple_list
    
    def perform_page_tuple_shuffle(self, block_tuple_num, buffer_tuple_num):
        page_shuffled_tuple_list = self.perform_only_page_shuffle(block_tuple_num)
        buffer = []

        page_tuple_shuffled_list = []

        for i in range(0, self.tuple_num):
            tuple = page_shuffled_tuple_list[i]

            buffer.append(tuple)

            if len(buffer) == buffer_tuple_num:
                random.shuffle(buffer)
                for t in buffer:
                    page_tuple_shuffled_list.append(t)
                buffer.clear()
            
        if buffer:
            for t in buffer:
                page_tuple_shuffled_list.append(t)
        
        return page_tuple_shuffled_list

def plot_shuffle_method(shuffle_mode, tuple_num, label_1_ratio, block_tuple_num, buffer_tuple_num, title, output_file):
    """Plot a single shuffle method with CorgiPile style."""
    
    # Create table and get shuffled data
    table = Table(tuple_num, label_1_ratio)
    tuple_list = []
    
    if shuffle_mode == 'no_shuffle':
        tuple_list = table.perform_no_shuffle()
    elif shuffle_mode == 'fully_shuffle':
        tuple_list = table.perform_fully_shuffle()
    elif shuffle_mode == 'only_page_shuffle':
        tuple_list = table.perform_only_page_shuffle(block_tuple_num)
    elif shuffle_mode == 'page_tuple_shuffle':
        tuple_list = table.perform_page_tuple_shuffle(block_tuple_num, buffer_tuple_num)
    elif shuffle_mode == 'sliding_window_shuffle':
        tuple_list = table.perform_sliding_window_shuffle()
    elif shuffle_mode == 'mrs_shuffle':
        tuple_list = table.perform_mrs_shuffle()
    
    # Prepare data for plotting
    tuple_ids = []
    processing_order = []
    
    for i in range(len(tuple_list)):
        tuple = tuple_list[i]
        tuple_id = tuple.get_id()
        tuple_ids.append(tuple_id)
        processing_order.append(i)
    
    # Create plot with CorgiPile style
    fig = plt.figure(figsize=(3.5, 3))
    ax = fig.add_subplot(111)
    
    plt.subplots_adjust(left=0.198, bottom=0.17, right=0.946, top=0.967,
                       wspace=0.205, hspace=0.2)
    
    ax.set_ylabel("Global tuple id", color='black')
    ax.set_xlabel("Processing order")
    
    # Plot with single blue color
    ax.scatter(processing_order, tuple_ids,
              s=1,  # Small points for consistency
              color=shuffle_color,
              alpha=0.8)
    
    # Set axis limits and ticks (similar to mock_logs_fig.py)
    max_processing = max(processing_order)
    max_global = max(tuple_ids)
    
    ax.set_xlim(xmin=0, xmax=max_processing)
    ax.set_ylim(ymin=0, ymax=max_global)
    
    # Set fixed ticks
    x_ticks = [i for i in range(0, max_processing + 1, 200)]  # 200 intervals for X
    if 1000 not in x_ticks:
        x_ticks.append(1000)
    ax.set_xticks(x_ticks)
    
    y_ticks = [i for i in range(0, max_global + 1, 200)]  # 200 intervals for Y
    if 1000 not in y_ticks:
        y_ticks.append(1000)
    ax.set_yticks(y_ticks)
    
    ax.grid(True)
    ax.set_title(title, fontsize=12, pad=10)
    
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()

def generate_all_shuffle_methods():
    """Generate all shuffle method visualizations."""
    
    # Parameters (same as fig3.py)
    tuple_num = 1000
    block_tuple_num = 25
    buffer_tuple_num = 250
    label_1_ratio = 0.5
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "existing_shuffle_results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Existing Shuffle Methods Visualization ===")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: {tuple_num} tuples, block_size={block_tuple_num}, buffer_size={buffer_tuple_num}")
    
    # Set random seed for reproducible results
    random.seed(42)
    
    # Define all shuffle methods to visualize
    shuffle_methods = [
        {
            'mode': 'no_shuffle',
            'title': 'No Shuffle',
            'filename': 'no_shuffle.png'
        },
        {
            'mode': 'fully_shuffle',
            'title': 'Fully Shuffle',
            'filename': 'fully_shuffle.png'
        },
        {
            'mode': 'only_page_shuffle',
            'title': 'Only Block Shuffle',
            'filename': 'only_block_shuffle.png'
        },
        {
            'mode': 'page_tuple_shuffle',
            'title': 'Block+Tuple Shuffle',
            'filename': 'block_tuple_shuffle.png'
        },
        {
            'mode': 'sliding_window_shuffle',
            'title': 'Sliding Window Shuffle',
            'filename': 'sliding_window_shuffle.png'
        },
        {
            'mode': 'mrs_shuffle',
            'title': 'MRS Shuffle',
            'filename': 'mrs_shuffle.png'
        }
    ]
    
    # Generate all visualizations
    for method in shuffle_methods:
        print(f"\nGenerating {method['title']}...")
        
        output_file = os.path.join(output_dir, method['filename'])
        
        plot_shuffle_method(
            shuffle_mode=method['mode'],
            tuple_num=tuple_num,
            label_1_ratio=label_1_ratio,
            block_tuple_num=block_tuple_num,
            buffer_tuple_num=buffer_tuple_num,
            title=method['title'],
            output_file=output_file
        )
    
    print(f"\n=== All shuffle methods generated successfully ===")
    print(f"Check output directory: {output_dir}")

if __name__ == '__main__':
    generate_all_shuffle_methods()
