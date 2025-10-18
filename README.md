# CorgiPile Dataset API

A high-performance, distributed dataset loading library for PyTorch with advanced shuffling algorithms and seamless scaling from single-machine to multi-machine distributed training.

## Core Advantages

![CorgiPile Family](docs/figures/corgipile_family.png)

**Left: Advanced Dual-Layer Shuffle | Center: Single-Machine Parallelism | Right: Multi-Machine Distribution**

<details>
<summary>View All Shuffle Methods Comparison</summary>

| Method | Visualization | Description |
|--------|---------------|-------------|
| **No Shuffle** | <img src="docs/figures/no_shuffle.png" alt="No Shuffle" width="300"/> | Sequential processing - poor training quality |
| **Sliding Window** | <img src="docs/figures/sliding_window_shuffle.png" alt="Sliding Window" width="300"/> | Window-based shuffle - limited randomness |
| **MRS** | <img src="docs/figures/mrs_shuffle.png" alt="MRS" width="300"/> | Multiplexd reservoir sampling - complex implementation |
| **CorgiPile Dual-Layer** | <img src="docs/figures/block_tuple_shuffle.png" alt="Block+Tuple" width="300"/> | **Optimal balance of randomness and efficiency** |
| **Fully Random(ideal)** | <img src="docs/figures/fully_shuffle.png" alt="Fully Shuffle" width="300"/> | Complete randomization - memory intensive |

</details>

### Key Features

- **Superior Shuffle Quality**: Block+Tuple dual-layer algorithm provides optimal balance of randomness and efficiency
- **Efficient Single-Machine Parallelism**: Automatic load balancing across workers with zero data overlap
- **Scalable Multi-Machine Distribution**: Seamless distributed training with file-level partitioning
- **Memory Efficient**: Block-level processing streams large datasets without loading everything into memory
- **Complete Traceability**: Every sample tracked with (file_id, inner_index) for debugging and reproducibility
- **Universal Storage Support**: Local filesystem and HDFS integration with extensible design

## Installation

```bash
git clone https://github.com/yourusername/corgipile-dataset-api.git
cd corgipile-dataset-api

# Install dependencies
pip install torch>=1.12.0 numpy>=1.21.0 filelock>=3.7.0

# For HDFS support (optional)
pip install pyarrow>=8.0.0
```

## Quick Start

### Basic Local Usage

```python
import torch
from torch.utils.data import DataLoader
from corgipile_dataset_api import CorgiPileLocalDataset

def my_data_loader(file_path: str, **kwargs):
    """Custom function to load data from files."""
    file_id = kwargs.get('file_id', 0)
    
    with open(file_path, 'r') as f:
        for inner_idx, line in enumerate(f):
            data, label = line.strip().split('\t')
            yield (data, int(label), (file_id, inner_idx))

# Create dataset
dataset = CorgiPileLocalDataset(
    data_dir="/path/to/your/data",
    block_size=100,
    load_data_fn=my_data_loader,
    shuffle=True,
    log_dir="./logs"  # Optional logging
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# Train your model
for batch_idx, (data, labels) in enumerate(dataloader):
    # Your training code here
    pass
```

### Distributed Training

```python
from corgipile_dataset_api import CorgiPileDistributedLocalDataset

# Multi-machine setup
dataset = CorgiPileDistributedLocalDataset(
    data_dir="/shared/training/data",
    block_size=100,
    load_data_fn=my_data_loader,
    rank=rank,  # Current machine rank
    world_size=world_size  # Total number of machines
)
```

### HDFS Support

```python
from corgipile_dataset_api import CorgiPileHDFSDataset

# Single-machine HDFS
dataset = CorgiPileHDFSDataset(
    hdfs_root="/user/data/training",
    hdfs_host="namenode-host",
    hdfs_port=9000,
    hdfs_user="hadoop-user",
    block_size=100,
    load_data_fn=hdfs_data_loader,
    shuffle=True
)
```

### Distributed HDFS Training

```python
from corgipile_dataset_api import CorgiPileDistributedHDFSDataset

# Multi-machine HDFS setup
dataset = CorgiPileDistributedHDFSDataset(
    hdfs_root="/user/data/training",
    hdfs_host="namenode-host", 
    hdfs_port=9000,
    hdfs_user="hadoop-user",
    block_size=100,
    load_data_fn=hdfs_data_loader,
    rank=rank,  # Current machine rank
    world_size=world_size,  # Total number of machines
    shuffle=True
)
```

## API Reference

### CorgiPileLocalDataset

**Parameters:**
- `data_dir` (str): Root directory containing data files
- `block_size` (int): Number of samples per block
- `load_data_fn` (Callable): Function to load data from file path
- `shuffle` (bool): Enable dual-layer shuffle. Default: True
- `log_dir` (Optional[str]): Directory for logging. If None, no logging
- `file_filter_fn` (Optional[Callable]): Function to filter valid files

### CorgiPileDistributedLocalDataset

**Additional Parameters:**
- `rank` (int): Current machine rank. Default: 0
- `world_size` (int): Total number of machines. Default: 1

### CorgiPileHDFSDataset

**Additional Parameters:**
- `hdfs_root` (str): HDFS root directory path
- `hdfs_host` (str): HDFS namenode hostname
- `hdfs_port` (int): HDFS namenode port
- `hdfs_user` (str): HDFS username

### CorgiPileDistributedHDFSDataset

**Additional Parameters:**
- `hdfs_root` (str): HDFS root directory path
- `hdfs_host` (str): HDFS namenode hostname  
- `hdfs_port` (int): HDFS namenode port
- `hdfs_user` (str): HDFS username
- `rank` (int): Current machine rank. Default: 0
- `world_size` (int): Total number of machines. Default: 1

## Architecture

### Block-Level Processing
- Memory-efficient streaming of large datasets
- Configurable block sizes for optimal performance
- Better CPU cache utilization through locality

### Dual-Layer Shuffle Algorithm
1. **Inter-block shuffle**: Randomize the order of data blocks
2. **Intra-block shuffle**: Shuffle samples within each block
3. **Result**: Superior training randomness with controlled memory usage

### Sample Traceability
Every sample includes source information `(file_id, inner_index)` enabling:
- Debugging data loading issues
- Reproducible training
- Data lineage tracking

## Performance Tips

1. **Choose appropriate block_size**: Balance memory usage and shuffle quality
   - Smaller blocks: Better shuffle, more memory overhead
   - Larger blocks: Better performance, less randomness
2. **Worker Configuration**: Set `num_workers > 0` in DataLoader for parallel processing
3. **HDFS Optimization**: Use `multiprocessing_context='spawn'` for HDFS datasets

## Examples

The `examples/` directory contains comprehensive examples:

```bash
# CIFAR-10 local mode
python examples/cifar_example.py --mode local

# Single-machine HDFS
python examples/cifar_example.py --mode hdfs

# Multi-machine local deployment
python examples/distributed_example.py --mode local --rank n  

# Multi-machine HDFS deployment
python examples/distributed_example.py --mode hdfs --rank n
```

