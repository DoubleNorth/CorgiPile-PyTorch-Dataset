# CorgiPile Dataset API

A high-performance, distributed dataset loading library for PyTorch that supports block-level data processing, dual-layer shuffling, and both local and HDFS storage backends.

## Features

🚀 **High Performance**
- Block-level data processing for efficient memory usage
- Dual-layer shuffle (intra-block + inter-block) for better training randomness
- Multi-worker support with automatic load balancing

🌐 **Distributed Training**
- Single-machine and multi-machine training modes
- File-level distribution across machines with no data overlap
- Sample traceability and logging for debugging

📁 **Storage Flexibility**
- Local filesystem support
- HDFS (Hadoop Distributed File System) support
- Extensible design for other storage backends

🔧 **Easy to Use**
- PyTorch Dataset API compatibility
- Custom data loader functions
- Optional logging with configurable output

## Installation

### Clone and Use

```bash
git clone https://github.com/yourusername/corgipile-dataset-api.git
cd corgipile-dataset-api

# Install dependencies
pip install torch>=1.12.0 numpy>=1.21.0 filelock>=3.7.0

# For HDFS support (optional)
pip install pyarrow>=8.0.0

# For examples (optional)  
pip install torchvision pandas pillow matplotlib
```

## Quick Start

### Basic Usage with Local Files

```python
import torch
from torch.utils.data import DataLoader
from corgipile_dataset_api import CorgiPileLocalDataset

def my_data_loader(file_path: str, **kwargs):
    """Custom function to load data from files."""
    file_id = kwargs.get('file_id', 0)
    
    # Your data loading logic here
    # Return: (data, label, (file_id, inner_index))
    with open(file_path, 'r') as f:
        for inner_idx, line in enumerate(f):
            data, label = line.strip().split('\t')
            yield (data, int(label), (file_id, inner_idx))

# Create dataset
dataset = CorgiPileLocalDataset(
    data_dir="/path/to/your/data",
    block_size=100,
    load_data_fn=my_data_loader,
    file_filter_fn=lambda x: x.endswith('.txt'),  # Optional file filter
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

### HDFS Usage

```python
from corgipile_dataset_api import CorgiPileHDFSDataset
import pyarrow.fs

def hdfs_data_loader(file_path: str, hdfs: pyarrow.fs.FileSystem, **kwargs):
    """Load data from HDFS files."""
    file_id = kwargs.get('file_id', 0)
    
    with hdfs.open_input_file(file_path) as f:
        content = f.read()
        # Process content and yield samples
        # Return: (data, label, (file_id, inner_index))

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

### Distributed Training

```python
from corgipile_dataset_api import CorgiPileDistributedLocalDataset

# In your distributed training script
dataset = CorgiPileDistributedLocalDataset(
    data_dir="/shared/training/data",
    block_size=100,
    load_data_fn=my_data_loader,
    rank=rank,  # Current machine rank
    world_size=world_size  # Total number of machines
)
```

## API Reference

### CorgiPileLocalDataset

Single-machine dataset for local filesystem.

**Parameters:**
- `data_dir` (str): Root directory containing data files
- `block_size` (int): Number of samples per block
- `load_data_fn` (Callable): Function to load data from file path
- `file_filter_fn` (Optional[Callable]): Function to filter valid files
- `shuffle` (bool): Enable dual-layer shuffle. Default: True
- `log_dir` (Optional[str]): Directory for logging. If None, no logging
- `**kwargs`: Additional arguments passed to load_data_fn

### CorgiPileHDFSDataset

Single-machine dataset for HDFS.

**Additional Parameters:**
- `hdfs_root` (str): HDFS root directory path
- `hdfs_host` (str): HDFS namenode hostname
- `hdfs_port` (int): HDFS namenode port
- `hdfs_user` (str): HDFS username

### CorgiPileDistributedLocalDataset

Multi-machine dataset for local filesystem.

**Additional Parameters:**
- `rank` (int): Current machine rank. Default: 0
- `world_size` (int): Total number of machines. Default: 1

### CorgiPileDistributedHDFSDataset

Multi-machine dataset for HDFS.

Combines all parameters from CorgiPileHDFSDataset and CorgiPileDistributedLocalDataset.

## Custom Data Loader Functions

Your `load_data_fn` should follow this signature:

```python
def load_data_fn(file_path: str, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """
    Load data from a file.
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments including:
            - file_id: Unique identifier for the file
            - worker_id: ID of the worker processing this file
            - rank: Machine rank (for distributed datasets)
    
    Yields:
        Tuple of (data, label, (file_id, inner_index))
    """
    file_id = kwargs.get('file_id', 0)
    
    # Your data loading implementation
    for inner_idx, (data, label) in enumerate(load_your_data(file_path)):
        yield (data, label, (file_id, inner_idx))
```

## Examples

The `examples/` directory contains comprehensive examples:

### CIFAR-10 Example

```bash
python examples/cifar_example.py --mode local
python examples/cifar_example.py --mode hdfs
```

### Distributed Training Example

```bash
# Local filesystem mode
python examples/distributed_example.py --mode local --data-dir /path/to/cifar-10

# HDFS mode  
python examples/distributed_example.py --mode hdfs \
  --hdfs-root /user/data/cifar-10 --hdfs-host namenode --hdfs-user hadoop

# Multi-machine deployment
# Machine 0:
python examples/distributed_example.py --mode local \
  --data-dir /shared/cifar-10 --master-addr master

# Machine 1:
python examples/distributed_example.py --mode local \
  --data-dir /shared/cifar-10 --master-addr <machine-0-ip>
```


## Advanced Features

### Block-level Processing

Data is processed in configurable blocks, enabling:
- Memory-efficient streaming of large datasets
- Fine-grained shuffle control
- Better cache utilization

### Dual-layer Shuffle

1. **Intra-block shuffle**: Samples within each block are shuffled
2. **Inter-block shuffle**: The order of blocks is shuffled

This provides better randomness than traditional dataset shuffling while maintaining block-level locality.

### Sample Traceability

Every sample includes source information `(file_id, inner_index)` enabling:
- Debugging data loading issues
- Reproducible training
- Data lineage tracking

### Logging

When `log_dir` is provided, the dataset creates detailed logs:
```
logs/
├── worker_0_samples.txt
├── worker_1_samples.txt
└── ...
```

Each log contains the processing order: `file_id-inner_index`

## Performance Tips

1. **Choose appropriate block_size**: Balance memory usage and shuffle quality
   - Smaller blocks: Better shuffle, more memory overhead
   - Larger blocks: Less memory, reduced shuffle quality

2. **Use multiple workers**: Set `num_workers > 0` in DataLoader for parallel processing

3. **File distribution**: Ensure files are evenly sized for balanced worker loads

4. **HDFS optimization**: Use `multiprocessing_context='spawn'` for HDFS datasets
