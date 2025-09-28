"""
True multi-machine distributed training example (supports explicit rank specification)
"""

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import os
import time
import argparse
import pickle
import numpy as np
from io import BytesIO
from datetime import timedelta
from typing import Iterable, Tuple, Any
import pyarrow.fs

from corgipile_dataset_api.local.distributed import CorgiPileDistributedLocalDataset
from corgipile_dataset_api.hdfs.distributed import CorgiPileDistributedHDFSDataset

def init_distributed_process(rank: int, world_size: int, master_addr: str, master_port: str):
    """Initialize distributed process group"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    dist.init_process_group(
        backend="gloo", 
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300)
    )
    print(f"Rank {rank}: Distributed process group initialized")


def distributed_example(args):
    """Distributed dataset processing logic"""
    rank = args.rank
    world_size = args.world_size
    
    # Initialize distributed environment
    init_distributed_process(rank, world_size, args.master_addr, args.master_port)
    
    print(f"=== Rank {rank}: Starting distributed {args.mode} mode processing ===")
    
    # Create distributed dataset based on mode
    if args.mode == "local":
        dataset = CorgiPileDistributedLocalDataset(
            data_dir=args.data_dir,
            block_size=args.block_size,
            load_data_fn=cifar_distributed_local_loader,
            file_filter_fn=cifar_file_filter,
            log_dir=args.log_dir,
            rank=rank,
            world_size=world_size,
            transform=None
        )
    elif args.mode == "hdfs":
        dataset = CorgiPileDistributedHDFSDataset(
            hdfs_root=args.hdfs_root,
            hdfs_host=args.hdfs_host,
            hdfs_port=args.hdfs_port,
            hdfs_user=args.hdfs_user,
            block_size=args.block_size,
            load_data_fn=cifar_distributed_hdfs_loader,
            file_filter_fn=cifar_file_filter,
            shuffle=True,
            log_dir=args.log_dir,
            rank=rank,
            world_size=world_size
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Data loader (multi-worker support)
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=args.num_workers,
        pin_memory=True,
        multiprocessing_context='spawn'
    )
    
    # Demonstrate data processing
    start_time = time.time()
    batch_count = 0
    for batch_idx, (data, label, source_info) in enumerate(loader):
        if batch_idx < 5:  # Print only first 5 batches
            print(f"Rank {rank}: Batch {batch_idx}, Number of samples: {len(data)}, Source file IDs: {[s[0] for s in source_info[:3]]}...")
        batch_count += 1
    
    total_time = time.time() - start_time
    print(f"Rank {rank}: Processing completed, total {batch_count} batches, time elapsed: {total_time:.2f} seconds")
    
    # Clean up distributed environment
    dist.destroy_process_group()


def cifar_distributed_local_loader(file_path: str, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """CIFAR data loading function for local file system"""
    file_id = kwargs.get('file_id', 0)
    
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        imgs = data_dict[b'data']
        labels = data_dict[b'labels']
        
        for inner_idx, (img, label) in enumerate(zip(imgs, labels)):
            # Convert to (H, W, C) format
            img_r = img[0:1024].reshape(32, 32)
            img_g = img[1024:2048].reshape(32, 32)
            img_b = img[2048:3072].reshape(32, 32)
            img = np.dstack((img_r, img_g, img_b))
            
            yield (img, label, (file_id, inner_idx))
            
    except Exception as e:
        raise RuntimeError(f"Failed to load file {file_path}: {e}")


def cifar_distributed_hdfs_loader(file_path: str, hdfs: pyarrow.fs.FileSystem,** kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """CIFAR data loading function for HDFS"""
    file_id = kwargs.get('global_file_id', 0)
    
    try:
        with hdfs.open_input_file(file_path) as f:
            file_content = f.read()
        
        data_dict = pickle.load(BytesIO(file_content), encoding='bytes')
        imgs = data_dict[b'data']
        labels = data_dict[b'labels']
        
        for inner_idx, (img, label) in enumerate(zip(imgs, labels)):
            img_r = img[0:1024].reshape(32, 32)
            img_g = img[1024:2048].reshape(32, 32)
            img_b = img[2048:3072].reshape(32, 32)
            img = np.dstack((img_r, img_g, img_b))
            
            yield (img, label, (file_id, inner_idx))
            
    except Exception as e:
        raise RuntimeError(f"Failed to load HDFS file {file_path}: {e}")


def cifar_file_filter(file_path: str) -> bool:
    """Filter CIFAR-10 batch files"""
    file_name = os.path.basename(file_path)
    return file_name.startswith('data_batch_') or file_name == 'test_batch'


def main():
    parser = argparse.ArgumentParser(description="Distributed dataset example")
    
    # Required parameter: current machine rank
    parser.add_argument("--rank", type=int, required=True, help="Current Rank (0-based)")
    
    # Distributed parameters
    parser.add_argument("--mode", choices=["local", "hdfs"], default="local", help="Storage backend: local filesystem or HDFS")
    parser.add_argument("--world-size", type=int, default=2, help="Total number of processes")
    parser.add_argument("--master-addr", default="master", help="Master Address")
    parser.add_argument("--master-port", default="23456", help="Master Port")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of DataLoader workers")


    # Local filesystem parameters
    parser.add_argument("--data-dir", default="/home/yuran/data/cifar-10-batches-py", help="Path to CIFAR-10 data directory (local mode)")
    
    # HDFS parameters
    parser.add_argument("--hdfs-root", default="/user/yuran/datasets/cifar-10-batches-py", help="HDFS root path (hdfs mode)")
    parser.add_argument("--hdfs-host", default="master", help="HDFS namenode host")
    parser.add_argument("--hdfs-port", type=int, default=9000, help="HDFS namenode port")
    parser.add_argument("--hdfs-user", default="yuran", help="HDFS username")
    
    # Dataset parameters
    parser.add_argument("--block-size", type=int, default=100, help="Block size")
    parser.add_argument("--log-dir", default="./distributed_logs_local", help="Log Directory")
    # parser.add_argument("--log-dir", default="./distributed_logs_hdfs", help="Log Directory")
    
    args = parser.parse_args()
    distributed_example(args)


if __name__ == "__main__":
    main()
