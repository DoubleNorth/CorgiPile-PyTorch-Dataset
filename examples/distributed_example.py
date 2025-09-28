"""
Distributed Training Example

This example shows how to use distributed datasets for multi-machine training.
Supports both local filesystem and HDFS storage backends.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    """Initialize distributed training process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    
    dist.init_process_group(
        backend="gloo", 
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300)
    )
    print(f"Rank {rank}: Distributed process group initialized")


def distributed_example(rank: int, world_size: int, args):
    """Distributed dataset processing example."""
    
    # Initialize distributed process group
    init_distributed_process(rank, world_size, args.master_addr, args.master_port)
    
    print(f"=== Rank {rank}: Starting distributed {args.mode} processing ===")
    
    # Create distributed dataset based on mode
    if args.mode == "local":
        dataset = CorgiPileDistributedLocalDataset(
            data_dir=args.data_dir,
            block_size=args.block_size,
            load_data_fn=cifar_distributed_local_loader,
            file_filter_fn=cifar_file_filter,
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
    
    print(f"Rank {rank}: Dataset created with {len(dataset)} samples")
    
    # Process data blocks
    num_blocks = dataset.total_blocks
    print(f"Rank {rank}: Processing {num_blocks} blocks")
    
    # Demonstrate block-based processing
    for block_id in range(min(3, num_blocks)):  # Process first 3 blocks as demo
        sample_indices = dataset.get_block_sample_indices(block_id, shuffle=True)
        print(f"Rank {rank}: Block {block_id} has {len(sample_indices)} samples")
        
        # Process a few samples from this block  
        for idx in sample_indices[:2]:  # Process 2 samples per block
            data, label, sample_idx, source_info = dataset[idx]
            file_id, inner_idx = source_info
            print(f"Rank {rank}: Processed sample {sample_idx} from file {file_id}")
    
    print(f"Rank {rank}: Processing completed")
    
    # Clean up
    dist.destroy_process_group()




def cifar_distributed_local_loader(file_path: str, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """CIFAR loader for distributed local filesystem."""
    file_id = kwargs.get('file_id', 0)
    
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        imgs = data_dict[b'data']
        labels = data_dict[b'labels']
        
        for inner_idx, (img, label) in enumerate(zip(imgs, labels)):
            # Convert flat RGB data to (H, W, C) format
            img_r = img[0:1024].reshape(32, 32)
            img_g = img[1024:2048].reshape(32, 32)
            img_b = img[2048:3072].reshape(32, 32)
            img = np.dstack((img_r, img_g, img_b))
            
            yield (img, label, (file_id, inner_idx))
            
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")


def cifar_distributed_hdfs_loader(file_path: str, hdfs: pyarrow.fs.FileSystem, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """CIFAR loader for distributed HDFS."""
    file_id = kwargs.get('file_id', 0)
    
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
    """Filter CIFAR-10 batch files."""
    file_name = os.path.basename(file_path)
    return file_name.startswith('data_batch_') or file_name == 'test_batch'


def main():
    parser = argparse.ArgumentParser(description="Distributed dataset example")
    
    # Storage mode
    parser.add_argument("--mode", choices=["local", "hdfs"], default="local",
                       help="Storage backend: local filesystem or HDFS")
    
    # Distributed training arguments
    parser.add_argument("--world-size", type=int, default=2, help="Total number of processes")
    parser.add_argument("--master-addr", default="master", help="Master node address")
    parser.add_argument("--master-port", default="12355", help="Master node port")
    
    # # Local filesystem arguments
    # parser.add_argument("--data-dir", default="/path/to/cifar-10-batches-py", 
    #                    help="Path to CIFAR-10 data directory (local mode)")
    
    # # HDFS arguments
    # parser.add_argument("--hdfs-root", default="/user/data/cifar-10-batches-py",
    #                    help="HDFS root path (hdfs mode)")
    # parser.add_argument("--hdfs-host", default="namenode", help="HDFS namenode host")
    # parser.add_argument("--hdfs-port", type=int, default=9000, help="HDFS namenode port")
    # parser.add_argument("--hdfs-user", default="hadoop", help="HDFS username")
    
    # # Dataset arguments
    # parser.add_argument("--block-size", type=int, default=100, help="Block size")
    # parser.add_argument("--log-dir", default="./distributed_logs", help="Log directory")

    # Local filesystem arguments
    parser.add_argument("--data-dir", default="/home/yuran/data/cifar-10-batches-py", 
                       help="Path to CIFAR-10 data directory (local mode)")
    
    # HDFS arguments
    parser.add_argument("--hdfs-root", default="/user/yuran/datasets/cifar-10-batches-py",
                       help="HDFS root path (hdfs mode)")
    parser.add_argument("--hdfs-host", default="master", help="HDFS namenode host")
    parser.add_argument("--hdfs-port", type=int, default=9000, help="HDFS namenode port")
    parser.add_argument("--hdfs-user", default="yuran", help="HDFS username")
    
    # Dataset arguments
    parser.add_argument("--block-size", type=int, default=100, help="Block size")
    parser.add_argument("--log-dir", default="./distributed_logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Print configuration
    print(f"=== Distributed {args.mode.upper()} Dataset Example ===")
    if args.mode == "local":
        print(f"Data directory: {args.data_dir}")
    else:
        print(f"HDFS root: {args.hdfs_root}")
        print(f"HDFS namenode: {args.hdfs_host}:{args.hdfs_port}")
        print(f"HDFS user: {args.hdfs_user}")
    
    print(f"Block size: {args.block_size}")
    print(f"World size: {args.world_size}")
    print(f"Log directory: {args.log_dir}")
    
    print("\n--- Running distributed processing example ---")
    print("Usage for real multi-machine deployment:")
    print(f"Machine 0: python distributed_example.py --mode {args.mode}")
    print(f"Machine 1: python distributed_example.py --mode {args.mode} --master-addr <machine-0-ip>")
    print("\nFor local testing, spawning multiple processes:")
    
    mp.spawn(
        distributed_example,
        args=(args.world_size, args),
        nprocs=args.world_size,
        join=True
    )


if __name__ == "__main__":
    main()
