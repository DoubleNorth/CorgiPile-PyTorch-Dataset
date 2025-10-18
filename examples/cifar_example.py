"""
CIFAR-10 Dataset Example

This example demonstrates how to implement custom load_data_fn for CIFAR-10
pickle files and use the CorgiPile datasets.
"""

import os
import pickle
import numpy as np
import time
import torch
import argparse
from io import BytesIO
from typing import Iterable, Tuple, Any
import pyarrow.fs
from torch.utils.data import DataLoader

from corgipile_dataset_api.local.single_machine import CorgiPileLocalDataset
from corgipile_dataset_api.hdfs.single_machine import CorgiPileHDFSDataset


def cifar_local_loader(file_path: str, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """
    Load CIFAR-10 data from local pickle files.
    
    Args:
        file_path (str): Path to CIFAR pickle file
        **kwargs: Additional arguments (worker_id, file_id, etc.)
        
    Yields:
        Tuple[image, label, (file_id, inner_idx)]: Image data, label, and source info
    """
    worker_id = kwargs.get('worker_id', 0)
    file_id = kwargs.get('file_id', 0)
    
    try:
        print(f"Worker {worker_id}: Loading {os.path.basename(file_path)}")
        
        # Read and parse pickle file
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        imgs = data_dict[b'data']  # shape: (10000, 3072)
        labels = data_dict[b'labels']
        
        print(f"Worker {worker_id}: Loaded {len(labels)} samples from file {file_id}")
        
        # Convert flat RGB data to (H, W, C) format
        for inner_idx, (img, label) in enumerate(zip(imgs, labels)):
            # Reshape from (3072,) to (32, 32, 3)
            img_r = img[0:1024].reshape(32, 32)
            img_g = img[1024:2048].reshape(32, 32)
            img_b = img[2048:3072].reshape(32, 32)
            img = np.dstack((img_r, img_g, img_b))
            
            # Return: (image, label, (file_id, inner_idx))
            yield (img, label, (file_id, inner_idx))
            
    except Exception as e:
        raise RuntimeError(f"Worker {worker_id}: Failed to load {os.path.basename(file_path)}: {e}")


def cifar_hdfs_loader(file_path: str, hdfs: pyarrow.fs.FileSystem, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """
    Load CIFAR-10 data from HDFS pickle files.
    
    Args:
        file_path (str): HDFS path to CIFAR pickle file
        hdfs (pyarrow.fs.FileSystem): HDFS filesystem instance
        **kwargs: Additional arguments (worker_id, file_id, etc.)
        
    Yields:
        Tuple[image, label, (file_id, inner_idx)]: Image data, label, and source info
    """
    worker_id = kwargs.get('worker_id', 0)
    file_id = kwargs.get('file_id', 0)
    
    try:
        file_name = os.path.basename(file_path)
        print(f"Worker {worker_id}: Loading {file_name} from HDFS (file_id: {file_id})")
        
        # Read from HDFS and parse pickle
        with hdfs.open_input_file(file_path) as f:
            file_content = f.read()
        
        data_dict = pickle.load(BytesIO(file_content), encoding='bytes')
        imgs = data_dict[b'data']
        labels = data_dict[b'labels']
        
        print(f"Worker {worker_id}: Loaded {len(labels)} samples from HDFS file {file_id}")
        
        # Convert flat RGB data to (H, W, C) format  
        for inner_idx, (img, label) in enumerate(zip(imgs, labels)):
            img_r = img[0:1024].reshape(32, 32)
            img_g = img[1024:2048].reshape(32, 32)
            img_b = img[2048:3072].reshape(32, 32)
            img = np.dstack((img_r, img_g, img_b))
            
            yield (img, label, (file_id, inner_idx))
            
    except Exception as e:
        raise RuntimeError(f"Worker {worker_id}: Failed to load HDFS file {file_name}: {e}")


def cifar_file_filter(file_path: str) -> bool:
    """
    Filter function to identify valid CIFAR-10 files.
    
    Args:
        file_path (str): File path to check
        
    Returns:
        bool: True if file is a valid CIFAR batch file
    """
    file_name = os.path.basename(file_path)
    return file_name.startswith('data_batch_') or file_name == 'test_batch'


def test_local_dataset():
    """Test LocalDataset with CIFAR-10 data."""
    # Configuration
    data_dir = "/path/to/cifar-10-batches-py"  # Update this path
    num_workers = 3
    log_dir = f"./logs/local_test_{num_workers}_workers" 
    block_size = 100
    batch_size = 32
    shuffle = True

    print("=== Testing LocalDataset ===")
    
    # Create dataset
    dataset = CorgiPileLocalDataset(
        data_dir=data_dir,
        block_size=block_size,
        load_data_fn=cifar_local_loader,
        file_filter_fn=cifar_file_filter,
        shuffle=shuffle,
        log_dir=log_dir  # Enable logging
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Test data loading
    print(f"Starting test with block_size={block_size}, shuffle={shuffle}, num_workers={num_workers}")
    start_time = time.time()
    
    for i, (images, labels) in enumerate(dataloader):
        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f"Batch {i+1}: images.shape={images.shape}, labels.shape={labels.shape}, "
                  f"label_range=[{labels.min()}, {labels.max()}]")

    elapsed = time.time() - start_time
    print(f"Test completed in {elapsed:.2f} seconds")
    print(f"Logs saved to: {os.path.abspath(log_dir)}")


def test_hdfs_dataset():
    """Test HDFSDataset with CIFAR-10 data."""
    # Configuration
    hdfs_root = "/path/to/cifar-10-batches-py"  # Update this path
    hdfs_host = "master"  # Update this
    hdfs_port = 9000
    hdfs_user = "hadoop-user"  # Update this
    log_dir = "./logs/hdfs_test"
    block_size = 100
    num_workers = 2
    batch_size = 32
    shuffle = True

    print("=== Testing HDFSDataset ===")
    
    # Create dataset
    dataset = CorgiPileHDFSDataset(
        hdfs_root=hdfs_root,
        hdfs_host=hdfs_host,
        hdfs_port=hdfs_port,
        hdfs_user=hdfs_user,
        block_size=block_size,
        load_data_fn=cifar_hdfs_loader,
        file_filter_fn=cifar_file_filter,
        shuffle=shuffle,
        log_dir=log_dir  # Enable logging
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        multiprocessing_context=torch.multiprocessing.get_context('spawn')  # Recommended for HDFS
    )

    # Test data loading
    print(f"Starting HDFS test with block_size={block_size}, shuffle={shuffle}")
    start_time = time.time()
    
    for i, (images, labels) in enumerate(dataloader):
        if (i + 1) % 100 == 0:
            print(f"Batch {i+1}: images.shape={images.shape}, labels.shape={labels.shape}")

    elapsed = time.time() - start_time
    print(f"HDFS test completed in {elapsed:.2f} seconds")
    print(f"Logs saved to: {os.path.abspath(log_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CIFAR-10 dataset loading")
    parser.add_argument("--mode", type=str, choices=["local", "hdfs"], default="local",
                      help="Test mode: local or hdfs")
    args = parser.parse_args()
    
    if args.mode == "local":
        test_local_dataset()
    else:
        test_hdfs_dataset()
