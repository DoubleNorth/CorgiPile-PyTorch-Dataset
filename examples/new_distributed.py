import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import os
import time
import random
import torchvision.transforms as transforms
from datetime import timedelta
from filelock import FileLock
from collections import defaultdict
from typing import Iterable, Tuple, Any

# -------------------------- Worker Initialization Function --------------------------
def worker_init_fn(worker_id: int) -> None:
    import builtins
    
    # Get global variables
    worker_blocks = builtins.worker_block_assignments[worker_id]
    node_name = builtins.node_name
    dataset = builtins.dataset
    log_root = builtins.log_root
    worker_file_ids = builtins.worker_file_assignments.get(worker_id, [])
    
    # Create log directory
    log_dir = os.path.join(log_root, node_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Log and flag file paths
    log_path = os.path.join(log_dir, f"worker_{worker_id}_samples.txt")
    done_flag_path = os.path.join(log_dir, f"worker_{worker_id}_done.flag")
    error_flag_path = os.path.join(log_dir, f"worker_{worker_id}_error.txt")

    try:
        # Process all assigned blocks
        print(f"[{node_name} Worker{worker_id}] Starting processing {len(worker_blocks)} blocks, responsible file IDs: {worker_file_ids}")
        with FileLock(log_path + ".lock"):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"# {node_name} Worker{worker_id} processing log\n")
                f.write(f"# Responsible file IDs: {worker_file_ids}\n")
                f.write(f"# Responsible block IDs: {worker_blocks}\n")
                f.write("# Sample format: global_index,file_id,inner_index\n")
                
                # Iterate through each block and write sample information
                for block_id in worker_blocks:
                    sample_indices = dataset.get_block_sample_indices(block_id, shuffle=True)
                    for idx in sample_indices:
                        _, _, _, (file_id, inner_idx) = dataset[idx]
                        f.write(f"{idx},{file_id},{inner_idx}\n")
        
        # Create completion flag
        with open(done_flag_path, "w", encoding="utf-8") as f:
            f.write(f"Processing completed at {time.ctime()}\n")
            f.write(f"Number of blocks processed: {len(worker_blocks)}\n")
            f.write(f"Processed file IDs: {worker_file_ids}\n")
        
        print(f"[{node_name} Worker{worker_id}] Processing completed, log written to: {log_path}")

    except Exception as e:
        # Record error information
        with open(error_flag_path, "w", encoding="utf-8") as f:
            f.write(f"Error occurred at {time.ctime()}\n")
            f.write(f"Error message: {str(e)}\n")
        print(f"[{node_name} Worker{worker_id}] Error occurred: {str(e)}")
        raise e


# -------------------------- Distributed Processing Main Logic --------------------------
def distributed_process(rank: int, world_size: int, args) -> None:
    # 1. Initialize distributed environment
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180)
    )
    
    node_name = f"node{rank}"
    print(f"[{node_name}] Distributed environment initialized (rank={rank}, world_size={world_size})")

    # 2. Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 3. Instantiate dataset
    dataset = DistributedBlockCorgiDataset(
        data_dir=args.data_dir,
        block_size=args.block_size,
        load_data_fn=args.load_data_fn,
        file_filter_fn=args.file_filter_fn,
        transform=transform,
        rank=rank,
        world_size=world_size
    )

    # 4. Get all local blocks for current node
    all_local_blocks = list(range(dataset.total_blocks))
    print(f"[{node_name}] Total local blocks: {len(all_local_blocks)}")

    # 5. Group blocks by file
    block_to_file = {}
    for block_id in all_local_blocks:
        start_idx = block_id * args.block_size
        _, _, _, (file_id, _) = dataset[start_idx]
        block_to_file[block_id] = file_id
    
    # Group blocks by file ID
    file_blocks = defaultdict(list)
    for block_id, file_id in block_to_file.items():
        file_blocks[file_id].append(block_id)
    
    print(f"[{node_name}] Blocks grouped by file: {[(f, len(bs)) for f, bs in file_blocks.items()]}")

    # 6. Shuffle blocks for each file
    for file_id in file_blocks:
        random.seed(42 + rank + file_id)
        random.shuffle(file_blocks[file_id])
    
    # 7. Assign blocks to workers by file
    num_workers = args.num_workers
    worker_block_assignments = {i: [] for i in range(num_workers)}
    worker_file_assignments = {i: [] for i in range(num_workers)}
    
    # Distribute files to workers in round-robin fashion
    file_ids = list(file_blocks.keys())
    for idx, file_id in enumerate(file_ids):
        worker_id = idx % num_workers
        worker_block_assignments[worker_id].extend(file_blocks[file_id])
        worker_file_assignments[worker_id].append(file_id)
    
    # Print worker assignments
    for i in range(num_workers):
        print(f"[{node_name}] Worker{i} assigned {len(worker_block_assignments[i])} blocks, responsible file IDs: {worker_file_assignments[i]}")

    # 8. Set global variables for workers
    import builtins
    builtins.node_name = node_name
    builtins.log_root = args.log_dir
    builtins.worker_block_assignments = worker_block_assignments
    builtins.worker_file_assignments = worker_file_assignments
    builtins.dataset = dataset

    # 9. Create DataLoader to trigger worker processing
    class DummyDataset(Dataset):
        """Empty dataset used only to trigger worker initialization"""
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return (torch.tensor(0.), torch.tensor(0), 0, (0, 0))  # Placeholder data

    loader = DataLoader(
        dataset=DummyDataset(),
        batch_size=1,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    # 10. Trigger worker execution
    print(f"[{node_name}] Starting worker data processing...")
    start_time = time.time()
    for _ in loader:
        pass  # Iterate to trigger all worker initializations

    # 11. Wait for all workers to complete
    log_dir = os.path.join(args.log_dir, node_name)
    os.makedirs(log_dir, exist_ok=True)

    all_done = False
    check_interval = 2
    max_wait_time = 600
    wait_elapsed = 0

    print(f"[{node_name}] Waiting for {num_workers} workers to complete...")
    while not all_done and wait_elapsed < max_wait_time:
        done_count = sum(
            os.path.exists(os.path.join(log_dir, f"worker_{i}_done.flag"))
            for i in range(num_workers)
        )
        if done_count == num_workers:
            all_done = True
            print(f"[{node_name}] All workers completed!")
        else:
            time.sleep(check_interval)
            wait_elapsed += check_interval
            print(f"[{node_name}] Waited {wait_elapsed} seconds, {done_count}/{num_workers} workers completed...")

    # 12. Timeout/error handling and log verification
    if not all_done:
        print(f"[{node_name}] Warning: Timeout waiting! Checking error logs...")
        for worker_id in range(num_workers):
            error_flag = os.path.join(log_dir, f"worker_{worker_id}_error.txt")
            if os.path.exists(error_flag):
                with open(error_flag, "r") as f:
                    print(f"[{node_name}] Worker{worker_id} error:\n{f.read()}")

    print(f"\n[{node_name}] Log verification:")
    for worker_id in range(num_workers):
        log_path = os.path.join(log_dir, f"worker_{worker_id}_samples.txt")
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            print(f"✅ {log_path}")
        else:
            print(f"❌ {log_path}")

    # 13. Complete processing
    total_time = time.time() - start_time
    print(f"\n[{node_name}] Total time elapsed: {total_time:.2f}seconds")
    dist.destroy_process_group()


# -------------------------- User-defined Loading and Filtering Functions --------------------------
def cifar_distributed_loader(file_path: str, **kwargs) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
    """User-defined CIFAR loading function"""
    import pickle
    import numpy as np
    
    global_file_id = kwargs.get('file_id', -1)
    file_name = os.path.basename(file_path)
    
    try:
        print(f"[Loading file] {file_name} (global ID: {global_file_id})")
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        imgs = data_dict[b'data']
        labels = data_dict[b'labels']
        
        for inner_idx, (img, label) in enumerate(zip(imgs, labels)):
            # Image format conversion
            img_r = img[0:1024].reshape(32, 32)
            img_g = img[1024:2048].reshape(32, 32)
            img_b = img[2048:3072].reshape(32, 32)
            img = np.dstack((img_r, img_g, img_b))
            
            yield (img, label, (global_file_id, inner_idx))
        
    except Exception as e:
        raise RuntimeError(f"Processing file {file_name} (global ID: {global_file_id}) failed: {str(e)}")


def cifar_filter(file_path: str) -> bool:
    """User-defined CIFAR file filtering function"""
    file_name = os.path.basename(file_path)
    return file_name.startswith('data_batch_') or file_name == 'test_batch'


# -------------------------- Entry Point --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed dataset processing (each worker handles specific files)")
    
    # Distributed parameters
    parser.add_argument("--rank", type=int, required=True, help="Current node rank (starting from 0)")
    parser.add_argument("--world-size", type=int, default=2, help="Total number of nodes")
    parser.add_argument("--master-addr", default="master", help="Master node address")
    parser.add_argument("--master-port", default="23456", help="Master node port")
    
    # Dataset parameters
    parser.add_argument("--data-dir", default="/home/yuran/data/cifar-10-batches-py", help="Data storage directory")
    parser.add_argument("--block-size", type=int, default=100, help="Block size")
    parser.add_argument("--num-workers", type=int, default=3, help="Number of workers per node")
    parser.add_argument("--log-dir", default="./distributed_multithread_worker_per_file", help="Log root directory")
    args = parser.parse_args()    

    # Bind loading and filtering functions
    args.load_data_fn = cifar_distributed_loader
    args.file_filter_fn = cifar_filter

    # Start distributed processing
    distributed_process(
        rank=args.rank,
        world_size=args.world_size,
        args=args
    )