"""
Distributed HDFS dataset implementation.
"""

import os
import random
import torch
import pyarrow.fs
from torch.utils.data import IterableDataset
from filelock import FileLock
from typing import Callable, Iterable, Tuple, Any, Optional


class CorgiPileDistributedHDFSDataset(IterableDataset):
    """
    Distributed HDFS dataset for multi-machine training with file-level distribution.
    
    Each machine processes a subset of HDFS files based on its rank, ensuring no overlap
    across machines while maintaining global file ID consistency.
    
    Args:
        hdfs_root (str): HDFS root directory path
        hdfs_host (str): HDFS namenode hostname
        hdfs_port (int): HDFS namenode port
        hdfs_user (str): HDFS username
        block_size (int): Number of samples per block
        load_data_fn (Callable): Function to load data from HDFS path
        file_filter_fn (Optional[Callable]): Function to filter valid files
        shuffle (bool): Enable dual-layer shuffle. Default: True
        log_dir (Optional[str]): Directory for logging. If None, no logging
        rank (int): Current machine rank. Default: 0
        world_size (int): Total number of machines. Default: 1
        **kwargs: Additional arguments passed to load_data_fn
    """
    
    def __init__(
        self,
        hdfs_root: str,
        hdfs_host: str,
        hdfs_port: int,
        hdfs_user: str,
        block_size: int,
        load_data_fn: Callable[[str, pyarrow.fs.FileSystem], Iterable[Tuple[Any, Any, Tuple[int, int]]]],
        file_filter_fn: Optional[Callable[[str], bool]] = None,
        shuffle: bool = True,
        log_dir: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        **kwargs
    ):
        self.hdfs_root = hdfs_root
        self.hdfs_host = hdfs_host
        self.hdfs_port = hdfs_port
        self.hdfs_user = hdfs_user
        self.block_size = block_size
        self.load_data_fn = load_data_fn
        self.file_filter_fn = file_filter_fn or (lambda x: True)
        self.shuffle = shuffle
        self.log_dir = log_dir
        self.rank = rank
        self.world_size = world_size
        self.kwargs = kwargs
        
        # Initialize HDFS filesystem
        self.hdfs = pyarrow.fs.HadoopFileSystem(
            host=hdfs_host, 
            port=hdfs_port, 
            user=hdfs_user
        )
        
        # Get all HDFS files (consistent across machines)
        self.all_data_files = self._list_hdfs_files()
        self.all_file_count = len(self.all_data_files)
        self.global_file_id_mapping = {file_path: idx for idx, file_path in enumerate(self.all_data_files)}
        
        print(f"Rank {rank}: Total HDFS files discovered: {self.all_file_count}")
        
        # Assign files to current machine
        self.assigned_files = self._assign_files_to_rank()
        self.assigned_file_count = len(self.assigned_files)
        self.assigned_global_ids = [self.global_file_id_mapping[f] for f in self.assigned_files]
        
        print(f"Rank {rank}: Assigned {self.assigned_file_count} files (IDs: {self.assigned_global_ids})")
        
        # Create log directory if logging is enabled
        if self.log_dir is not None:
            os.makedirs(os.path.join(self.log_dir, f"node{rank}"), exist_ok=True)

    def _list_hdfs_files(self) -> list:
        """List all valid HDFS files and apply filter."""
        try:
            all_files = []
            for entry in self.hdfs.get_file_info(
                pyarrow.fs.FileSelector(self.hdfs_root, recursive=False)
            ):
                if entry.is_file:
                    all_files.append(entry.path)
            
            # Apply file filter if provided
            filtered_files = [path for path in all_files if self.file_filter_fn(path)]
            print(f"Rank {self.rank}: Filtered {len(filtered_files)}/{len(all_files)} files")
            
            if not filtered_files:
                raise ValueError(f"No valid files found in HDFS directory {self.hdfs_root}")
            
            return sorted(filtered_files)  # Ensure consistent ordering
        except Exception as e:
            raise RuntimeError(f"Failed to list HDFS files: {str(e)}")

    def _assign_files_to_rank(self) -> list:
        """Assign files to current rank using modulo distribution."""
        if self.world_size <= 1:
            return self.all_data_files
        
        return [
            file for idx, file in enumerate(self.all_data_files)
            if idx % self.world_size == self.rank
        ]

    def _get_worker_files(self, worker_id: int, num_workers: int) -> list:
        """Distribute assigned files across workers."""
        return [
            file for i, file in enumerate(self.assigned_files)
            if i % num_workers == worker_id
        ]

    def _create_block_iterator(self, worker_files: list, worker_id: int) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
        """Create iterator with dual-layer shuffle."""
        current_block = []
        all_blocks = []
        
        print(f"Rank {self.rank} Worker {worker_id}: Processing {len(worker_files)} HDFS files")

        for file_path in worker_files:
            file_id = self.global_file_id_mapping[file_path]
            for item in self.load_data_fn(
                file_path, 
                self.hdfs, 
                **{**self.kwargs, 
                   'worker_id': worker_id, 
                   'rank': self.rank,
                   'global_file_id': file_id
                }
            ):
                sample = (item[0], item[1])
                source_info = item[2]
                current_block.append((sample, source_info))

                # Complete block when reaching block_size
                if len(current_block) >= self.block_size:
                    if self.shuffle:
                        random.shuffle(current_block)  # Intra-block shuffle
                    all_blocks.append(current_block)
                    current_block = []

        # Handle remaining samples
        if current_block:
            if self.shuffle:
                random.shuffle(current_block)
            all_blocks.append(current_block)

        # Inter-block shuffle with rank-specific seed
        if self.shuffle and len(all_blocks) > 1:
            random.seed(42 + self.rank * 100 + worker_id)
            random.shuffle(all_blocks)
            print(f"Rank {self.rank} Worker {worker_id}: Shuffled {len(all_blocks)} blocks")

        # Yield all samples
        for block in all_blocks:
            for sample, source_info in block:
                yield (sample[0], sample[1], source_info)

    def __iter__(self):
        """Iterator with multi-worker support."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            random.seed(worker_info.seed % (2**32 - 1))
        
        # Get files assigned to this worker
        worker_files = self._get_worker_files(worker_id, num_workers)
        sample_iterator = self._create_block_iterator(worker_files, worker_id)
        
        # Optional logging with file locking
        if self.log_dir is not None:
            log_file = os.path.join(self.log_dir, f"node{self.rank}", f"worker_{worker_id}_samples.txt")
            with FileLock(log_file + ".lock"):
                with open(log_file, "w", encoding="utf-8") as f:
                    for idx, (data, label, source_info) in enumerate(sample_iterator):
                        file_id, inner_idx = source_info
                        f.write(f"{file_id}-{inner_idx}\n")
                        yield (data, label)
        else:
            for data, label, _ in sample_iterator:
                yield (data, label)
