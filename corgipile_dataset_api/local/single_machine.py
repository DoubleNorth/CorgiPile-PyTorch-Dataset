"""
Single-machine local filesystem dataset implementation.
"""

import os
import random
import torch
from torch.utils.data import IterableDataset
from typing import Callable, Iterable, Tuple, Any, Optional


class CorgiPileLocalDataset(IterableDataset):
    """
    Block-level dataset for local filesystem with dual-layer shuffle.
    
    Features:
    - Block-level data processing with configurable block size
    - Dual-layer shuffle: intra-block + inter-block shuffling
    - Multi-worker support with automatic file distribution
    - Optional sample traceability logging
    
    Args:
        data_dir (str): Root directory containing data files
        block_size (int): Number of samples per block
        load_data_fn (Callable): Function to load data from file path
        file_filter_fn (Optional[Callable]): Function to filter valid files
        shuffle (bool): Enable dual-layer shuffle. Default: True
        log_dir (Optional[str]): Directory for logging. If None, no logging
        **kwargs: Additional arguments passed to load_data_fn
    """
    
    def __init__(
        self,
        data_dir: str,
        block_size: int,
        load_data_fn: Callable[[str], Iterable[Tuple[Any, Any]]],
        file_filter_fn: Optional[Callable[[str], bool]] = None,
        shuffle: bool = True,
        log_dir: Optional[str] = None,
        **kwargs
    ):
        self.data_dir = data_dir
        self.block_size = block_size
        self.load_data_fn = load_data_fn
        self.file_filter_fn = file_filter_fn
        self.shuffle = shuffle
        self.log_dir = log_dir
        self.kwargs = kwargs
        
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
        
        # Get all valid data files
        self.data_files = self._list_files()
        self.file_id_mapping = {file_path: idx for idx, file_path in enumerate(self.data_files)}
        
        print(f"Found {len(self.data_files)} files in {data_dir}")
        
        # Create log directory if logging is enabled
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _list_files(self) -> list:
        """List all valid data files in the directory."""
        try:
            all_files = []
            for filename in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, filename)
                if os.path.isfile(file_path):
                    all_files.append(file_path)
            
            # Apply file filter if provided
            if self.file_filter_fn is not None:
                filtered_files = [path for path in all_files if self.file_filter_fn(path)]
                print(f"Filtered {len(filtered_files)}/{len(all_files)} files")
            else:
                filtered_files = all_files
            
            if not filtered_files:
                raise ValueError(f"No valid files found in {self.data_dir}")
            
            return sorted(filtered_files)
        except Exception as e:
            raise RuntimeError(f"Failed to list files: {str(e)}")

    def _get_worker_files(self, worker_id: int, num_workers: int) -> list:
        """Distribute files across workers using round-robin."""
        return [file for i, file in enumerate(self.data_files) if i % num_workers == worker_id]

    def _create_block_iterator(self, worker_files: list, worker_id: int) -> Iterable[Tuple[Any, Any, Tuple[int, int]]]:
        """
        Create iterator with dual-layer shuffle:
        1. Collect all blocks
        2. Shuffle within each block (intra-block)
        3. Shuffle blocks order (inter-block)
        """
        current_block = []
        all_blocks = []
        
        print(f"Worker {worker_id} processing {len(worker_files)} files")

        for file_path in worker_files:
            file_id = self.file_id_mapping[file_path]
            for item in self.load_data_fn(file_path, **{**self.kwargs, 'worker_id': worker_id, 'file_id': file_id}):
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

        # Inter-block shuffle
        if self.shuffle and len(all_blocks) > 1:
            random.shuffle(all_blocks)
            print(f"Worker {worker_id} shuffled {len(all_blocks)} blocks")

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
        
        # Optional logging
        if self.log_dir is not None:
            log_file = os.path.join(self.log_dir, f"worker_{worker_id}_samples.txt")
            with open(log_file, "w") as f:
                for idx, (data, label, source_info) in enumerate(sample_iterator):
                    file_id, inner_idx = source_info
                    f.write(f"{file_id}-{inner_idx}\n")
                    yield (data, label)
        else:
            for data, label, _ in sample_iterator:
                yield (data, label)
