"""
Distributed local filesystem dataset implementation.
"""

import os
import random
from typing import Callable, Iterable, Tuple, Any, Optional
from torch.utils.data import Dataset


class CorgiPileDistributedLocalDataset(Dataset):
    """
    Distributed dataset for multi-machine training with file-level distribution.
    
    Each machine processes a subset of files based on its rank, ensuring no overlap
    across machines while maintaining global file ID consistency.
    
    Args:
        data_dir (str): Root directory containing data files
        block_size (int): Number of samples per block
        load_data_fn (Callable): Function to load data from file path
        file_filter_fn (Optional[Callable]): Function to filter valid files
        rank (int): Current machine rank. Default: 0
        world_size (int): Total number of machines. Default: 1
        transform (Optional[Callable]): Data transformation function
        **kwargs: Additional arguments passed to load_data_fn
    """
    
    def __init__(
        self,
        data_dir: str,
        block_size: int,
        load_data_fn: Callable[[str, dict], Iterable[Tuple[Any, Any, Tuple[int, int]]]],
        file_filter_fn: Optional[Callable[[str], bool]] = None,
        rank: int = 0,
        world_size: int = 1,
        transform=None,
        **kwargs
    ):
        self.data_dir = data_dir
        self.block_size = block_size
        self.load_data_fn = load_data_fn
        self.file_filter_fn = file_filter_fn or (lambda x: True)
        self.transform = transform
        self.kwargs = kwargs
        self.rank = rank
        self.world_size = world_size

        # Get all files (same order across all machines)
        self.all_data_files = self._list_files()
        self.all_file_count = len(self.all_data_files)
        self.global_file_id_mapping = {f: idx for idx, f in enumerate(self.all_data_files)}
        
        print(f"Total files discovered: {self.all_file_count}")

        # Assign files to current machine
        self.assigned_files = self._assign_files_to_rank()
        self.assigned_file_count = len(self.assigned_files)
        self.assigned_global_ids = [self.global_file_id_mapping[f] for f in self.assigned_files]
        
        print(f"Rank {rank} assigned {self.assigned_file_count} files (IDs: {self.assigned_global_ids})")

        # Load samples from assigned files only
        self.samples = self._load_assigned_samples()
        self.total_samples = len(self.samples)
        
        print(f"Rank {rank} loaded {self.total_samples} samples")

        # Create block indices
        self.total_blocks = (self.total_samples + block_size - 1) // block_size
        self.block_indices = self._create_block_indices()
        
        print(f"Rank {rank} created {self.total_blocks} blocks")

    def _list_files(self) -> list:
        """List all valid files (consistent across machines)."""
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                    if os.path.isfile(os.path.join(self.data_dir, f))]
        filtered_files = [f for f in all_files if self.file_filter_fn(f)]
        return sorted(filtered_files)  # Ensure consistent ordering

    def _assign_files_to_rank(self) -> list:
        """Assign files to current rank using modulo distribution."""
        if self.world_size <= 1:
            return self.all_data_files
        
        return [
            file for idx, file in enumerate(self.all_data_files)
            if idx % self.world_size == self.rank
        ]

    def _load_assigned_samples(self) -> list:
        """Load samples from files assigned to current machine."""
        samples = []
        
        for file_path in self.assigned_files:
            global_file_id = self.global_file_id_mapping[file_path]
            for item in self.load_data_fn(file_path, **{**self.kwargs, 'file_id': global_file_id}):
                samples.append(item)
        return samples

    def _create_block_indices(self) -> list:
        """Create block index ranges."""
        return [
            (i * self.block_size, min((i + 1) * self.block_size, self.total_samples))
            for i in range(self.total_blocks)
        ]

    def get_block_sample_indices(self, block_id: int, shuffle: bool = True) -> list:
        """Get sample indices for a specific block."""
        if block_id < 0 or block_id >= self.total_blocks:
            raise ValueError(f"Block ID {block_id} out of range (max: {self.total_blocks})")
            
        start, end = self.block_indices[block_id]
        indices = list(range(start, end))
        
        if shuffle:
            # Use different seeds for different machines and blocks
            random.seed(100 + block_id + self.rank * 1000)
            random.shuffle(indices)
            
        return indices

    def __getitem__(self, idx: int) -> Tuple[Any, Any, int, Tuple[int, int]]:
        """Get sample by index."""
        data, label, source_info = self.samples[idx]
        if self.transform:
            data = self.transform(data)
        return data, label, idx, source_info

    def __len__(self) -> int:
        return self.total_samples
