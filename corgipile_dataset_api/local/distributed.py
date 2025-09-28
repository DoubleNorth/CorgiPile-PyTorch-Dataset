from torch.utils.data import Dataset
import os
import random
from typing import Iterable, Tuple, Any, Callable, Optional
from filelock import FileLock
import torch

class CorgiPileDistributedLocalDataset(Dataset):
    """
    Distributed dataset for multi-machine training with file-level data partitioning.
    
    Each machine processes a subset of files based on its rank to ensure no data overlap 
    across machines, while maintaining consistent global file ID mapping.
    
    Args:
        data_dir (str): Root directory containing training data files
        block_size (int): Number of samples per data block
        load_data_fn (Callable): Custom function to load data from file paths
        file_filter_fn (Optional[Callable]): Function to filter valid data files (default: accept all)
        rank (int): Rank of current machine (0-based index). Default: 0
        world_size (int): Total number of machines in distributed cluster. Default: 1
        transform (Optional[Callable]): Data preprocessing/augmentation function. Default: None
        log_dir (Optional[str]): Directory path for saving sample logs. Default: None
        **kwargs: Additional parameters passed to load_data_fn
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
        log_dir: Optional[str] = None,
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
        
        # Logging initialization
        self.log_dir = log_dir
        self._init_logging()

        # Get all valid files (consistent order across all machines)
        self.all_data_files = self._list_files()
        self.all_file_count = len(self.all_data_files)
        self.global_file_id_mapping = {f: idx for idx, f in enumerate(self.all_data_files)}
        
        print(f"Total files discovered: {self.all_file_count}")

        # Assign files to current machine based on rank
        self.assigned_files = self._assign_files_to_rank()
        self.assigned_file_count = len(self.assigned_files)
        self.assigned_global_ids = [self.global_file_id_mapping[f] for f in self.assigned_files]
        
        print(f"Rank {rank} assigned {self.assigned_file_count} files (IDs: {self.assigned_global_ids})")

        # Load samples from assigned files
        self.samples = self._load_assigned_samples()
        self.total_samples = len(self.samples)
        
        print(f"Rank {rank} loaded {self.total_samples} samples")

        # Create block index ranges
        self.total_blocks = (self.total_samples + block_size - 1) // block_size
        self.block_indices = self._create_block_indices()
        
        print(f"Rank {rank} created {self.total_blocks} blocks")

    def _init_logging(self):
        """Initialize log directories and file lock mechanism"""
        if self.log_dir is None:
            self.logging_enabled = False
            return
            
        self.logging_enabled = True
        
        # Create rank-specific log directory (e.g., node0, node1)
        self.node_log_dir = os.path.join(self.log_dir, f"node{self.rank}")
        os.makedirs(self.node_log_dir, exist_ok=True)
        
        # Log file templates (separated by worker ID)
        self.log_file_template = os.path.join(self.node_log_dir, "worker_{worker_id}_samples.txt")
        self.lock_file_template = os.path.join(self.node_log_dir, "worker_{worker_id}_samples.txt.lock")

    def _log_sample(self, worker_id: int, sample_info: Tuple[int, int]):
        """
        Log sample metadata to file
        
        Args:
            worker_id: ID of the DataLoader worker process
            sample_info: Sample source metadata (file_id, inner_index)
        """
        if not self.logging_enabled:
            return
            
        # Get worker-specific log and lock files
        log_file = self.log_file_template.format(worker_id=worker_id)
        lock_file = self.lock_file_template.format(worker_id=worker_id)
        
        # Use file lock to ensure thread-safe writing
        with FileLock(lock_file):
            with open(log_file, "a") as f:
                f.write(f"{sample_info[0]}-{sample_info[1]}\n")

    def _list_files(self) -> list:
        """List all valid data files with consistent ordering across machines"""
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                    if os.path.isfile(os.path.join(self.data_dir, f))]
        filtered_files = [f for f in all_files if self.file_filter_fn(f)]
        return sorted(filtered_files)  # Ensure same file order on all machines

    def _assign_files_to_rank(self) -> list:
        """Assign files to current rank using modulo-based partitioning"""
        if self.world_size <= 1:
            return self.all_data_files
        
        return [
            file for idx, file in enumerate(self.all_data_files)
            if idx % self.world_size == self.rank
        ]

    def _load_assigned_samples(self) -> list:
        """Load samples from files assigned to current machine"""
        samples = []
        
        for file_path in self.assigned_files:
            global_file_id = self.global_file_id_mapping[file_path]
            for item in self.load_data_fn(file_path, **{**self.kwargs, 'file_id': global_file_id}):
                samples.append(item)
        return samples

    def _create_block_indices(self) -> list:
        """Create index ranges for each data block"""
        return [
            (i * self.block_size, min((i + 1) * self.block_size, self.total_samples))
            for i in range(self.total_blocks)
        ]

    def get_block_sample_indices(self, block_id: int, shuffle: bool = True) -> list:
        """Get sample indices for a specific block with optional shuffling"""
        if block_id < 0 or block_id >= self.total_blocks:
            raise ValueError(f"Block ID {block_id} out of range (max: {self.total_blocks-1})")
            
        start, end = self.block_indices[block_id]
        indices = list(range(start, end))
        
        if shuffle:
            # Use rank-specific seed for consistent shuffling
            random.seed(100 + block_id + self.rank * 1000)
            random.shuffle(indices)
            
        return indices

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Tuple[int, int]]:
        """Get sample by index and log metadata if enabled"""
        data, label, source_info = self.samples[idx]
        
        # Get current worker ID
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        # Log sample metadata
        self._log_sample(worker_id, source_info)
        
        if self.transform:
            data = self.transform(data)
            
        return data, label, source_info

    def __len__(self) -> int:
        return self.total_samples
