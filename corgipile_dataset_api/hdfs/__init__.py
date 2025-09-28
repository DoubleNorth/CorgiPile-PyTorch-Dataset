"""HDFS dataset implementations."""

from corgipile_dataset_api.hdfs import distributed
from corgipile_dataset_api.hdfs import single_machine

from corgipile_dataset_api.hdfs.single_machine import CorgiPileHDFSDataset
from corgipile_dataset_api.hdfs.distributed import CorgiPileDistributedHDFSDataset

__all__ = ["CorgiPileHDFSDataset", "CorgiPileDistributedHDFSDataset"]
