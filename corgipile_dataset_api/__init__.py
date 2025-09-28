"""
CorgiPile Dataset API - Distributed Block-level Dataset for Machine Learning

A high-performance, distributed dataset loading library for PyTorch that supports:
- Block-level data processing with dual-layer shuffle
- Single machine and distributed training modes  
- Local filesystem and HDFS storage backends
- Sample traceability and logging
"""

from corgipile_dataset_api import local
from corgipile_dataset_api import hdfs
