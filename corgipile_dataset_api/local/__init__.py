"""Local filesystem dataset implementations."""

from corgipile_dataset_api.local import distributed
from corgipile_dataset_api.local import single_machine

from corgipile_dataset_api.local.single_machine import CorgiPileLocalDataset
from corgipile_dataset_api.local.distributed import CorgiPileDistributedLocalDataset

__all__ = ["CorgiPileLocalDataset", "CorgiPileDistributedLocalDataset"]
