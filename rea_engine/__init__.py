"""REA core engine: behavior extraction, synthetic datasets, logic validation."""

from .dataset_gen import (
    ActivationRecord,
    BehaviorDataset,
    ProbeExample,
    SyntheticDatasetGenerator,
)
from .extractor import BehaviorExtractor, TargetModel
from .validator import LogicValidator, ValidationReport

__all__ = [
    "ActivationRecord",
    "BehaviorDataset",
    "ProbeExample",
    "SyntheticDatasetGenerator",
    "BehaviorExtractor",
    "TargetModel",
    "LogicValidator",
    "ValidationReport",
]
