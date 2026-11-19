"""Multi-engine reconstruction foundation: registry, adapter contract, execution manager, evidence store.

Design rule shared by every module here: a missing fact stays missing. An engine that is not verifiably available is
reported UNAVAILABLE with the reason; a confidence nobody measured is ``None``; a fallback is recorded, never hidden.
"""

from __future__ import annotations

from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    CheckpointRequirement,
    Device,
    EngineAdapter,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineRun,
    EngineRunStatus,
    EngineSpec,
    HardwareRequirement,
    ResourceUsage,
    Task,
)
from engine.multiengine.evidence_store import EvidenceStore, EvidenceStoreError
from engine.multiengine.executor import (
    ExecutionManager,
    ExecutionPlan,
    ExecutionReport,
    PlanEntry,
    Policy,
    ReportStatus,
    ResourceProbe,
    TorchResourceProbe,
)
from engine.multiengine.observation import EngineObservation, ObservationDraft, make_observation
from engine.multiengine.registry import EngineRegistry, RegistryError, default_registry
from engine.multiengine.sequential_pipeline import (
    MaximumFidelityJobReport,
    SequentialMaximumFidelityPipeline,
)
from engine.multiengine.stages import STAGE_DEFINITIONS, PipelineStage, StageId

__all__ = [
    "Availability",
    "AvailabilityStatus",
    "CheckpointRequirement",
    "Device",
    "EngineAdapter",
    "EngineError",
    "EngineErrorKind",
    "EngineRequest",
    "EngineRun",
    "EngineRunStatus",
    "EngineSpec",
    "HardwareRequirement",
    "ResourceUsage",
    "Task",
    "EvidenceStore",
    "EvidenceStoreError",
    "ExecutionManager",
    "ExecutionPlan",
    "ExecutionReport",
    "PlanEntry",
    "Policy",
    "ReportStatus",
    "ResourceProbe",
    "TorchResourceProbe",
    "EngineObservation",
    "ObservationDraft",
    "make_observation",
    "EngineRegistry",
    "RegistryError",
    "default_registry",
    "MaximumFidelityJobReport",
    "SequentialMaximumFidelityPipeline",
    "STAGE_DEFINITIONS",
    "PipelineStage",
    "StageId",
]
