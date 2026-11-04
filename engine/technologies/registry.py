"""Technology Registry and Availability Discovery for the 50 Open-Source Technologies.

Inspects system environment (PATH binaries, installed Python modules, GPU capability)
and provides honest, un-faked availability status, adapter resolution, and execution classification.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.technologies.catalog import (
    ALL_50_TECHNOLOGIES,
    TECHNOLOGIES_BY_ID,
    TechnologyDescriptor,
    TechnologySubsystem,
)
from engine.verification.classification import (
    CapabilityVerificationStatus,
    ExecutionClassification,
)

# Registry of verified self-contained algorithmic implementations within Reality Engine
_INTERNAL_MODULE_MAP: Dict[str, str] = {
    "teaser_plus_plus": "reconstruction.scale.teaser_solver",
    "manifold": "geometry.kernels.manifold_kernel",
    "pcl": "perception.pointcloud.pcl_core",
    "gtsam": "trajectories.optimization.factor_graph",
    "xfeat": "reconstruction.matching.xfeat",
    "dedode": "reconstruction.matching.dedode",
    "oneformer": "perception.segmentation.oneformer_backend",
    "meshoptimizer": "reconstruction.meshing.meshopt_simplifier",
    "sophus": "reconstruction.geometry.lie_algebra",
    "onnx_runtime": "perception.acceleration.onnx_quantizer",
    "whitebox_tools": "geo.terrain_analysis",
    "gdal": "geo.terrain_analysis",
    "kiss_icp": "reconstruction.registration.kiss_icp",
    "kornia": "reconstruction.features.kornia_primitives",
    "openvdb": "geometry.openvdb_sdf",
    "openmesh": "geometry.halfedge_mesh",
    "yolox": "perception.detection.backends",
}


@dataclass(frozen=True)
class TechnologyStatus:
    """Current live discovery state of an evaluated technology."""

    descriptor: TechnologyDescriptor
    available: bool
    classification: ExecutionClassification = ExecutionClassification.UNAVAILABLE
    verification_status: CapabilityVerificationStatus = CapabilityVerificationStatus.NOT_IMPLEMENTED
    binary_found: Optional[str] = None
    module_found: Optional[str] = None
    internal_module: Optional[str] = None
    notes: str = ""
    has_adapter: bool = True


class TechnologyRegistry:
    """Central registry providing discovery, status queries, and adapter resolution."""

    def __init__(self) -> None:
        self._descriptors: Dict[str, TechnologyDescriptor] = dict(TECHNOLOGIES_BY_ID)
        self._cached_status: Dict[str, TechnologyStatus] = {}

    def get_descriptor(self, tech_id: str) -> Optional[TechnologyDescriptor]:
        """Fetch descriptor by id slug."""
        return self._descriptors.get(tech_id)

    def check_technology_availability(
        self, tech_id: str, force_refresh: bool = False
    ) -> TechnologyStatus:
        """Inspect environment for binary, package, or internal algorithmic availability."""
        if not force_refresh and tech_id in self._cached_status:
            return self._cached_status[tech_id]

        desc = self._descriptors.get(tech_id)
        if desc is None:
            raise KeyError(f"Unknown technology ID: {tech_id}")

        # Check binaries
        found_bin: Optional[str] = None
        for b_name in desc.binary_names:
            resolved = shutil.which(b_name)
            if resolved:
                found_bin = resolved
                break

        # Check external python modules
        found_mod: Optional[str] = None
        for m_name in desc.python_modules:
            try:
                spec = importlib.util.find_spec(m_name)
                if spec is not None:
                    found_mod = m_name
                    break
            except (ModuleNotFoundError, ValueError, ImportError):
                continue

        # Check internal verified algorithm implementation
        internal_mod: Optional[str] = _INTERNAL_MODULE_MAP.get(tech_id)
        if internal_mod:
            try:
                spec = importlib.util.find_spec(internal_mod)
                if spec is None:
                    internal_mod = None
            except (ModuleNotFoundError, ValueError, ImportError):
                internal_mod = None

        # Determine truthful execution classification and verification status
        if found_bin:
            classification = ExecutionClassification.NATIVE_EXECUTED
            verification_status = CapabilityVerificationStatus.IMPLEMENTED
            is_available = True
        elif found_mod:
            classification = ExecutionClassification.WRAPPED_EXECUTED
            verification_status = CapabilityVerificationStatus.IMPLEMENTED
            is_available = True
        elif internal_mod:
            classification = ExecutionClassification.INTERNAL_IMPLEMENTATION
            verification_status = CapabilityVerificationStatus.SYNTHETICALLY_VERIFIED
            is_available = True
        else:
            classification = ExecutionClassification.UNAVAILABLE
            verification_status = CapabilityVerificationStatus.NOT_IMPLEMENTED
            is_available = False

        notes_parts: List[str] = []
        if is_available:
            if found_bin:
                notes_parts.append(f"Native binary located at: {found_bin}")
            elif found_mod:
                notes_parts.append(f"Python package '{found_mod}' installed")
            elif internal_mod:
                notes_parts.append(f"Internal algorithm implementation: '{internal_mod}'")
        else:
            missing_items = list(desc.binary_names) + list(desc.python_modules)
            notes_parts.append(
                f"Candidate not installed on host (requires: {', '.join(missing_items)})"
            )

        if desc.license_notes:
            notes_parts.append(f"License notice: {desc.license_notes}")

        status = TechnologyStatus(
            descriptor=desc,
            available=is_available,
            classification=classification,
            verification_status=verification_status,
            binary_found=found_bin,
            module_found=found_mod,
            internal_module=internal_mod,
            notes="; ".join(notes_parts),
            has_adapter=True,
        )
        self._cached_status[tech_id] = status
        return status

    def scan_all(self, force_refresh: bool = False) -> List[TechnologyStatus]:
        """Scan all 50 technologies and return their live discovery status."""
        results: List[TechnologyStatus] = []
        for tech in ALL_50_TECHNOLOGIES:
            results.append(self.check_technology_availability(tech.id, force_refresh=force_refresh))
        return results

    def filter_technologies(
        self,
        subsystem: Optional[TechnologySubsystem] = None,
        available_only: bool = False,
        min_priority: Optional[int] = None,
    ) -> List[TechnologyStatus]:
        """Filter technologies by subsystem, availability, and priority tier."""
        all_statuses = self.scan_all()
        filtered: List[TechnologyStatus] = []
        for s in all_statuses:
            if subsystem and s.descriptor.subsystem != subsystem:
                continue
            if available_only and not s.available:
                continue
            if min_priority and s.descriptor.evaluation_priority > min_priority:
                continue
            filtered.append(s)
        return filtered

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate high-level counts across all 7 specialist subsystems."""
        all_statuses = self.scan_all()
        by_subsystem: Dict[str, Dict[str, int]] = {}
        for s in all_statuses:
            sub = s.descriptor.subsystem.value
            if sub not in by_subsystem:
                by_subsystem[sub] = {"total": 0, "available": 0, "priority_1": 0}
            by_subsystem[sub]["total"] += 1
            if s.available:
                by_subsystem[sub]["available"] += 1
            if s.descriptor.evaluation_priority == 1:
                by_subsystem[sub]["priority_1"] += 1

        total_available = sum(1 for s in all_statuses if s.available)
        by_classification = {}
        for s in all_statuses:
            cls_name = s.classification.value
            by_classification[cls_name] = by_classification.get(cls_name, 0) + 1

        return {
            "total_technologies": len(all_statuses),
            "total_available": total_available,
            "total_missing": len(all_statuses) - total_available,
            "by_classification": by_classification,
            "subsystems": by_subsystem,
        }


# Global Singleton
_DEFAULT_REGISTRY: Optional[TechnologyRegistry] = None


def get_technology_registry() -> TechnologyRegistry:
    """Return the global TechnologyRegistry instance."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = TechnologyRegistry()
    return _DEFAULT_REGISTRY
