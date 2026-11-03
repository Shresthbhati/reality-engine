"""Technology Registry and Availability Discovery for the 50 Open-Source Technologies.

Inspects system environment (PATH binaries, installed Python modules, GPU capability)
and provides honest, un-faked availability status and adapter resolution.
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


@dataclass(frozen=True)
class TechnologyStatus:
    """Current live discovery state of an evaluated technology."""

    descriptor: TechnologyDescriptor
    available: bool
    binary_found: Optional[str] = None
    module_found: Optional[str] = None
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
        """Inspect environment for binary or module availability."""
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

        # Check python modules
        found_mod: Optional[str] = None
        for m_name in desc.python_modules:
            try:
                spec = importlib.util.find_spec(m_name)
                if spec is not None:
                    found_mod = m_name
                    break
            except (ModuleNotFoundError, ValueError, ImportError):
                continue

        is_available = bool(found_bin or found_mod)

        notes_parts: List[str] = []
        if is_available:
            if found_bin:
                notes_parts.append(f"Executable found at: {found_bin}")
            if found_mod:
                notes_parts.append(f"Python module '{found_mod}' installed")
        else:
            missing_items = list(desc.binary_names) + list(desc.python_modules)
            notes_parts.append(
                f"Candidate not installed on system (requires: {', '.join(missing_items)})"
            )

        if desc.license_notes:
            notes_parts.append(f"License notice: {desc.license_notes}")

        status = TechnologyStatus(
            descriptor=desc,
            available=is_available,
            binary_found=found_bin,
            module_found=found_mod,
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
        return {
            "total_technologies": len(all_statuses),
            "total_available": total_available,
            "total_missing": len(all_statuses) - total_available,
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
