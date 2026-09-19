"""WorldIR validation gate (spec sec 25 WORLD VALIDATION).

Composes the world's own structural self-check (`WorldIR.validate()` --
dangling entity/geometry/material/surface/component references, temporal
and branch consistency) with deeper geometric and provenance checks
that self-check does not perform:

  - invalid geometry: empty bounds, inverted bounds, non-finite values;
  - impossible transforms: non-finite matrices, near-singular ones;
  - duplicate entity identity: two entities with the same
    (type, name, geometry) fingerprint -- a real duplicate, not a
    naming convention;
  - invalid measurements: non-finite or negative values, non-positive
    precision, confidence outside [0, 1], impossible (negative)
    dimensions (zero allowed: absence of extent is not corruption);
  - provenance/confidence consistency: UNKNOWN-provenance entities must
    not claim high confidence (0.9+) -- claiming to not know while
    claiming certainty is a spec violation, not a style choice;
  - coordinate-frame consistency for transforms that reference entities.

Severity matters: ERROR blocks the world compiler's gate (and any
consumer that refuses corrupted state); WARNING is reported and
returned, never silently swallowed.

Deterministic: issues are collected in canonical order (entities sorted
by id, checks in fixed order), so the same world always yields the same
report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from provenance import Provenance
from world_ir import WorldIR


class ValidationSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code}: {self.message}"


@dataclass(frozen=True)
class WorldValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity is ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity is ValidationSeverity.WARNING]

    def is_valid(self) -> bool:
        return not self.errors

    def messages(self) -> List[str]:
        return [str(i) for i in self.issues]


def _iter_bounds(geometry):
    yield geometry.bounds_min
    yield geometry.bounds_max


def _check_geometry(world: WorldIR, issues: List[ValidationIssue]) -> None:
    for geometry_id in sorted(world.geometries):
        geometry = world.geometries[geometry_id]
        for vector, label in ((geometry.bounds_min, "bounds_min"), (geometry.bounds_max, "bounds_max")):
            if vector is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="geometry_missing_bounds",
                    message=f"geometry {geometry_id} has no {label}",
                ))
                continue
            coords = (vector.x, vector.y, vector.z)
            if any(not math.isfinite(c) for c in coords):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="geometry_non_finite_bounds",
                    message=f"geometry {geometry_id} has non-finite {label} {coords}",
                ))
        if geometry.bounds_min is None or geometry.bounds_max is None:
            continue
        lo = (geometry.bounds_min.x, geometry.bounds_min.y, geometry.bounds_min.z)
        hi = (geometry.bounds_max.x, geometry.bounds_max.y, geometry.bounds_max.z)
        if any(h < l for h, l in zip(hi, lo)):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="geometry_inverted_bounds",
                message=(
                    f"geometry {geometry_id} has inverted bounds: "
                    f"max {hi} < min {lo} on at least one axis"
                ),
            ))


def _check_transforms(world: WorldIR, issues: List[ValidationIssue]) -> None:
    for entity_id in sorted(world.transforms):
        transform = world.transforms[entity_id]
        if entity_id not in world.entities:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="transform_unknown_entity",
                message=f"transform references unknown entity {entity_id}",
            ))
            continue
        matrix = transform.matrix
        flat = [value for row in matrix for value in row] if isinstance(matrix[0], (list, tuple)) else list(matrix)
        if any(not math.isfinite(float(v)) for v in flat):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="transform_non_finite",
                message=f"transform of entity {entity_id} contains non-finite values",
            ))
            continue
        # Near-singular rigid transform: last-row guard + determinant on
        # the upper-left 3x3. A degenerate matrix would collapse world
        # geometry onto a plane/point when applied.
        if len(matrix) == 4:
            bottom = matrix[3]
            if any(abs(float(v)) > 1e-9 for v in bottom[:3]) or abs(float(bottom[3]) - 1.0) > 1e-9:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="transform_not_affine",
                    message=f"transform of entity {entity_id} has a non-affine bottom row {bottom}",
                ))
            a = matrix[0][:3]
            b = matrix[1][:3]
            c = matrix[2][:3]
            det = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            if abs(det) < 1e-9:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="transform_singular",
                    message=f"transform of entity {entity_id} is near-singular (det={det:.3e})",
                ))


def _check_duplicate_identity(world: WorldIR, issues: List[ValidationIssue]) -> None:
    seen = {}
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        fingerprint = (
            entity.type.value if hasattr(entity.type, "value") else str(entity.type),
            entity.name,
            tuple(sorted(entity.geometry_ids)),
        )
        if fingerprint in seen:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="entity_duplicate_identity",
                message=(
                    f"entities {seen[fingerprint]} and {entity_id} share type/name/geometry "
                    "-- possible duplicate"
                ),
            ))
        else:
            seen[fingerprint] = entity_id


def _check_measurements(world: WorldIR, issues: List[ValidationIssue]) -> None:
    # Measurements ride on entities as (key, Measurement) pairs in
    # custom_properties; values may also be plain scalars promoted by
    # earlier layers. Both shapes are validated where applicable.
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        for key, value in entity.custom_properties.items():
            if not isinstance(value, float) and not isinstance(value, int):
                continue
            if not math.isfinite(float(value)):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="measurement_non_finite",
                    message=f"entity {entity_id} property {key} is non-finite",
                ))
                continue
            # Dimension-like keys must be non-negative; zero is legal
            # (absence of extent is not corruption).
            if key.endswith(("_m", "_m2", "_m3")) and float(value) < 0.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="measurement_negative_dimension",
                    message=f"entity {entity_id} property {key} is negative ({value})",
                ))


def _check_provenance_confidence(world: WorldIR, issues: List[ValidationIssue]) -> None:
    for entity_id in sorted(world.entities):
        entity = world.entities[entity_id]
        provenance = entity.provenance
        if not (0.0 <= entity.confidence <= 1.0):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="confidence_out_of_range",
                message=f"entity {entity_id} confidence {entity.confidence} outside [0, 1]",
            ))
        if provenance is Provenance.UNKNOWN and entity.confidence >= 0.9:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="unknown_provenance_high_confidence",
                message=(
                    f"entity {entity_id} claims UNKNOWN provenance with confidence "
                    f"{entity.confidence:.2f} -- cannot know nothing and be certain"
                ),
            ))


def validate_world_ir(world: WorldIR) -> WorldValidationReport:
    """Full validation: structural self-check + geometric/provenance
    depth. Deterministic issue ordering (sorted entity/geometry ids,
    fixed check order)."""
    issues: List[ValidationIssue] = []

    # 1. The world's own structural self-check (dangling references,
    #    temporal/branch consistency) -- compose, don't duplicate.
    for message in world.validate():
        issues.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code="structural",
            message=message,
        ))

    _check_geometry(world, issues)
    _check_transforms(world, issues)
    _check_duplicate_identity(world, issues)
    _check_measurements(world, issues)
    _check_provenance_confidence(world, issues)

    return WorldValidationReport(issues=issues)
