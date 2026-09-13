"""Reality Engine public SDK (platformization campaign, Phases 1/2/80).

Before this module, every capability that exists (world compilation,
validation, diffing, physics compilation, export) was reachable only by
importing internal modules directly (`evidence.promote_planes`,
`engine.compiler.world_compiler`, `exporters.gltf.exporter`, ...). There
was no single stable surface an external application could depend on
without reaching into implementation details that are free to change.

This module is that surface: a thin, deliberately small set of
functions that call the real engine underneath -- no separate fake
runtime, no reimplementation. Every function here is a direct pass-
through to the module that actually does the work; this file adds
nothing except a stable name and a stable import path.

Scope (Phase 1 "smallest stable public core" -- grows only when a real
external need justifies it, not speculatively):
  - compile_world_from_reconstruction: evidence -> WorldIR
  - validate: WorldIR -> structured validation report
  - diff: WorldIR x WorldIR -> structured WorldDiff
  - compile_physics: WorldIR -> physics bodies + diagnostics
  - export: WorldIR -> (content, ExportReport) for gltf/usd/blender

What this SDK explicitly does NOT do (named so it isn't mistaken for an
oversight): it does not add authentication, job queuing, a service
layer, or a plugin system -- those are separate, much larger platform
campaign phases (24, 11-13) this module makes no attempt to cover.
"""

from __future__ import annotations

from typing import Optional, Tuple

from engine.compiler.physics_compiler import PhysicsCompileDiagnostics, compile_physics_world
from engine.compiler.world_compiler import (
    CompileDiagnostics,
    CompileInputError,
    CompileOptions,
    WorldValidationGateError,
    compile_reconstruction_to_world,
)
from exporters.blender.exporter import export_to_blender_script_with_report
from exporters.gltf.exporter import export_to_gltf_with_report
from exporters.usd.exporter import export_to_usda_with_report
from world_ir.diff import WorldDiff, diff_worlds
from world_ir.validation import WorldValidationReport, validate_world_ir
from world_ir.world_v1 import WorldIR

__all__ = [
    "CompileInputError",
    "CompileOptions",
    "UnsupportedExportFormatError",
    "WorldValidationGateError",
    "compile_physics",
    "compile_world_from_reconstruction",
    "diff",
    "export",
    "validate",
]

#: format name -> the exporter's _with_report function. Every entry here
#: is a real, tested exporter (exporters/gltf, exporters/usd,
#: exporters/blender) -- adding an entry means the format is actually
#: implemented, never a placeholder.
_EXPORTERS = {
    "gltf": export_to_gltf_with_report,
    "usda": export_to_usda_with_report,
    "blender": export_to_blender_script_with_report,
}


class UnsupportedExportFormatError(ValueError):
    """Raised by export() for any format name not in _EXPORTERS -- an
    explicit, typed failure (platformization campaign rule: "no generic
    success objects containing error strings"), never a silent no-op."""


def compile_world_from_reconstruction(
    reconstruction_result, options: Optional[CompileOptions] = None,
) -> Tuple[WorldIR, CompileDiagnostics]:
    """Evidence -> WorldIR. Direct pass-through to
    engine.compiler.world_compiler.compile_reconstruction_to_world();
    see that function's docstring for the full pipeline and error modes
    (CompileInputError, WorldValidationGateError)."""
    return compile_reconstruction_to_world(reconstruction_result, options)


def validate(world: WorldIR) -> WorldValidationReport:
    """WorldIR -> structured validation report. Direct pass-through to
    world_ir.validation.validate_world_ir()."""
    return validate_world_ir(world)


def diff(before: WorldIR, after: WorldIR) -> WorldDiff:
    """Deterministic structural diff between two WorldIR snapshots.
    Direct pass-through to world_ir.diff.diff_worlds()."""
    return diff_worlds(before, after)


def compile_physics(world: WorldIR) -> PhysicsCompileDiagnostics:
    """WorldIR -> physics bodies + diagnostics. Direct pass-through to
    engine.compiler.physics_compiler.compile_physics_world()."""
    return compile_physics_world(world)


def export(world: WorldIR, format: str):
    """WorldIR -> (content, ExportReport) for `format` in {"gltf", "usda",
    "blender"}. Raises UnsupportedExportFormatError for anything else --
    the SDK never silently no-ops on an unknown format."""
    exporter_fn = _EXPORTERS.get(format)
    if exporter_fn is None:
        raise UnsupportedExportFormatError(
            f"unsupported export format {format!r}; supported: {sorted(_EXPORTERS)}"
        )
    return exporter_fn(world)
