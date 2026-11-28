"""Structured export report shared by every WorldIR exporter (studio
campaign Phase 7: "Generate an export report ... entities exported,
entities skipped, warnings, deterministic export hash").

Before this module, `exporters/gltf/exporter.py`, `exporters/usd/exporter.py`,
and `exporters/blender/exporter.py` each silently skipped entities with
no transform or no exportable geometry -- correct behavior (documented
in their own docstrings), but invisible to a caller: nothing told you
*which* entities were skipped or *why*, and there was no way to verify
two export runs produced byte-identical output without diffing the raw
file yourself.

`ExportReport` makes that explicit and machine-checkable. Each exporter
gains an `export_..._with_report()` function alongside its existing
(unchanged, still-tested) `export_...()`/`write_..._file()` API -- this
is purely additive, nothing existing changes behavior.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple


def content_hash(content) -> str:
    """Deterministic sha256 hex digest of exported content (str or bytes).
    Same content -> same hash, always -- the "deterministic export hash"
    Phase 7 asks for, used to verify two export runs agree byte-for-byte."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ExportReport:
    format: str  # "gltf" | "usda" | "blender-script"
    world_id: str
    world_version: int
    entities_exported: Tuple[str, ...]
    entities_skipped: Tuple[str, ...]
    #: one entry per skipped entity, same order as entities_skipped --
    #: e.g. "no transform", "no BOX/PLANE geometry with real bounds".
    #: Never silently discarded (Phase 7: "never silently discard major
    #: information").
    skip_reasons: Tuple[str, ...]
    content_hash: str

    def __post_init__(self):
        if len(self.entities_skipped) != len(self.skip_reasons):
            raise ValueError(
                f"entities_skipped ({len(self.entities_skipped)}) and skip_reasons "
                f"({len(self.skip_reasons)}) must have matching length"
            )

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "world_id": self.world_id,
            "world_version": self.world_version,
            "entities_exported": list(self.entities_exported),
            "entities_skipped": list(self.entities_skipped),
            "skip_reasons": list(self.skip_reasons),
            "content_hash": self.content_hash,
        }
