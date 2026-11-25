"""Reproducible development / validation / test splits BY PHYSICAL GROUP, with leakage checks.

A group is a physical space (ScanNet space, MultiScan environment, 3RScan reference + rescans). All scenes of a group land in the same split, so the
same room never appears on both sides. The assignment depends only on (seed, group id, fractions): input order and run date are irrelevant.
Image-level leakage (the same file bytes in two scenes / splits) is delegated to ``engine.multiengine.benchmark_manifest.validate_manifest``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from engine.multiengine.benchmark_manifest import validate_manifest

from .manifest import SceneManifest, to_benchmark_manifest

DEFAULT_FRACTIONS = {"development": 0.5, "validation": 0.2, "test": 0.3}


def assign_splits(scenes: Sequence[SceneManifest], *, seed: int, fractions: Mapping[str, float] = DEFAULT_FRACTIONS) -> list[SceneManifest]:
    """New manifests with ``split`` set. Groups are ordered by sha256(seed:group) and cut by largest-remainder rounding, so every non-zero
    fraction gets at least one group whenever there are enough groups."""
    names = [k for k, v in fractions.items() if v > 0]
    if not names or abs(sum(fractions.values()) - 1.0) > 1e-9 or any(k not in ("development", "validation", "test") for k in fractions):
        raise ValueError("fractions must be over development/validation/test and sum to 1")
    groups = sorted({s.group_id for s in scenes}, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    n = len(groups)
    raw = {k: fractions[k] * n for k in names}
    counts = {k: int(raw[k]) for k in names}
    for k in sorted(names, key=lambda k: (-(raw[k] - counts[k]), k))[: n - sum(counts.values())]:
        counts[k] += 1
    for k in names:                                  # never leave a requested split empty when a group can be spared
        if counts[k] == 0 and n >= len(names):
            donor = max(names, key=lambda x: counts[x])
            counts[donor] -= 1
            counts[k] += 1
    assign: dict[str, str] = {}
    i = 0
    for k in ("development", "validation", "test"):
        for g in groups[i:i + counts.get(k, 0)]:
            assign[g] = k
        i += counts.get(k, 0)
    return [replace(s, split=assign[s.group_id]) for s in scenes]


def leakage_problems(scenes: Sequence[SceneManifest], root: str | Path, benchmark_id: str = "leakage-check") -> list[str]:
    """Group-level leakage + every check ``benchmark_manifest.validate_manifest`` makes (hashes, duplicate scene ids, image sharing, split enum)."""
    problems: list[str] = []
    home: dict[str, tuple[str, str]] = {}
    for s in scenes:
        if s.split == "unassigned":
            problems.append(f"scene {s.scene_id!r} has no split assigned")
        prev = home.setdefault(s.group_id, (s.scene_id, s.split))
        if prev[1] != s.split:
            problems.append(f"group leakage: group {s.group_id!r} is in split {prev[1]!r} (scene {prev[0]!r}) and {s.split!r} (scene {s.scene_id!r})")
    if not problems:
        problems += validate_manifest(to_benchmark_manifest(list(scenes), benchmark_id), root)
    return problems
