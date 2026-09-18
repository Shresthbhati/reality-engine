"""Incremental compilation (P13): new evidence updates only the
AFFECTED world -- never a full rebuild.

Model:
- a pipeline is an ordered list of stages; each stage consumes named
  inputs (upstream stage outputs or external sources) and produces a
  content-addressed output digest;
- a stage's cache key is (stage name, sorted dependency digests) --
  deterministic, so the same inputs always hit the same entry;
- on compile, each stage's key is looked up: hit -> the cached output
  is REUSED (the stage's compute is not called); miss -> the stage
  RECOMPUTES, and its new output changes the keys of every downstream
  dependent (invalidation propagates through the dependency graph,
  and only through it);
- CompileResult records `recomputed` (in execution order) and
  `reused`, plus every stage's final output digest -- the audit trail
  that says exactly which parts of the world new evidence touched.

Caches: InMemoryCache (session) and FileCache (persistent across
process restarts, content-addressed under a root directory). The
FileCache is what makes the restart property hold: a fresh compiler
over the same root reuses prior work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class _Cache:
    def get(self, key: str) -> Optional[bytes]:
        raise NotImplementedError

    def put(self, key: str, value: bytes) -> None:
        raise NotImplementedError


class InMemoryCache(_Cache):
    def __init__(self):
        self._entries: Dict[str, bytes] = {}

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value


class FileCache(_Cache):
    """Persistent, content-addressed cache (survives restarts)."""

    def __init__(self, root):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.cache"

    def get(self, key):
        path = self._path(key)
        if not path.exists():
            return None
        return path.read_bytes()

    def put(self, key, value):
        self._path(key).write_bytes(value)


@dataclass
class CompileResult:
    recomputed: List[str] = field(default_factory=list)
    reused: List[str] = field(default_factory=list)
    outputs: Dict[str, bytes] = field(default_factory=dict)


def _key_for(stage_name: str, dep_digests: List[Tuple[str, bytes]]) -> str:
    h = hashlib.sha256()
    h.update(stage_name.encode())
    for name, digest in sorted(dep_digests):
        h.update(name.encode())
        h.update(digest)
    return h.hexdigest()


class IncrementalCompiler:
    def __init__(self, cache: _Cache):
        self._cache = cache

    def compile(
        self,
        *,
        stages: Sequence,
        sources: Dict[str, bytes],
    ) -> CompileResult:
        """`stages` are objects with .name, .deps (input names: other
        stage names or source names), and .compute(inputs: dict[name,
        bytes]) -> bytes. Source inputs come from `sources` (new
        evidence lands here). Execution order = the given list order;
        every stage's deps must precede it or be a source."""
        result = CompileResult()
        outputs: Dict[str, bytes] = dict(sources)

        for stage in stages:
            missing = [d for d in stage.deps if d not in outputs]
            if missing:
                raise ValueError(
                    f"stage {stage.name!r} depends on {missing!r} which "
                    "is neither an earlier stage nor a provided source"
                )
            dep_digests = [(d, outputs[d]) for d in stage.deps]
            key = _key_for(stage.name, dep_digests)
            cached = self._cache.get(key)
            if cached is not None:
                outputs[stage.name] = cached
                result.reused.append(stage.name)
            else:
                fresh = stage.compute(
                    {d: outputs[d] for d in stage.deps}
                )
                self._cache.put(key, fresh)
                outputs[stage.name] = fresh
                result.recomputed.append(stage.name)

        result.outputs = outputs
        return result
