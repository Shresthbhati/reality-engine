"""Artifact lineage: which stage produced an artifact and which stages consumed it (RE-ARCH-000014).

Pure data, no I/O. Built by the runner as stages finish; query it to find what a changed artifact invalidates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LineageGraph:
    producer: dict[str, str] = field(default_factory=dict)          # artifact name -> stage that wrote it
    reads: dict[str, tuple[str, ...]] = field(default_factory=dict)  # stage -> artifact names it read
    order: list[str] = field(default_factory=list)                  # stage names, run order

    def add(self, stage: str, inputs, outputs) -> None:
        self.order.append(stage)
        self.reads[stage] = tuple(inputs)
        for name in outputs:
            self.producer[name] = stage  # a later stage re-writing a name takes over as producer

    def consumers(self, artifact: str) -> list[str]:
        return [s for s in self.order if artifact in self.reads[s]]

    def invalidated_by(self, artifact: str) -> list[str]:
        """Stages that must rerun if `artifact` changes, in run order. Reverse-direction BFS: the
        consumers of the artifact, then the consumers of everything those stages wrote, and so on."""
        writes: dict[str, list[str]] = {}
        for name, stage in self.producer.items():
            writes.setdefault(stage, []).append(name)
        hit: set[str] = set()
        queue = deque([artifact])
        while queue:
            for stage in self.consumers(queue.popleft()):
                if stage not in hit:
                    hit.add(stage)
                    queue.extend(writes.get(stage, ()))
        return [s for s in self.order if s in hit]

    def to_dict(self) -> dict:
        return {"producer": dict(self.producer), "reads": {k: list(v) for k, v in self.reads.items()}, "order": list(self.order)}
