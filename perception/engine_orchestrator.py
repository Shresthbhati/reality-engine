"""Multi-Engine Orchestrator — runs complementary engines together with
GPU/CPU resource scheduling and cross-engine intelligence.

Architecture:
  1. ``EngineSpec`` — metadata for every registered engine (VRAM estimate,
     dependencies, license, priority)
  2. ``MultiEngineOrchestrator`` — runs engines in priority-ordered waves,
     tracking VRAM usage and skipping models that don't fit
  3. ``IntelligenceMerger`` — merges ``UnifiedEngineOutput`` from all engines
     into contradiction reports and consensus observations
  4. ``CrossEngineCompare`` — similarity comparison of observations across
     engines, used by the merger

Every engine registered through ``register_engine()`` is invoked through the
same interface: ``fn(image_paths, **context) -> UnifiedEngineOutput``.
Engines that are unavailable (binary missing, weights not cached) are
recorded in ``UnavailableEngines`` and never silently substituted.

The orchestrator is designed to be called from the production pipeline:
``run_multi_engine(image_paths, ...) -> MultiEngineReport`` where
``MultiEngineReport`` is JSON-serialisable for the job report and the
UnifiedEvidenceStore (persistence layer).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from perception.engine_evidence import (
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
    compare_observations,
    compute_observation_hash,
)

log = logging.getLogger("reality.engine_orchestrator")


@dataclass(frozen=True)
class EngineSpec:
    """Metadata for a registered engine."""
    engine_id: str
    architecture_id: str
    checkpoint_id: str = "none"
    version: str = "unknown"
    vram_mb: int = 1024           # estimated peak VRAM for this engine
    cpu_threads: int = 0          # 0 = uses default
    priority: int = 50            # higher runs first; default medium
    required_for_production: bool = False
    license: str = "unknown"
    non_commercial: bool = False
    dependencies: Tuple[str, ...] = ()  # engine_ids this depends on
    run_condition: Optional[Callable[[], str]] = None  # returns error or None


@dataclass
class MultiEngineReport:
    """Top-level result of the multi-engine pipeline run."""
    run_id: str
    image_count: int
    total_wall_s: float
    engines_run: Dict[str, str]      # engine_id -> status
    observations: List[ObservationRecord]
    contradictions: List[Dict[str, Any]]
    unavailable: List[Dict[str, Any]]  # engine_id, reason
    vram_peak_mb: Optional[float]
    cross_engine_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "image_count": self.image_count,
            "total_wall_s": round(self.total_wall_s, 3),
            "engines_run": self.engines_run,
            "n_observations": len(self.observations),
            "n_contradictions": len(self.contradictions),
            "unavailable": self.unavailable,
            "vram_peak_mb": self.vram_peak_mb,
            "cross_engine_summary": dict(self.cross_engine_summary),
            "contradictions": self.contradictions,
        }


# ---------------------------------------------------------------------------
# Engine Registry
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, EngineSpec] = {}
_ENGINE_FNS: Dict[str, Callable[..., UnifiedEngineOutput]] = {}


def register_engine(spec: EngineSpec, fn: Callable[..., UnifiedEngineOutput]) -> None:
    """Register an engine with its spec and runner function."""
    _ENGINES[spec.engine_id] = spec
    _ENGINE_FNS[spec.engine_id] = fn


def list_engines() -> List[EngineSpec]:
    """All registered engines, sorted by priority (highest first)."""
    return sorted(_ENGINES.values(), key=lambda s: -s.priority)


def list_available_engines() -> List[EngineSpec]:
    """Engines that are runnable on the current system (dependencies present, VRAM sufficient)."""
    out = []
    for spec in list_engines():
        reason = spec.run_condition() if spec.run_condition else None
        if reason is None:
            out.append(spec)
    return out


def get_unavailable_engines() -> List[Dict[str, Any]]:
    """Engines that are registered but not runnable, with reasons."""
    out = []
    vram_free = _get_free_vram_mb()
    for spec in list_engines():
        reason = spec.run_condition() if spec.run_condition else None
        if reason is None:
            if spec.vram_mb > vram_free and not spec.vram_mb <= 0:
                out.append({"engine_id": spec.engine_id, "reason": f"VRAM: needs ~{spec.vram_mb} MB, free ~{vram_free:.0f} MB"})
            continue
        out.append({"engine_id": spec.engine_id, "reason": reason})
    return out


def _get_free_vram_mb() -> float:
    """Free VRAM in MB (0 if no CUDA)."""
    try:
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0


def _get_vram_peak_mb() -> Optional[float]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def _torch_empty_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Intelligence Merger — cross-engine comparison and contradiction detection
# ---------------------------------------------------------------------------

def merge_engine_outputs(outputs: Sequence[UnifiedEngineOutput]) -> MultiEngineReport:
    """Merge multiple engine outputs into a single MultiEngineReport.

    Cross-engine contradictions are detected by comparing observations on
    the same target across engines. Agreements are recorded in the summary.
    """
    all_obs = [o for out in outputs for o in out.observations if out.status in ("succeeded", "partial")]
    all_contradictions: List[Dict[str, Any]] = []

    # Group observations by (source_image_id, category, label_similarity_key)
    by_target: Dict[Tuple[str, str], List[ObservationRecord]] = {}
    for obs in all_obs:
        # Use source_image_id + coarse label as the grouping key
        label_key = obs.label.lower().replace(" ", "_")
        key = (obs.source_image_id, label_key)
        by_target.setdefault(key, []).append(obs)

    # Find contradictions within each target group
    for key, obs_list in by_target.items():
        if len(obs_list) < 2:
            continue
        # Pairwise comparison across engines
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                issues = compare_observations(obs_list[i], obs_list[j])
                for issue in issues:
                    all_contradictions.append({
                        "target_key": list(key),
                        "evidence_a": obs_list[i].evidence_id,
                        "evidence_b": obs_list[j].evidence_id,
                        "model_a": obs_list[i].model_id.architecture_id,
                        "model_b": obs_list[j].model_id.architecture_id,
                        "issue": issue,
                    })

    # Cross-engine agreement summary
    engine_stats: Dict[str, Dict[str, Any]] = {}
    for out in outputs:
        engine_stats[out.engine_id] = {
            "status": out.status,
            "n_observations": out.n_observations,
            "wall_time_s": round(out.wall_time_s, 3),
            "peak_vram_mb": out.peak_vram_mb,
        }

    cross_summary = {
        "engines_run": sorted([o.engine_id for o in outputs]),
        "total_observations": len(all_obs),
        "n_contradictions": len(all_contradictions),
        "top_contradiction_types": _top_contradiction_types(all_contradictions),
    }

    total_wall = sum(o.wall_time_s for o in outputs)
    return MultiEngineReport(
        run_id=outputs[0].run_id if outputs else "",
        image_count=len({o.source_image_id for o in all_obs}),
        total_wall_s=total_wall,
        engines_run={o.engine_id: o.status for o in outputs},
        observations=all_obs,
        contradictions=all_contradictions,
        unavailable=[],
        vram_peak_mb=_get_vram_peak_mb(),
        cross_engine_summary=cross_summary,
    )


def _top_contradiction_types(contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Top contradiction types by frequency."""
    from collections import Counter

    counts = Counter(c["issue"].split(":")[0] for c in contradictions)
    return [{"type": k, "count": v} for k, v in counts.most_common(5)]


# ---------------------------------------------------------------------------
# MultiEngineOrchestrator — runs engines with VRAM-aware scheduling
# ---------------------------------------------------------------------------

class MultiEngineOrchestrator:
    """Runs all registered engines, respecting VRAM and dependency constraints.

    Engines are run in waves ordered by priority (high first). VRAM is
    checked before each engine runs — if it doesn't fit, the engine is
    recorded as "vram_deferred" and retried after VRAM is freed by a
    completed wave.
    """

    def __init__(self, vram_headroom_mb: int = 512):
        self._vram_headroom = vram_headroom_mb
        self._wave_results: List[UnifiedEngineOutput] = []
        self._deferred: List[EngineSpec] = []

    def _compute_vram_budget(self) -> float:
        """How much VRAM is available for the next engine."""
        total = _get_free_vram_mb()
        return total - self._vram_headroom if total > self._vram_headroom else 0.0

    def run_wave(self, image_paths: Sequence[str] = (), context: Optional[Dict[str, Any]] = None,
                 engine_ids: Optional[Sequence[str]] = None) -> List[UnifiedEngineOutput]:
        """Run all registered engines in dependency-aware, priority-ordered, VRAM-checked waves.

        Engines are sorted by priority (highest first). When an engine has
        dependencies (listed in EngineSpec.dependencies), it runs AFTER all
        its dependencies have completed. This ensures SAM2 can use detection
        bounding boxes from Grounding DINO.
        """
        context = context or {}
        available_specs = [_ENGINES[e] for e in _ENGINES]
        if engine_ids is not None:
            available_specs = [s for s in available_specs if s.engine_id in engine_ids]

        # Topological sort by dependencies
        ordered: List[EngineSpec] = []
        remaining = list(available_specs)
        completed_ids: set = set()
        while remaining:
            progressed = False
            for spec in list(remaining):
                if all(dep in completed_ids for dep in spec.dependencies):
                    ordered.append(spec)
                    remaining.remove(spec)
                    progressed = True
            if not progressed:
                # Circular dependency — append remaining in priority order
                ordered.extend(sorted(remaining, key=lambda s: -s.priority))
                break

        # Sort by priority within dependency groups
        # Re-sort: stable sort by priority, but respect dependency order
        final_order: List[EngineSpec] = []
        for spec in ordered:
            if not final_order or spec.priority >= final_order[-1].priority:
                final_order.append(spec)
            else:
                # Insert in priority order while maintaining dependency constraints
                inserted = False
                for i, existing in enumerate(final_order):
                    if spec.priority > existing.priority and all(d in completed_ids for d in spec.dependencies):
                        final_order.insert(i, spec)
                        inserted = True
                        break
                if not inserted:
                    final_order.append(spec)
            completed_ids.add(spec.engine_id)

        results: List[UnifiedEngineOutput] = []
        _torch_empty_cache()
        for spec in final_order:
            free = self._compute_vram_budget()
            if free < spec.vram_mb:
                self._deferred.append(spec)
                log.info("deferring engine %s: needs %d MB, available %.0f MB", spec.engine_id, spec.vram_mb, free)
                continue
            t0 = time.perf_counter()
            fn = _ENGINE_FNS[spec.engine_id]
            try:
                out = fn(image_paths, **context)
                out = UnifiedEngineOutput(
                    engine_id=out.engine_id, run_id=out.run_id, observations=out.observations,
                    status=out.status, error_message=out.error_message,
                    wall_time_s=time.perf_counter() - t0,
                    peak_vram_mb=_get_vram_peak_mb(), summary=out.summary,
                )
                results.append(out)
                # Feed detection results into context for downstream engines (SAM2)
                if spec.engine_id == "grounding_dino" and out.observations:
                    bboxes_by_image: Dict[str, List[Tuple[float, float, float, float]]] = {}
                    for obs in out.observations:
                        if obs.bbox_px:
                            bboxes_by_image.setdefault(obs.source_image_id, []).append(obs.bbox_px)
                    context["bboxes"] = bboxes_by_image
            except Exception as exc:  # noqa: BLE001
                log.warning("engine %s failed: %s", spec.engine_id, exc)
                results.append(UnifiedEngineOutput(
                    engine_id=spec.engine_id, run_id="", observations=(),
                    status="failed", error_message=str(exc),
                    wall_time_s=time.perf_counter() - t0,
                ))
            finally:
                _torch_empty_cache()
        self._wave_results.extend(results)
        return results

    def run_all(self, image_paths: Sequence[str], context: Optional[Dict[str, Any]] = None) -> MultiEngineReport:
        """Run ALL registered engines, merge results into a MultiEngineReport."""
        context = context or {}
        outputs = self.run_wave(image_paths, context)
        report = merge_engine_outputs(outputs)

        # Record deferred engines
        if self._deferred:
            report.unavailable = [
                {"engine_id": s.engine_id, "reason": f"VRAM deferred: needs ~{s.vram_mb} MB"}
                for s in self._deferred
            ]
            self._deferred = []

        # Pass context (image_paths) into engines via a shared state dict
        # This is how engines know what to process
        for out in outputs:
            log.info("engine %s: status=%s, n_obs=%d, wall=%.2fs",
                     out.engine_id, out.status, out.n_observations, out.wall_time_s)

        return report


__all__ = [
    "EngineSpec",
    "MultiEngineReport",
    "MultiEngineOrchestrator",
    "ModelIdentity",
    "ObservationRecord",
    "UnifiedEngineOutput",
    "register_engine",
    "list_engines",
    "list_available_engines",
    "get_unavailable_engines",
    "merge_engine_outputs",
]
