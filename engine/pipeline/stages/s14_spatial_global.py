"""Optional stage `spatial_global`: joint multi-view optimization + global validation of structural observations.

Registered ONLY in the non-default profile ``interior_survey_spatial`` (registry.PIPELINE_PROFILES), selected by the host when
``RE_SPATIAL_GLOBAL`` is on. The stage reads ``ctx.state["spatial_inputs"]`` (the `spatial_inputs.v1` dict) and optionally
``ctx.state["spatial_store"]`` / ``["spatial_parent"]`` (a WorldStore and the previous spatial version, for lineage). With no
inputs it is ``skipped`` with the reason -- never succeeded: no producer, no optimization. Facts are left in
``ctx.state["spatial_facts"]`` for the host's job report. It is optional (a failure grades the job PARTIAL, never FAILED).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from engine.pipeline.stages.base import StageContext, StageResult, StageSpec


class SpatialGlobal:
    spec = StageSpec(name="spatial_global", version="1", resource_class="cpu", required=False, idempotent=True,
                     inputs=None, requires_outputs=())

    def run(self, ctx: StageContext) -> StageResult:
        from reconstruction.spatial.stage import run_spatial_global

        started = datetime.now(timezone.utc)

        def done(status, warnings=(), metrics=None, outputs=None, error=None):
            return StageResult(name=self.spec.name, status=status, inputs_hash="", outputs=dict(outputs or {}),
                               metrics=dict(metrics or {}), warnings=list(warnings), error=error, started_at=started,
                               ended_at=datetime.now(timezone.utc), peak_rss_mb=None)

        inputs = ctx.state.get("spatial_inputs")
        if inputs is None:
            # Lazy per-job provider (set by the host before running stages): avoids module-global state
            # so concurrent jobs never observe each other's observations.
            provider = ctx.state.get("spatial_inputs_provider")
            if provider is not None:
                try:
                    inputs = provider()
                except Exception:  # noqa: BLE001 - a producer bug skips the stage, never sinks the job
                    import logging
                    logging.getLogger("reality.api.spatial_global").exception("spatial inputs provider failed")
                    inputs = None
            if inputs is None:
                # Module-global seam (backward compatibility with tests and ad-hoc registration).
                # The per-job provider above takes precedence; this is the fallback.
                try:
                    from apps.api.spatial_global_stage import inputs_for_job as _inputs_for_job
                    inputs = _inputs_for_job(ctx.state.get("job_obj"))
                except Exception:  # noqa: BLE001
                    inputs = None
            if inputs is None:
                ctx.state["spatial_facts"] = {"status": "skipped", "reason": "no structural observations were supplied by a producer"}
                return done("skipped", ["no structural observations supplied: nothing to optimize (producer not wired for this job)"])
        store, parent, warn = ctx.state.get("spatial_store"), ctx.state.get("spatial_parent"), []
        opener = ctx.state.get("spatial_store_opener")
        if opener is not None and store is None:
            try:
                store, parent = opener()
            except ValueError as exc:     # an invalid tenant/world id: run without persistence rather than guess a path
                warn.append(f"spatial world not persisted: {exc}")
        out = run_spatial_global(inputs, store=store, parent=parent,
                                 solver=ctx.params.get("spatial_solver", "auto"))
        ctx.state["spatial_facts"] = out.facts
        if out.status in ("failed", "refused"):
            e = out.errors[0]
            return done("failed", [f"{e['code']}: {e['message']}"], error=f"{e['code']}: {e['message']}")
        if out.status == "insufficient":
            return done("skipped", [f"{out.facts['code']}: {out.facts['reason']}"])
        if out.status == "unchanged":
            return done("skipped", ["identical input hash: the existing spatial version stands"])
        f = out.facts
        f["warnings"] = list(f["warnings"]) + warn
        o = f["optimization"]
        blob = json.dumps(out.world.to_dict(), sort_keys=True, default=str).encode()
        metrics = {"n_images": float(f["n_images"]), "n_observations": float(f["n_observations"]), "cost_before": o["cost_before"],
                   "cost_after": o["cost_after"], "nfev": float(o["nfev"]), "optimizer_seconds": o["seconds"],
                   "outliers": float(len(f["outliers"])), "rooms": float(f["completeness"]["topological"]["rooms_enclosed"]),
                   "structural_errors": float(f["validation"]["checks"]["structural_errors"])}
        return done("degraded" if f["warnings"] else "succeeded", f["warnings"], metrics,
                    {"spatial_world": hashlib.sha256(blob).hexdigest()})
