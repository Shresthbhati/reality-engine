"use client";

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, X, FlaskConical } from "lucide-react";
import { isApiError } from "@/lib/api/client";
import type { WorldIR } from "@/types/worldir";
import {
  evidenceHeadline,
  getEvidenceEntity,
  getEvidenceReadiness,
  getEvidenceWorld,
  listEvidenceEntities,
  listEvidenceRuns,
  listRejectedPredictions,
  type EvidenceEntityDetail,
  type EvidenceEntityRow,
  type EvidenceReadiness,
  type EvidenceRunSummary,
  type RejectedPrediction,
} from "@/lib/api/evidence-world";

const err = (e: unknown, fallback: string) => (isApiError(e) ? e.describe() : e instanceof Error ? e.message : fallback);

/** Evidence runs of one world: what ran, what was persisted, what is missing, what was refused, and every entity traced to its
 * evidence. A partial or empty result is shown as such -- never as a finished reconstruction. */
export default function EvidenceRunPanel({
  worldId,
  onClose,
  onViewWorld,
}: {
  worldId: string;
  onClose: () => void;
  /** Show this run's persisted WorldIR in the 3D viewport (labelled as a candidate, never replacing the committed world). */
  onViewWorld?: (runId: string, world: WorldIR) => void;
}) {
  const [runs, setRuns] = useState<EvidenceRunSummary[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  // everything below is keyed by the run it was loaded for, so switching runs never shows the previous run's data and no effect
  // has to reset state synchronously
  const [loaded, setLoaded] = useState<{ runId: string; readiness: EvidenceReadiness; entities: EvidenceEntityRow[]; rejected: RejectedPrediction[] } | null>(null);
  const [detailFor, setDetailFor] = useState<{ runId: string; detail: EvidenceEntityDetail } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const current = loaded && loaded.runId === runId ? loaded : null;
  const readiness = current?.readiness ?? null;
  const entities = current?.entities ?? [];
  const rejected = current?.rejected ?? [];
  const detail = detailFor && detailFor.runId === runId ? detailFor.detail : null;
  const busy = !!runId && !current && !error;

  useEffect(() => {
    let off = false;
    listEvidenceRuns(worldId)
      .then((r) => {
        if (off) return;
        setRuns(r);
        if (r.length) setRunId(r[r.length - 1].run_id);
      })
      .catch((e) => !off && setError(err(e, "Could not load evidence runs.")));
    return () => {
      off = true;
    };
  }, [worldId]);

  useEffect(() => {
    if (!runId) return;
    let off = false;
    Promise.all([getEvidenceReadiness(worldId, runId), listEvidenceEntities(worldId, runId), listRejectedPredictions(worldId, runId)])
      .then(([rd, en, rj]) => {
        if (off) return;
        setError(null);
        setLoaded({ runId, readiness: rd, entities: en, rejected: rj });
      })
      .catch((e) => !off && setError(err(e, "Could not load this evidence run.")));
    return () => {
      off = true;
    };
  }, [worldId, runId]);

  const inspect = (id: string) =>
    runId &&
    getEvidenceEntity(worldId, runId, id)
      .then((d) => setDetailFor({ runId, detail: d }))
      .catch((e) => setError(err(e, "Could not load this entity.")));

  const w = readiness?.world;
  const partial = !!w && w.status === "persisted" && w.state !== "complete";

  return (
    <div
      role="dialog"
      aria-labelledby="evidence-panel-title"
      className="absolute top-4 right-4 z-20 w-[26rem] max-h-[85vh] overflow-y-auto rounded-lg p-4 flex flex-col gap-3 shadow-2xl backdrop-blur-md"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
    >
      <div className="flex items-center justify-between pb-1 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-[var(--accent)]" />
          <h3 id="evidence-panel-title" className="text-sm font-semibold">Evidence runs</h3>
        </div>
        <button type="button" onClick={onClose} aria-label="Close evidence runs" className="p-1 rounded hover:bg-[var(--surface-hover)]">
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>
      </div>

      {error && (
        <div role="alert" className="flex gap-2 text-xs rounded p-2" style={{ background: "var(--surface-sunken)" }}>
          <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
        </div>
      )}
      {runs === null && !error && <Loader2 className="w-4 h-4 animate-spin" aria-label="Loading evidence runs" />}
      {runs !== null && runs.length === 0 && (
        <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
          No evidence run exists for this world. The evidence stage is opt-in and has not been run, so there is nothing to show.
        </p>
      )}

      {runs && runs.length > 0 && (
        <label className="text-xs flex flex-col gap-1">
          Run
          <select
            value={runId ?? ""}
            onChange={(e) => setRunId(e.target.value)}
            className="rounded p-1 text-xs"
            style={{ background: "var(--surface-sunken)", border: "1px solid var(--border)" }}
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.status}</option>
            ))}
          </select>
        </label>
      )}

      {busy && <Loader2 className="w-4 h-4 animate-spin" aria-label="Loading run" />}

      {readiness && !busy && (
        <>
          <div className="text-xs rounded p-2 flex gap-2" style={{ background: "var(--surface-sunken)" }} data-testid="evidence-headline">
            {partial || w?.status !== "persisted" ? <AlertTriangle className="w-4 h-4 shrink-0" /> : <CheckCircle2 className="w-4 h-4 shrink-0" />}
            <span>{evidenceHeadline(readiness)}</span>
          </div>
          <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
            <dt>Stage</dt><dd>{readiness.stage_status}</dd>
            <dt>Photographs</dt><dd>{readiness.evidence_received?.sources ?? "—"}</dd>
            <dt>Entities with position</dt><dd>{w?.entities_with_position ?? 0} / {w?.entities ?? 0}</dd>
            <dt>Rejected predictions</dt><dd>{readiness.rejected_predictions ?? "—"}</dd>
            <dt>Scale</dt><dd>{readiness.scale?.status ?? "—"}</dd>
            <dt>Physical accuracy</dt><dd>{readiness.physical_accuracy}</dd>
            <dt>Export</dt>
            <dd>{readiness.export_ready ? (readiness.metric_export_ready ? "ready (metric)" : "ready (NOT metres)") : "not ready"}</dd>
          </dl>
          {onViewWorld && runId && w?.status === "persisted" && (
            <button
              type="button"
              className="text-xs rounded px-2 py-1 self-start"
              style={{ border: "1px solid var(--border)" }}
              onClick={() =>
                getEvidenceWorld(worldId, runId)
                  .then((r) => onViewWorld(runId, r.world))
                  .catch((e) => setError(err(e, "Could not load this run's world.")))
              }
            >
              View this run&apos;s world in 3D
            </button>
          )}
          {readiness.export_scale_warning && <p className="text-[11px]" style={{ color: "var(--text-secondary)" }}>{readiness.export_scale_warning}</p>}
          {readiness.export_not_ready_reason && <p className="text-[11px]" style={{ color: "var(--text-secondary)" }}>{readiness.export_not_ready_reason}</p>}
          {(readiness.missing_evidence?.limitations.length ?? 0) > 0 && (
            <details className="text-[11px]">
              <summary>Limitations and missing evidence ({readiness.missing_evidence?.capture_requests ?? 0} capture requests)</summary>
              <ul className="list-disc pl-4 mt-1">{readiness.missing_evidence?.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
            </details>
          )}

          <section aria-label="Entities" className="flex flex-col gap-1">
            <h4 className="text-xs font-semibold">Entities ({entities.length})</h4>
            {entities.length === 0 && <p className="text-[11px]" style={{ color: "var(--text-secondary)" }}>No entity was derived from this run&apos;s evidence.</p>}
            {entities.map((e) => (
              <button key={e.entity_id} type="button" onClick={() => inspect(e.entity_id)} className="text-left text-[11px] rounded px-2 py-1 hover:bg-[var(--surface-hover)]">
                {e.ontology_path} <span style={{ color: "var(--text-tertiary)" }}>· {e.members.length} observations · {e.entity_id}</span>
              </button>
            ))}
          </section>

          {detail && (
            <section aria-label="Entity evidence" className="text-[11px] rounded p-2 flex flex-col gap-1" style={{ background: "var(--surface-sunken)" }}>
              <strong>{detail.entity.name}</strong>
              <span>
                {detail.entity.transform
                  ? `position ${JSON.stringify(detail.entity.transform["position"])} (${String(detail.entity.transform["units"] ?? "units undeclared")})`
                  : "No measured position: this entity has a class and 2D evidence only."}
              </span>
              <span>{detail.observations.length} observations, {detail.rejected_observations.length} rejected, {detail.capture_requests.length} capture requests</span>
              {detail.association_decisions.map((d) => <span key={`${d.a}-${d.b}`}>{d.outcome}: {d.reason}</span>)}
              {detail.trace_problems.length > 0 && <span role="alert">Trace problems: {detail.trace_problems.join("; ")}</span>}
            </section>
          )}

          <section aria-label="Rejected predictions" className="flex flex-col gap-1">
            <h4 className="text-xs font-semibold">Rejected predictions ({rejected.length})</h4>
            {rejected.slice(0, 20).map((r, i) => (
              <p key={i} className="text-[11px]"><strong>{r.engine_id}</strong> ({r.task}): {r.reason}</p>
            ))}
            {rejected.length > 20 && <p className="text-[11px]">… and {rejected.length - 20} more</p>}
          </section>
        </>
      )}
    </div>
  );
}
