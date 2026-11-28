"use client";

import { useEffect, useState } from "react";
import { Link2, Loader2, FileQuestion } from "lucide-react";
import {
  fetchEntityProvenance,
  type EntityProvenance,
} from "@/lib/api/provenance";

/**
 * entity -> reconstruction -> observation -> source evidence.
 *
 * The chain is only as precise as the backend's own trace. The API reports
 * three levels, and this component states which one it got rather than
 * flattening them:
 *
 *   observation  an Observation's data_hash/data_uri matched a real
 *                Evidence row -- file-level, exact.
 *   session      no per-observation match, so this falls back to every
 *                Evidence row from the version's source sessions -- real,
 *                but coarser ("came from this capture", not "this frame").
 *   none         procedurally generated, or predates session tracking.
 *
 * Rendering a session-level trace as if it were file-level would overstate
 * what the engine actually knows.
 */
export default function EvidenceTrace({
  worldId,
  entityId,
  onSelectEvidence,
}: {
  worldId: string;
  entityId: string;
  onSelectEvidence?: (evidenceId: string) => void;
}) {
  const [state, setState] = useState<
    { kind: "loading" } | { kind: "loaded"; data: EntityProvenance } | { kind: "error" }
  >({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ kind: "loading" });
    fetchEntityProvenance(worldId, entityId)
      .then((data) => {
        if (cancelled) return;
        setState(data ? { kind: "loaded", data } : { kind: "error" });
      })
      .catch(() => {
        if (!cancelled) setState({ kind: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [worldId, entityId]);

  if (state.kind === "loading") {
    return (
      <div className="flex items-center gap-2 text-[11px] text-neutral-500 py-2">
        <Loader2 className="w-3 h-3 animate-spin" />
        Tracing provenance...
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="text-[11px] text-neutral-500 border border-dashed border-neutral-800 rounded px-2.5 py-2">
        Provenance for this entity is unavailable. The API did not return a
        trace.
      </div>
    );
  }

  const { data } = state;
  const levelLabel: Record<EntityProvenance["trace_level"], string> = {
    observation: "File-level match",
    session: "Session-level (no per-frame link)",
    none: "No source evidence",
  };

  return (
    <div className="space-y-2" data-testid="evidence-trace">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-neutral-500">
        <Link2 className="w-3 h-3" />
        {levelLabel[data.trace_level]}
      </div>

      <dl className="text-[11px] space-y-0.5">
        <div className="flex justify-between gap-2">
          <dt className="text-neutral-500">Version</dt>
          <dd className="font-mono text-neutral-300 truncate">{data.version_id}</dd>
        </div>
        <div className="flex justify-between gap-2">
          <dt className="text-neutral-500">Provenance</dt>
          <dd className="font-mono text-neutral-300">
            {data.provenance ?? "unrecorded"}
          </dd>
        </div>
      </dl>

      {data.trace_level === "none" ? (
        <div className="flex items-start gap-2 text-[11px] text-neutral-500 border border-dashed border-neutral-800 rounded px-2.5 py-2 leading-relaxed">
          <FileQuestion className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>
            {data.reason ??
              "This entity has no source evidence recorded against it."}
          </span>
        </div>
      ) : data.evidence.length === 0 ? (
        <div className="text-[11px] text-neutral-600">No evidence rows matched.</div>
      ) : (
        <ul className="space-y-0.5 max-h-48 overflow-y-auto">
          {data.evidence.map((e) => (
            <li key={`${e.observation_id ?? "s"}-${e.evidence_id}`}>
              <button
                type="button"
                onClick={() => onSelectEvidence?.(e.evidence_id)}
                disabled={!onSelectEvidence}
                className="w-full text-left text-[11px] text-neutral-300 hover:text-[#00e5ff] disabled:hover:text-neutral-300 transition-colors truncate"
              >
                {e.evidence_name}
                <span className="text-neutral-600"> &middot; {e.evidence_type}</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {data.trace_level === "session" && data.evidence.length > 0 && (
        <p className="text-[10px] text-neutral-600 leading-relaxed">
          These are all frames from the version&rsquo;s source capture
          {data.source_session_ids?.length
            ? ` (session ${data.source_session_ids.join(", ")})`
            : ""}
          , not the specific frames this entity came from.
        </p>
      )}
    </div>
  );
}