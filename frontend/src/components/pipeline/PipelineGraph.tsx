"use client";

import Link from "next/link";
import { AlertTriangle, CheckCircle2, Circle, CircleSlash, Clock, Loader2, MinusCircle, XCircle } from "lucide-react";
import { useApi } from "@/lib/api/hooks";
import { fetchWorldReport } from "@/lib/api/worldir";
import { useWorldStatus } from "@/lib/api/reconstructions";
import {
  NOT_RECORDED,
  buildPipelineNodes,
  buildPipelineSummary,
  formatDuration,
  jobDurationSeconds,
  type NodeStatus,
  type PipelineJob,
  type ReportStages,
  type SummaryTone,
} from "@/lib/api/pipeline-graph";

const STATUS_LABEL: Record<NodeStatus, string> = {
  queued: "Queued",
  running: "Running",
  succeeded: "Succeeded",
  partial: "Partial",
  failed: "Failed",
  cancelled: "Cancelled",
  skipped: "Skipped",
  "not-run": "Not run",
};

const STATUS_COLOR: Record<NodeStatus, string> = {
  queued: "var(--text-tertiary)",
  running: "var(--accent)",
  succeeded: "var(--success)",
  partial: "var(--warning)",
  failed: "var(--error)",
  cancelled: "var(--text-secondary)",
  skipped: "var(--text-secondary)",
  "not-run": "var(--text-tertiary)",
};

const TONE_COLOR: Record<SummaryTone, string> = {
  success: "var(--success)",
  warning: "var(--warning)",
  error: "var(--error)",
  active: "var(--accent)",
  neutral: "var(--text-secondary)",
};

function StatusIcon({ status }: { status: NodeStatus }) {
  const cls = "w-3.5 h-3.5 shrink-0";
  const style = { color: STATUS_COLOR[status] };
  switch (status) {
    case "succeeded":
      return <CheckCircle2 className={cls} style={style} aria-hidden />;
    case "running":
      return <Loader2 className={`${cls} animate-spin`} style={style} aria-hidden />;
    case "failed":
      return <XCircle className={cls} style={style} aria-hidden />;
    case "partial":
      return <AlertTriangle className={cls} style={style} aria-hidden />;
    case "cancelled":
      return <CircleSlash className={cls} style={style} aria-hidden />;
    case "skipped":
      return <MinusCircle className={cls} style={style} aria-hidden />;
    case "queued":
      return <Clock className={cls} style={style} aria-hidden />;
    default:
      return <Circle className={cls} style={style} aria-hidden />;
  }
}

async function fetchJob(jobId: string): Promise<PipelineJob | null> {
  const res = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) return null;
  return (await res.json()) as PipelineJob;
}

/**
 * The pipeline graph for one world: the stages the job runner really reports, each with the status derived from the
 * newest reconstruction job, plus an honest world summary. Nothing is estimated; unrecorded values read
 * "Not recorded". Polling is the status endpoint's own (fast while the engine works, slow once settled); the job and
 * report are re-read only when that status changes, so a terminal job is not polled.
 */
export default function PipelineGraph({ worldId, compact = false }: { worldId: string; compact?: boolean }) {
  const { status, error, loading } = useWorldStatus(worldId);
  const jobId = status?.job?.id ?? null;
  const { data: job } = useApi(
    () => (jobId ? fetchJob(jobId) : Promise.resolve(null)),
    [jobId, status?.job?.status, status?.job?.stage],
  );
  const versionId = status?.model?.version_id ?? null;
  const { data: report } = useApi(
    () => (versionId ? fetchWorldReport(worldId) : Promise.resolve(null)),
    [worldId, versionId],
  );

  if (error && !status) {
    return (
      <p role="alert" className="text-xs font-mono" style={{ color: "var(--error)" }}>
        Pipeline status unavailable: {error.message}
      </p>
    );
  }
  if (loading || !status) {
    return (
      <p className="text-xs font-mono" style={{ color: "var(--text-tertiary)" }}>
        Loading pipeline...
      </p>
    );
  }

  const stages = (report?.stages ?? null) as ReportStages | null;
  const nodes = buildPipelineNodes({
    job: job ?? null,
    stages,
    evidenceCount: status.evidence_summary?.count,
    excludedCount: status.evidence_summary?.excluded?.length,
  });
  const summary = buildPipelineSummary(status, stages?.compile?.entities);
  const duration = formatDuration(jobDurationSeconds(job ?? null));
  const toneColor = TONE_COLOR[summary.tone];

  return (
    <div className="flex flex-col gap-3 min-w-0" data-testid="pipeline-graph">
      <ol
        className={`flex items-stretch gap-0 ${compact ? "overflow-x-auto pb-1" : "flex-wrap gap-y-4"}`}
        aria-label="Reconstruction pipeline stages"
      >
        {nodes.map((n, i) => (
          <li key={n.id} className="flex items-center min-w-0">
            <div
              className={`rounded-md border flex flex-col gap-1.5 ${compact ? "w-40 p-2" : "w-56 p-3"}`}
              style={{
                background: "var(--bg-surface)",
                borderColor: n.status === "running" || n.status === "failed" ? STATUS_COLOR[n.status] : "var(--border)",
              }}
              data-status={n.status}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold truncate" style={{ color: "var(--text-primary)" }}>
                  {n.label}
                </span>
                <span className="flex items-center gap-1 text-[10px] uppercase tracking-wide" style={{ color: STATUS_COLOR[n.status] }}>
                  <StatusIcon status={n.status} />
                  {STATUS_LABEL[n.status]}
                </span>
              </div>
              {n.detail && (
                <p className="text-[10px] leading-snug" style={{ color: "var(--text-secondary)" }}>
                  {n.detail}
                </p>
              )}
              {!compact && n.facts.length > 0 && (
                <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-[11px]">
                  {n.facts.map((f) => (
                    <div key={f.label} className="contents">
                      <dt style={{ color: "var(--text-tertiary)" }}>{f.label}</dt>
                      <dd
                        className="font-mono-num text-right truncate"
                        style={{ color: f.value === NOT_RECORDED ? "var(--text-tertiary)" : "var(--text-primary)" }}
                      >
                        {f.value}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
              {!compact && (
                <p className="text-[10px]" style={{ color: "var(--text-tertiary)" }}>
                  Stage timing: {NOT_RECORDED}
                </p>
              )}
            </div>
            {i < nodes.length - 1 && (
              <span
                className="w-5 h-px shrink-0"
                style={{ background: nodes[i].status === "succeeded" ? "var(--success)" : "var(--border)" }}
                aria-hidden
              />
            )}
          </li>
        ))}
      </ol>

      <section
        className={`rounded-md border ${compact ? "px-3 py-2" : "p-4"}`}
        style={{ background: "var(--bg-elevated)", borderColor: "var(--border)" }}
        aria-label="Pipeline summary"
      >
        <div className="flex items-center gap-3 flex-wrap">
          <h3 className={`${compact ? "text-xs" : "text-lg"} font-semibold`} style={{ color: toneColor }}>
            {summary.headline}
          </h3>
          <span className="text-[11px] font-mono-num" style={{ color: "var(--text-tertiary)" }}>
            Run time: {duration}
          </span>
          {compact && (
            <Link href={`/worlds/${encodeURIComponent(worldId)}/pipeline`} className="text-[11px] ml-auto hover:underline" style={{ color: "var(--accent)" }}>
              Open pipeline view
            </Link>
          )}
        </div>
        <dl className={`grid gap-x-6 gap-y-1 mt-2 text-xs ${compact ? "grid-cols-3" : "grid-cols-2 md:grid-cols-3 lg:grid-cols-6"}`}>
          {summary.rows.map((r) => (
            <div key={r.label} className="min-w-0">
              <dt className="text-[10px] uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
                {r.label}
              </dt>
              <dd
                className="font-mono-num truncate"
                title={r.value}
                style={{ color: r.value === NOT_RECORDED ? "var(--text-tertiary)" : "var(--text-primary)" }}
              >
                {r.value}
              </dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  );
}
