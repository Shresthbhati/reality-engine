/**
 * Pipeline graph view-model. Pure and dependency-free (no imports) so it runs under `node --test`.
 *
 * The nodes are the stages the job runner (apps/api/jobs.py) really reports through `job.stage`, in the order it
 * sets them. The runner records only the CURRENT stage -- no per-stage timestamps, progress or confidence -- so:
 *   - a node's status is derived from the job status and the stage position, nothing else;
 *   - anything the API did not record is `NOT_RECORDED`, never a made-up number;
 *   - failed is never shown as succeeded, and a stage the runner never reached is "not-run".
 */

export const NOT_RECORDED = "Not recorded";

export type NodeStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "partial"
  | "failed"
  | "cancelled"
  | "skipped"
  | "not-run";

/** The job fields the view-model reads (GET /api/jobs/{id}). */
export interface PipelineJob {
  id: string;
  status: string;
  stage: string | null;
  created_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  payload?: { adopted?: boolean; unchanged?: boolean; version_id?: string | null } | null;
}

export interface PipelineFact {
  label: string;
  value: string;
}

export interface PipelineNode {
  id: string;
  label: string;
  /** the `job.stage` values that mean "the runner is in this node" */
  stages: readonly string[];
  status: NodeStatus;
  /** one honest line explaining the status, when the status alone is not enough */
  detail: string | null;
  facts: PipelineFact[];
}

/** Report `stages` (the head version's report) -- every field optional: older versions did not record them. */
export interface ReportStages {
  reconstruction?: {
    backend?: string | null;
    cameras_registered?: number;
    cameras_input?: number;
    registration_status?: string | null;
    points?: number;
  } | null;
  scale?: { state?: string | null; meters_per_unit?: number | null } | null;
  depth?: { status?: string | null } | null;
  perception?: { status?: string | null } | null;
  mesh?: { status?: string | null } | null;
  compile?: { entities?: number; measurements?: number; relationships?: number } | null;
}

export interface PipelineInputs {
  job: PipelineJob | null;
  stages?: ReportStages | null;
  /** photographs in the world / excluded from geometry, from the status endpoint */
  evidenceCount?: number | null;
  excludedCount?: number | null;
}

interface NodeDef {
  id: string;
  label: string;
  stages: readonly string[];
}

/** Order and names mirror the `job.stage = ...` assignments in apps/api/jobs.py (reconstruct job). */
export const PIPELINE_DEFS: readonly NodeDef[] = [
  { id: "backend", label: "Backend check", stages: ["checking_reconstruction_backend"] },
  { id: "evidence", label: "Resolve evidence", stages: ["resolving_evidence"] },
  { id: "analyze", label: "Analyze evidence", stages: ["analyzing_evidence"] },
  { id: "reconstruct", label: "Reconstruct", stages: ["building_rough_model", "reconstructing", "refining"] },
  { id: "commit", label: "Commit version", stages: ["committing_version"] },
];

export function nodeIndexForStage(stage: string | null | undefined): number {
  if (!stage) return -1;
  return PIPELINE_DEFS.findIndex((d) => d.stages.includes(stage));
}

const str = (v: unknown): string => (typeof v === "string" && v ? v : NOT_RECORDED);
const num = (v: unknown): string => (typeof v === "number" && Number.isFinite(v) ? String(v) : NOT_RECORDED);

function factsFor(id: string, inp: PipelineInputs): PipelineFact[] {
  const s = inp.stages ?? null;
  switch (id) {
    case "evidence":
      return [{ label: "Photos", value: num(inp.evidenceCount) }];
    case "analyze":
      return [{ label: "Excluded from geometry", value: num(inp.excludedCount) }];
    case "reconstruct": {
      const r = s?.reconstruction;
      return [
        { label: "Backend", value: str(r?.backend) },
        {
          label: "Photos placed",
          value:
            typeof r?.cameras_registered === "number" && typeof r?.cameras_input === "number"
              ? `${r.cameras_registered} of ${r.cameras_input}`
              : NOT_RECORDED,
        },
        { label: "Points", value: num(r?.points) },
        { label: "Scale", value: str(s?.scale?.state) },
        { label: "Depth", value: str(s?.depth?.status) },
        { label: "Perception", value: str(s?.perception?.status) },
        { label: "Mesh", value: str(s?.mesh?.status) },
      ];
    }
    case "commit": {
      const c = s?.compile;
      return [
        { label: "Entities", value: num(c?.entities) },
        { label: "Measurements", value: num(c?.measurements) },
        { label: "Relationships", value: num(c?.relationships) },
      ];
    }
    default:
      return [];
  }
}

/**
 * Per-node status from the job. Rules:
 *  - no job: every node "not-run" (no run recorded);
 *  - queued: nothing has started, every node "queued";
 *  - running at stage i: before i succeeded (the runner moved on), i running, after i queued; an unknown/missing
 *    stage marks nothing running;
 *  - succeeded: all succeeded, except the commit node is "skipped" when the candidate was not adopted;
 *  - partial: all succeeded except "reconstruct", which carries the degradation;
 *  - failed/cancelled at stage i: before i succeeded, i failed/cancelled, after i not-run; with no recorded stage
 *    the failing node cannot be placed, so every node is "not-run" and says so;
 *  - any other status is treated as failed (never optimistic).
 */
export function buildPipelineNodes(inp: PipelineInputs): PipelineNode[] {
  const job = inp.job;
  const at = job ? nodeIndexForStage(job.stage) : -1;
  const nodes: PipelineNode[] = PIPELINE_DEFS.map((d) => ({
    id: d.id,
    label: d.label,
    stages: d.stages,
    status: "not-run" as NodeStatus,
    detail: null,
    facts: factsFor(d.id, inp),
  }));
  if (!job) {
    nodes.forEach((n) => (n.detail = "No run recorded"));
    return nodes;
  }
  const set = (i: number, status: NodeStatus, detail: string | null = null) => {
    nodes[i].status = status;
    nodes[i].detail = detail;
  };
  const last = nodes.length - 1;
  switch (job.status) {
    case "queued":
      nodes.forEach((_, i) => set(i, "queued", i === 0 ? "Waiting for a worker" : null));
      break;
    case "running":
      nodes.forEach((_, i) => {
        if (at < 0) set(i, "queued", i === 0 ? "Current stage not reported" : null);
        else set(i, i < at ? "succeeded" : i === at ? "running" : "queued");
      });
      break;
    case "succeeded":
      nodes.forEach((_, i) => set(i, "succeeded"));
      if (job.payload?.adopted === false) {
        set(
          last,
          "skipped",
          job.payload.unchanged
            ? "Nothing new: the current version already reflects the evidence"
            : "Candidate not adopted: the current model was kept",
        );
      }
      break;
    case "partial":
      nodes.forEach((_, i) => set(i, "succeeded"));
      set(3, "partial", "Finished with degraded stages; see the reconstruct facts");
      break;
    default: {
      const bad: NodeStatus = job.status === "cancelled" ? "cancelled" : "failed";
      if (at < 0) {
        nodes.forEach((_, i) =>
          set(i, "not-run", i === 0 ? `${bad === "failed" ? "Failed" : "Cancelled"}; the stage it stopped at was not recorded` : null),
        );
      } else {
        nodes.forEach((_, i) => set(i, i < at ? "succeeded" : i === at ? bad : "not-run"));
      }
    }
  }
  return nodes;
}

/** Wall-clock seconds between two recorded ISO timestamps; null when either is missing or unparseable. */
export function jobDurationSeconds(job: Pick<PipelineJob, "started_at" | "completed_at"> | null): number | null {
  if (!job?.started_at || !job.completed_at) return null;
  const a = Date.parse(job.started_at);
  const b = Date.parse(job.completed_at);
  return Number.isFinite(a) && Number.isFinite(b) && b >= a ? (b - a) / 1000 : null;
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null) return NOT_RECORDED;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${Math.round(seconds - m * 60)} s`;
}

/** The slice of GET /api/worlds/{id}/status the summary reads. */
export interface SummaryInput {
  state: string;
  state_label?: string;
  in_progress?: boolean;
  has_model?: boolean;
  model?: {
    images_used?: number;
    images_registered?: number;
    scale?: { state?: string | null } | null;
    uncertainties?: unknown[];
    dense?: { state?: string } | null;
  } | null;
  versions?: { label: string; is_current: boolean }[];
  failure?: { kind: string; message: string | null } | null;
}

export type SummaryTone = "success" | "warning" | "error" | "active" | "neutral";

export interface PipelineSummary {
  headline: string;
  tone: SummaryTone;
  rows: PipelineFact[];
}

/** The headline is "World constructed" only for READY_TO_INSPECT -- a cleanly graded, adopted version. */
export function buildPipelineSummary(s: SummaryInput, entities: number | null | undefined): PipelineSummary {
  let headline: string;
  let tone: SummaryTone;
  switch (s.state) {
    case "READY_TO_INSPECT":
      headline = "World constructed";
      tone = "success";
      break;
    case "PARTIALLY_COMPLETE":
      headline = "World partially constructed";
      tone = "warning";
      break;
    case "NEEDS_MORE_EVIDENCE":
      headline = "More evidence needed";
      tone = "warning";
      break;
    case "FAILED":
      headline = "Construction failed";
      tone = "error";
      break;
    case "CANCELLED":
      headline = "Construction cancelled";
      tone = "neutral";
      break;
    case "CAPTURING":
      headline = s.has_model ? "Collecting evidence" : "No world constructed yet";
      tone = s.has_model ? "active" : "neutral";
      break;
    default:
      headline = s.in_progress ? "Constructing world" : "State not recognised";
      tone = s.in_progress ? "active" : "neutral";
  }
  const m = s.model ?? null;
  const current = s.versions?.find((v) => v.is_current);
  const warnings: string[] = [];
  if (m?.uncertainties?.length) warnings.push(`${m.uncertainties.length} uncertaint${m.uncertainties.length === 1 ? "y" : "ies"}`);
  if (s.failure) warnings.push(s.failure.message ? `${s.failure.kind}: ${s.failure.message}` : s.failure.kind);
  return {
    headline,
    tone,
    rows: [
      { label: "Version", value: current?.label ?? "No version" },
      {
        label: "Photos placed",
        value:
          typeof m?.images_registered === "number" && typeof m?.images_used === "number"
            ? `${m.images_registered} of ${m.images_used}`
            : NOT_RECORDED,
      },
      { label: "Entities", value: num(entities) },
      { label: "Scale", value: str(m?.scale?.state) },
      { label: "Geometry", value: m?.dense?.state ? (m.dense.state === "dense" ? "Dense" : "Sparse") : NOT_RECORDED },
      { label: "Warnings", value: warnings.length ? warnings.join("; ") : "None recorded" },
    ],
  };
}
