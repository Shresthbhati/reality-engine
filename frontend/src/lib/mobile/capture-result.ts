/**
 * View-models for the mobile capture screens. Pure functions over what the API
 * ALREADY measured (GET /api/worlds/{id}/status): every sentence here maps to a
 * backend field. No coverage percentages, no scores, no "good enough" verdicts
 * are computed or invented on the phone.
 */
import type { QueueItem } from "./upload-queue";

export type Tone = "good" | "warn" | "bad" | "working" | "neutral";

/** The subset of `EvidenceStatus` (lib/api/reconstructions.ts) this module reads. */
export interface EvidenceStatusLike {
  id: string;
  evidence_class: string | null;
  used_for_geometry: boolean | null;
  registered: boolean | null;
  contribution: string | null;
  in_current_model: boolean;
  registration?: { state: "registered" | "waiting"; attempts: number; ever_registered: boolean } | null;
}

/** The subset of `WorldStatus` this module reads. */
export interface WorldStatusLike {
  state: string;
  state_label: string;
  in_progress: boolean;
  has_model: boolean;
  model: { model_state: "ROUGH" | "PARTIAL" | "REFINED" | null; images_registered: number; images_used: number } | null;
  failure: { kind: string; message: string | null } | null;
  last_run?: { adopted: boolean; unchanged?: boolean } | null;
  guidance: { kind: string; message: string; basis: string }[];
}

export interface ResultView {
  tone: Tone;
  label: string;
  detail: string | null;
}

const CONTRIBUTION_COPY: Record<string, string> = {
  first: "First view of this place",
  new_view: "Adds a new viewpoint",
  redundant: "Mostly repeats a view the model already has",
  disconnected: "Shares nothing verifiable with the rest",
  unknown: "Not analysed yet",
};

/** What the engine did with ONE uploaded photo (queue state + the backend's per-photo status). */
export function photoResult(
  item: Pick<QueueItem, "status" | "progress" | "error" | "rejectedReason">,
  ev: EvidenceStatusLike | undefined,
): ResultView {
  switch (item.status) {
    case "queued":
      return { tone: "neutral", label: "Queued", detail: "Waiting for the previous photo to finish uploading." };
    case "uploading":
      return item.progress >= 1
        ? { tone: "working", label: "Sent", detail: "Waiting for the engine to accept it." }
        : { tone: "working", label: `Uploading ${Math.round(item.progress * 100)}%`, detail: null };
    case "failed":
      return { tone: "bad", label: "Upload failed", detail: item.error ?? "The upload did not complete." };
    case "rejected":
      return { tone: "bad", label: "Not accepted", detail: item.rejectedReason ?? "The engine rejected this photo." };
    case "accepted":
      break;
  }

  if (!ev) return { tone: "working", label: "Accepted", detail: "Waiting for analysis." };
  if (ev.used_for_geometry === false) {
    return {
      tone: "neutral",
      label: "Kept as context",
      detail: `Saved${ev.evidence_class ? ` as ${ev.evidence_class}` : ""}; it does not shape the 3D model.`,
    };
  }
  const contribution = ev.contribution ? (CONTRIBUTION_COPY[ev.contribution] ?? ev.contribution) : null;
  if (ev.registration?.state === "waiting") {
    return {
      tone: "warn",
      label: "Waiting to be placed",
      detail: `Kept and retried on every rebuild (tried ${ev.registration.attempts} time${ev.registration.attempts === 1 ? "" : "s"}).`,
    };
  }
  if (ev.registered === true || ev.in_current_model) {
    return { tone: "good", label: "Placed in the model", detail: contribution };
  }
  if (ev.registered === false) {
    return { tone: "warn", label: "Not placed", detail: contribution ?? "The engine could not place this photo in the model." };
  }
  if (!ev.contribution) return { tone: "working", label: "Accepted", detail: "Waiting for analysis." };
  return { tone: "neutral", label: "Accepted", detail: contribution };
}

export interface WorldHeadline {
  tone: Tone;
  headline: string;
  detail: string | null;
  /** "3 of 5 photos placed", only when a model exists */
  placed: string | null;
  /** the backend's top guidance sentence, verbatim; null when it gave none */
  guidance: string | null;
}

/**
 * Honest headline for the reconstruction state. Vocabulary follows
 * components/reconstruct/ReconstructionStatusBar.tsx. `modelCopy` is injected
 * (it is `modelStateCopy` from lib/api) so this module stays dependency-free.
 */
export function worldHeadline(
  s: WorldStatusLike,
  modelCopy: (m: "ROUGH" | "PARTIAL" | "REFINED" | null) => { label: string; detail: string },
): WorldHeadline {
  const guidance = s.guidance[0]?.message ?? null;
  const placed = s.model ? `${s.model.images_registered} of ${s.model.images_used} photos placed` : null;
  const copy = s.model ? modelCopy(s.model.model_state) : null;
  const modelLine = copy && copy.detail ? `${copy.label}. ${copy.detail}` : null;
  const base = { placed, guidance };

  if (s.failure) {
    const f = s.failure;
    if (f.kind === "cancelled") {
      return { ...base, tone: "neutral", headline: "Cancelled", detail: "Your photos are kept and the previous model is unchanged." };
    }
    if (f.kind === "insufficient_evidence") {
      return { ...base, tone: "warn", headline: "Needs more photos", detail: f.message ?? "No usable photographs yet." };
    }
    return {
      ...base,
      tone: "bad",
      headline: "Could not finish",
      detail: `${f.message ?? "No details were recorded."}${s.has_model ? " The previous model is still available." : ""}`,
    };
  }
  if (s.in_progress) {
    return {
      ...base,
      tone: "working",
      headline: s.state_label,
      detail: s.has_model ? "Rebuilding. The previous model stays available until the new one is ready." : "The first model appears as soon as it is ready.",
    };
  }
  if (s.last_run && s.last_run.unchanged) {
    return { ...base, tone: "good", headline: "Already up to date", detail: "This model reflects all your photos." };
  }
  if (s.last_run && !s.last_run.adopted) {
    return {
      ...base,
      tone: "warn",
      headline: "Kept your current model",
      detail: "The new photos are stored but did not improve it yet. They are tried again as more evidence arrives.",
    };
  }
  if (s.state === "NEEDS_MORE_EVIDENCE") {
    return { ...base, tone: "warn", headline: s.state_label, detail: modelLine };
  }
  if (s.has_model) {
    return { ...base, tone: s.state === "READY_TO_INSPECT" ? "good" : "neutral", headline: s.state_label, detail: modelLine };
  }
  return { ...base, tone: "neutral", headline: s.state_label || "No model yet", detail: "No model has been built yet." };
}

export interface EvidenceQualityEntry {
  result: string;
  score: number | null;
}

/**
 * Tolerant reader for GET /api/evidence/quality. Accepts a bare array or
 * `{items: [...]}`; an entry counts only if it names its evidence and carries a
 * result string. Anything else is ignored (never defaulted to a score).
 */
export function parseEvidenceQuality(raw: unknown): Map<string, EvidenceQualityEntry> {
  const out = new Map<string, EvidenceQualityEntry>();
  const list = Array.isArray(raw) ? raw : raw && typeof raw === "object" ? (raw as { items?: unknown }).items : null;
  if (!Array.isArray(list)) return out;
  for (const entry of list) {
    if (!entry || typeof entry !== "object") continue;
    const e = entry as Record<string, unknown>;
    const id = typeof e.evidence_id === "string" ? e.evidence_id : typeof e.id === "string" ? e.id : null;
    const result = [e.result, e.verdict, e.status].find((v): v is string => typeof v === "string" && v.trim().length > 0);
    if (!id || !result) continue;
    const scoreRaw = typeof e.score === "number" ? e.score : typeof e.quality_score === "number" ? e.quality_score : null;
    out.set(id, { result, score: scoreRaw !== null && Number.isFinite(scoreRaw) ? scoreRaw : null });
  }
  return out;
}

/** "REDUNDANT" / "needs_more" -> "Redundant" / "Needs more" for a chip. */
export function resultLabel(result: string): string {
  const t = result.replace(/[_-]+/g, " ").trim().toLowerCase();
  return t.charAt(0).toUpperCase() + t.slice(1);
}
