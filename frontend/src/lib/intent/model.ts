/**
 * World Intent — pure model (no imports, no I/O) so it can be unit-tested with `npm run test:contract`.
 *
 * Mirrors apps/api/routes_intent.py: the same 7 scales and 6 capabilities, and the resource-profile
 * shape. A profile field is never rendered as a number unless the API sent one; `null` is shown as
 * "Unknown" together with the API's own reason.
 */

export const SCALES = [
  { id: "interior_space", label: "Interior Space", blurb: "A room, floor, or indoor environment" },
  { id: "building", label: "Building", blurb: "A single building or structure" },
  { id: "street", label: "Street", blurb: "A street or road segment" },
  { id: "neighborhood", label: "Neighborhood", blurb: "Multiple streets and buildings" },
  { id: "locality", label: "Locality", blurb: "A part of a town or suburb" },
  { id: "district", label: "District", blurb: "A district or large urban area" },
  { id: "city", label: "City", blurb: "An entire city or metropolitan area" },
] as const;

export const CAPABILITIES = [
  { id: "geometry", label: "Geometry", blurb: "Reconstruct 3D shape and structure", example: "e.g. buildings, terrain, objects" },
  { id: "semantic_understanding", label: "Semantic Understanding", blurb: "Detect and classify real-world objects", example: "e.g. buildings, roads, vegetation, vehicles" },
  { id: "measurement", label: "Measurement", blurb: "Enable accurate measurements", example: "e.g. distances, heights, areas, volumes" },
  { id: "evidence_provenance", label: "Evidence Provenance", blurb: "Track source evidence for every part", example: "e.g. link images, videos, scans to the world" },
  { id: "temporal_state", label: "Temporal State", blurb: "Capture how the world changes over time", example: "e.g. construction, seasonal changes, traffic" },
  { id: "high_precision", label: "High Precision", blurb: "Use advanced processing for higher accuracy", example: "e.g. survey-grade alignment, fine details" },
] as const;

export type ScaleId = (typeof SCALES)[number]["id"];
export type CapabilityId = (typeof CAPABILITIES)[number]["id"];

/** The reference design's default selection (Geometry, Semantic, Measurement, Provenance). */
export const DEFAULT_CAPABILITIES: CapabilityId[] = [
  "geometry",
  "semantic_understanding",
  "measurement",
  "evidence_provenance",
];

export const WIZARD_STEPS = ["Intent", "Configure", "Review", "Run"] as const;

export function isScale(v: unknown): v is ScaleId {
  return SCALES.some((s) => s.id === v);
}

export function isCapability(v: unknown): v is CapabilityId {
  return CAPABILITIES.some((c) => c.id === v);
}

export function scaleLabel(id: ScaleId | null): string | null {
  return SCALES.find((s) => s.id === id)?.label ?? null;
}

/** Toggle a capability; the result is always in canonical (CAPABILITIES) order and de-duplicated. */
export function toggleCapability(current: readonly CapabilityId[], id: CapabilityId): CapabilityId[] {
  const set = new Set(current);
  if (set.has(id)) set.delete(id);
  else set.add(id);
  return CAPABILITIES.map((c) => c.id).filter((c) => set.has(c));
}

export interface IntentDraft {
  scale: ScaleId | null;
  capabilities: CapabilityId[];
  name: string;
}

/** Per-step blocking problems, as user-facing sentences. Empty array = the step may be left. */
export function stepProblems(step: number, draft: IntentDraft): string[] {
  const out: string[] = [];
  if (step >= 0 && draft.scale === null) out.push("Choose what you are trying to construct.");
  if (step >= 0 && draft.capabilities.length === 0) out.push("Select at least one capability.");
  if (step >= 1 && draft.name.trim().length === 0) out.push("Give the world a name.");
  return out;
}

// ---- resource profile (API -> view) ------------------------------------------------------------------

export type FieldKind = "detected" | "derived" | "unknown";

export interface ProfileField<T> {
  value: T | null;
  basis: string;
  kind: FieldKind;
  reason?: string;
  sample_size?: number;
}

export interface ResourceProfileDto {
  compute: ProfileField<{ cuda_gpu: boolean; colmap: boolean }>;
  memory: ProfileField<number>;
  storage: ProfileField<number>;
  time: ProfileField<number>;
  warnings: string[];
}

export interface ProfileRow {
  key: "compute" | "memory" | "storage" | "time";
  label: string;
  /** Human text; "Unknown" when the API had no value. Never a made-up figure. */
  text: string;
  known: boolean;
  kind: FieldKind;
  /** Why the value is what it is (the API's basis, or its reason when unknown). */
  note: string;
}

export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "Unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? v : v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "Unknown";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const mins = seconds / 60;
  if (mins < 90) return `${Math.round(mins)} min`;
  return `${(mins / 60).toFixed(1)} h`;
}

function computeText(v: { cuda_gpu: boolean; colmap: boolean }): string {
  return `${v.cuda_gpu ? "CUDA GPU detected" : "No CUDA GPU detected"} · COLMAP ${v.colmap ? "available" : "not found"}`;
}

export function toProfileRows(p: ResourceProfileDto): ProfileRow[] {
  const row = <T,>(key: ProfileRow["key"], label: string, f: ProfileField<T>, fmt: (v: T) => string): ProfileRow => {
    const known = f.value !== null && f.value !== undefined;
    return {
      key,
      label,
      known,
      kind: known ? f.kind : "unknown",
      text: known ? fmt(f.value as T) : "Unknown",
      note: known ? f.basis : f.reason ?? f.basis,
    };
  };
  return [
    row("compute", "Compute", p.compute, computeText),
    row("memory", "Memory", p.memory, (v) => `${formatBytes(v)} installed`),
    row("storage", "Storage (inputs)", p.storage, formatBytes),
    row("time", "Estimated Time", p.time, (v) => `~ ${formatDuration(v)}`),
  ];
}

/** Total size + video count of a file selection (videos are detected by extension / MIME). */
export function summarizeFiles(files: ReadonlyArray<{ name: string; size: number; type?: string }>): {
  file_count: number;
  total_bytes: number;
  video_count: number;
} {
  const isVideo = (f: { name: string; type?: string }) =>
    (f.type ?? "").startsWith("video/") || /\.(mp4|mov|m4v|avi|mkv|webm)$/i.test(f.name);
  return {
    file_count: files.length,
    total_bytes: files.reduce((s, f) => s + f.size, 0),
    video_count: files.filter(isVideo).length,
  };
}
