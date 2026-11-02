/**
 * Inspector evidence model (plan V2): shapes what a WorldIR entity actually records about its own evidence.
 * Pure (no React / three.js) so it is unit-testable under Node. Anything the entity does not carry is reported as
 * `null` / "not recorded"; nothing is guessed or defaulted to a better state.
 */

export const NOT_RECORDED = "not recorded";

export type DimState = "MEASURED" | "RECONSTRUCTED" | "ESTIMATED" | "VERIFIED";
const DIM_STATES: readonly string[] = ["MEASURED", "RECONSTRUCTED", "ESTIMATED", "VERIFIED"];

export interface DimensionEvidence {
  key: string;
  value: number;
  unit: string | null;
  /** null = the dimension carries no state (rendered as "not recorded") */
  state: DimState | null;
  basis: string | null;
  uncertaintyM: number | null;
}

export interface ColourEvidence {
  /** "measured" only when the appearance says so AND carries an rgb; everything else is "not measured" */
  state: "measured" | "not_measured";
  rgb: [number, number, number] | null;
  basis: string;
  coverage: number | null;
  viewsUsed: string[];
  hasTexture: boolean;
  notes: string[];
}

export interface EntityEvidence {
  entityId: string;
  kind: string;
  isTarget: boolean;
  dimensions: DimensionEvidence[];
  confidence: number | null;
  provenance: string | null;
  colour: ColourEvidence;
  /** photo / view ids that supported the colour measurement; empty = not recorded */
  supportingViews: string[];
}

type Props = Record<string, unknown>;
const isRec = (v: unknown): v is Props => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string | null => (typeof v === "string" && v.trim() !== "" ? v : null);

/** Entity kinds the inspector shows an evidence block for. */
export const EVIDENCE_KINDS = ["wall", "door", "window", "floor", "ceiling", "opening", "surface"] as const;

export function isEvidenceTarget(type: unknown): boolean {
  return typeof type === "string" && (EVIDENCE_KINDS as readonly string[]).includes(type.toLowerCase());
}

export function buildDimensionEvidence(cp: Props | undefined): DimensionEvidence[] {
  const dims = cp?.dimensions;
  if (!isRec(dims)) return [];
  const out: DimensionEvidence[] = [];
  for (const [key, raw] of Object.entries(dims)) {
    if (!isRec(raw)) continue;
    const value = num(raw.value);
    if (value === null) continue;
    const s = typeof raw.state === "string" ? raw.state.toUpperCase() : "";
    out.push({
      key,
      value,
      unit: str(raw.unit),
      state: DIM_STATES.includes(s) ? (s as DimState) : null,
      basis: str(raw.basis),
      uncertaintyM: num(raw.uncertainty_m),
    });
  }
  return out;
}

export function buildColourEvidence(cp: Props | undefined): ColourEvidence {
  const app = cp?.appearance;
  const a = isRec(app) ? app : {};
  const rgbRaw = a.color_rgb;
  const rgb =
    Array.isArray(rgbRaw) && rgbRaw.length === 3 && rgbRaw.every((n) => typeof n === "number" && Number.isFinite(n))
      ? (rgbRaw as [number, number, number])
      : null;
  const measured = a.color_state === "measured" && rgb !== null;
  const views = Array.isArray(a.views_used) ? a.views_used.map(String) : [];
  return {
    state: measured ? "measured" : "not_measured",
    rgb: measured ? rgb : null,
    basis: measured ? `projected from ${views.length} photo view(s)` : "colour not measured",
    coverage: num(a.coverage),
    viewsUsed: views,
    hasTexture: isRec(a.texture) && !!str(a.texture.uri),
    notes: Array.isArray(a.notes) ? a.notes.map(String) : [],
  };
}

export function buildEntityEvidence(entity: {
  id: string;
  type?: string;
  confidence?: unknown;
  provenance?: unknown;
  custom_properties?: unknown;
}): EntityEvidence {
  const cp = isRec(entity.custom_properties) ? entity.custom_properties : undefined;
  const colour = buildColourEvidence(cp);
  return {
    entityId: entity.id,
    kind: String(entity.type ?? "").toLowerCase(),
    isTarget: isEvidenceTarget(entity.type),
    dimensions: buildDimensionEvidence(cp),
    confidence: num(entity.confidence),
    provenance: str(entity.provenance),
    colour,
    supportingViews: colour.viewsUsed,
  };
}

/** Display helpers: one place decides how a missing value reads. */
export const fmtDimension = (d: DimensionEvidence): string =>
  `${Number.isInteger(d.value) ? d.value : d.value.toFixed(3)}${d.unit ? ` ${d.unit}` : ""}`;
export const fmtUncertainty = (m: number | null): string => (m === null ? NOT_RECORDED : `±${m.toFixed(3)} m`);
export const fmtConfidence = (c: number | null): string => (c === null ? NOT_RECORDED : `${Math.round(c * 100)}%`);
