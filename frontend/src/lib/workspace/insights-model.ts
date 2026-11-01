/**
 * Reconstruction insights built from REAL WorldIR data only. A card with no data behind it is not produced,
 * and "nothing needs attention" is only claimed when a diagnostics block was actually present.
 */
import type { Entity, WorldIR } from "@/types/worldir";
import { classifyEntity, entityOrigin } from "./scene-tree-model.ts";

export type InsightTone = "info" | "warning";

export interface InsightCard {
  id: string;
  tone: InsightTone;
  label: string;
  /** Longer explanation (tooltip / expandable). */
  detail?: string;
  /** Entities this card is about; clicking selects the first. */
  entityIds: string[];
  /** Data colour (0-255 rgb) for a swatch, wall-colour card only. */
  rgb?: [number, number, number];
}

export interface InsightsModel {
  cards: InsightCard[];
  /** Evidence ids linked from entities, in first-seen order. */
  photoIds: string[];
  /** True when the WorldIR carried diagnostics we could evaluate. */
  diagnosticsEvaluated: boolean;
  /** True when there is anything to show (cards or photos). */
  hasInsights: boolean;
}

type Rec = Record<string, unknown>;
const cp = (e: Entity): Rec => (e.custom_properties ?? {}) as Rec;
const rec = (v: unknown): Rec | null => (v && typeof v === "object" && !Array.isArray(v) ? (v as Rec) : null);
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

function rgbOf(v: unknown): [number, number, number] | null {
  if (!Array.isArray(v) || v.length < 3) return null;
  const c = v.slice(0, 3).map(Number);
  return c.every((x) => Number.isFinite(x) && x >= 0 && x <= 255) ? [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])] : null;
}

/** Messages a diagnostics block reports as problems. */
function diagnosticProblems(world: WorldIR): { evaluated: boolean; problems: string[] } {
  const md = (world.metadata ?? {}) as Rec;
  const problems: string[] = [];
  let evaluated = false;
  const warn = (v: unknown) => {
    if (Array.isArray(v)) for (const w of v) problems.push(typeof w === "string" ? w : JSON.stringify(w));
  };

  const dc = rec(md.dimension_consistency);
  if (dc) {
    evaluated = true;
    warn(dc.warnings);
    for (const r of Array.isArray(dc.rooms) ? (dc.rooms as Rec[]) : []) {
      warn(r.warnings);
      for (const c of Array.isArray(r.checks) ? (r.checks as Rec[]) : []) {
        if (c.ok === false) problems.push(String(c.detail ?? c.check ?? "dimension check failed"));
      }
    }
  }
  const rl = rec(md.room_layout);
  if (rl) {
    evaluated = true;
    warn(rl.warnings);
    for (const r of Array.isArray(rl.rooms) ? (rl.rooms as Rec[]) : []) {
      const cons = rec(r.consistency);
      if (cons) warn(cons.mismatches);
      if (Array.isArray(r.unobserved_gaps) && r.unobserved_gaps.length > 0) problems.push(`${plural(r.unobserved_gaps.length, "unobserved gap")} in a room layout`);
    }
  }
  const sem = rec(md.interior_semantics);
  if (sem) warn(sem.warnings);
  return { evaluated, problems };
}

export function buildInsights(world: WorldIR | null | undefined): InsightsModel {
  const empty: InsightsModel = { cards: [], photoIds: [], diagnosticsEvaluated: false, hasInsights: false };
  if (!world) return empty;
  const entities = Object.values(world.entities ?? {}).sort((a, b) => (a.id < b.id ? -1 : 1));
  if (entities.length === 0) return empty;
  const cards: InsightCard[] = [];
  const idsOf = (k: ReturnType<typeof classifyEntity>) => entities.filter((e) => classifyEntity(e) === k).map((e) => e.id);

  const windows = idsOf("window");
  const doors = idsOf("door");
  const furniture = idsOf("furniture");
  if (windows.length) cards.push({ id: "windows", tone: "info", label: `${plural(windows.length, "window")} detected`, entityIds: windows });
  if (doors.length) cards.push({ id: "doors", tone: "info", label: `${plural(doors.length, "door")} detected`, entityIds: doors });
  if (furniture.length) cards.push({ id: "furniture", tone: "info", label: `${plural(furniture.length, "furniture item")} detected`, entityIds: furniture });

  const imported = entities.filter((e) => entityOrigin(e) === "imported").map((e) => e.id);
  const generated = entities.filter((e) => entityOrigin(e) === "generated").map((e) => e.id);
  if (imported.length) {
    cards.push({
      id: "imported-from-map",
      tone: "info",
      label: `${plural(imported.length, "object")} imported from OpenStreetMap - approximate`,
      detail: "Positions and heights come from map data, not from your photos.",
      entityIds: imported,
    });
  }
  if (generated.length) {
    cards.push({
      id: "generated-placeholders",
      tone: "warning",
      label: `${plural(generated.length, "generated placeholder")} - capture this area to replace ${generated.length === 1 ? "it" : "them"}`,
      detail: "These shapes were imagined to join captured areas; they are not reconstructions.",
      entityIds: generated,
    });
  }

  // Wall colour: a room's wall_color_rgb, else the first wall estimate that exists.
  const walls = entities.filter((e) => classifyEntity(e) === "wall");
  let wallRgb: [number, number, number] | null = null;
  let wallIds: string[] = [];
  for (const e of entities) {
    if (classifyEntity(e) === "room") {
      const c = rgbOf(cp(e).wall_color_rgb);
      if (c) {
        wallRgb = c;
        wallIds = walls.map((w) => w.id);
        break;
      }
    }
  }
  if (!wallRgb) {
    for (const w of walls) {
      const c = rgbOf(cp(w).mean_color_rgb);
      if (c) {
        wallRgb = c;
        wallIds = [w.id];
        break;
      }
    }
  }
  if (wallRgb) cards.push({ id: "wall-colour", tone: "info", label: "Wall colour estimate available", entityIds: wallIds, rgb: wallRgb });

  // Dimension honesty: every stamped dimension that is not MEASURED.
  let dims = 0;
  let uncertain = 0;
  const uncertainIds = new Set<string>();
  for (const e of entities) {
    const d = rec(cp(e).dimensions);
    if (!d) continue;
    for (const v of Object.values(d)) {
      const s = rec(v)?.state;
      if (typeof s !== "string") continue;
      dims++;
      if (s.toUpperCase() !== "MEASURED") {
        uncertain++;
        uncertainIds.add(e.id);
      }
    }
  }
  if (uncertain > 0) {
    cards.push({
      id: "uncertain-dimensions",
      tone: "warning",
      label: `${uncertain} of ${plural(dims, "dimension")} estimated, not measured`,
      detail: "These sizes were inferred or reconstructed from the photographs; they are not scale-verified measurements.",
      entityIds: [...uncertainIds],
    });
  }

  // Missing geometry: walls whose observed fraction is below 1 or that list unobserved spans.
  const partial = walls.filter((w) => {
    const p = cp(w);
    const f = typeof p.observed_fraction === "number" ? p.observed_fraction : null;
    return (f !== null && f < 1) || (Array.isArray(p.unobserved_spans_m) && p.unobserved_spans_m.length > 0);
  });
  if (partial.length) {
    cards.push({
      id: "missing-geometry",
      tone: "warning",
      label: `${plural(partial.length, "wall")} partly unobserved`,
      detail: "Part of each of these walls was never seen in a photograph, so its geometry there is filled in.",
      entityIds: partial.map((w) => w.id),
    });
  }

  const { evaluated, problems } = diagnosticProblems(world);
  if (problems.length) {
    cards.push({
      id: "review",
      tone: "warning",
      label: `${plural(problems.length, "area")} ${problems.length === 1 ? "needs" : "need"} review`,
      detail: problems.slice(0, 6).join("\n"),
      entityIds: [],
    });
  }

  // Supporting photographs: evidence ids the entities themselves cite.
  const seen = new Set<string>();
  const photoIds: string[] = [];
  const addIds = (v: unknown) => {
    if (!Array.isArray(v)) return;
    for (const id of v) if (typeof id === "string" && id && !seen.has(id)) (seen.add(id), photoIds.push(id));
  };
  for (const e of entities) {
    const p = cp(e);
    addIds(p.supporting_evidence_ids);
    addIds(p.evidence_ids);
  }

  return { cards, photoIds, diagnosticsEvaluated: evaluated, hasInsights: cards.length > 0 || photoIds.length > 0 };
}

/** The strip's one-line state: "attention" when any warning card exists, "clear" only when diagnostics ran. */
export function insightsStatus(m: InsightsModel): "attention" | "clear" | "unknown" {
  if (m.cards.some((c) => c.tone === "warning")) return "attention";
  return m.diagnosticsEvaluated ? "clear" : "unknown";
}

/** Keep only photographs that actually exist as stored evidence, so the strip never shows (or counts) a dead link. */
export function withKnownPhotos(m: InsightsModel, knownEvidenceIds: ReadonlySet<string>): InsightsModel {
  const photoIds = m.photoIds.filter((id) => knownEvidenceIds.has(id));
  return { ...m, photoIds, hasInsights: m.cards.length > 0 || photoIds.length > 0 };
}
