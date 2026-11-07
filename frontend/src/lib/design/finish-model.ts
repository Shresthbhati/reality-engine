/**
 * Surface finishes (plan T-4b.2): wording and colour arithmetic only, no I/O, so it can be tested.
 * A finish is a design layer on a wall, floor or ceiling. The survey's own colour and photo texture are kept and shown in the
 * evidence view; nothing here edits them.
 */

export interface Swatch { id: string; name: string; rgb: [number, number, number] }
export interface FinishOptions { floor_materials: Swatch[]; paints: Swatch[]; note: string }

export interface DesignFinish {
  kind: "paint" | "material" | "colour";
  name: string;
  colour_rgb: [number, number, number];
  paint?: string;
  material?: string;
}

type EntityLike = { type?: string; custom_properties?: Record<string, unknown> | null };

export type SurfaceType = "wall" | "floor" | "ceiling";

export function surfaceType(entity: EntityLike | null | undefined): SurfaceType | null {
  const t = entity?.type;
  return t === "wall" || t === "floor" || t === "ceiling" ? t : null;
}

export function currentFinish(entity: EntityLike | null | undefined): DesignFinish | null {
  const f = entity?.custom_properties?.design_finish as Partial<DesignFinish> | null | undefined;
  const rgb = f?.colour_rgb;
  if (!f || !Array.isArray(rgb) || rgb.length !== 3 || !rgb.every((v) => Number.isInteger(v) && v >= 0 && v <= 255)) return null;
  return { kind: f.kind ?? "colour", name: f.name ?? "Custom colour", colour_rgb: rgb as [number, number, number], paint: f.paint, material: f.material };
}

const hex2 = (n: number) => n.toString(16).padStart(2, "0");

export function rgbToHex(rgb: readonly number[]): string {
  return `#${rgb.map((v) => hex2(Math.max(0, Math.min(255, Math.round(v))))).join("")}`;
}

/** "#aabbcc" (or "aabbcc") to [r,g,b]; null for anything else, never a silent black. */
export function hexToRgb(hex: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : null;
}

/** Which swatches the panel offers for this surface: floors get materials and paints, walls and ceilings only paints. */
export function offersMaterials(type: SurfaceType | null): boolean {
  return type === "floor";
}

export function finishSummary(f: DesignFinish | null): string {
  if (!f) return "Survey colour (no finish applied)";
  return f.kind === "material" ? `${f.name} (floor material)` : f.kind === "paint" ? `${f.name} (paint)` : `Custom colour ${rgbToHex(f.colour_rgb)}`;
}

export const FINISH_NOTE =
  "A finish is a design choice drawn in the normal view. The measured colour and photo texture are kept and still show in the evidence view; exports of the as-built survey do not change.";

export function applyAllLabel(type: SurfaceType | null, count: number): string | null {
  if (type === null || count < 2) return null;
  return `Apply to all ${count} ${type}s`;
}
