/**
 * Capture-coverage presentation model (plan F1). Pure: turns the API's plan-view wall coverage into SVG geometry
 * and labels. Walls are numbered around the footprint ("wall 3 (between corner C and D)"), never by compass: the
 * API has no north reference and this model must not invent one.
 */

export type WallStatus = "seen" | "weak" | "not_seen";

export interface CoverageWall {
  index: number;
  label: string;
  between: [string, string];
  p0: [number, number];
  p1: [number, number];
  status: WallStatus;
  covered_fraction: number;
  multi_view_fraction: number;
  hint: string | null;
}

export interface CaptureCoverageDto {
  available: boolean;
  reason: string | null;
  provisional?: boolean;
  walls: CoverageWall[];
  corners?: { label: string; xy: [number, number] }[];
  cameras?: { id: string | null; xy: [number, number]; dir: [number, number] }[];
  hints: string[];
  summary?: Record<WallStatus, number>;
  notes?: string[];
}

export const STATUS_LABEL: Record<WallStatus, string> = {
  seen: "Seen",
  weak: "Seen weakly",
  not_seen: "Not seen",
};

/** Colour-blind-safe-ish palette; status is also carried by dash pattern and text, never colour alone. */
export const STATUS_STYLE: Record<WallStatus, { stroke: string; dash: string | undefined }> = {
  seen: { stroke: "var(--success)", dash: undefined },
  weak: { stroke: "var(--warning)", dash: "6 4" },
  not_seen: { stroke: "var(--error)", dash: "2 4" },
};

export interface MapGeometry {
  width: number;
  height: number;
  walls: { index: number; x1: number; y1: number; x2: number; y2: number; status: WallStatus; label: string; mx: number; my: number }[];
  corners: { label: string; x: number; y: number }[];
  cameras: { x: number; y: number; tx: number; ty: number }[];
}

/** Fit the plan into a width x height box (y flipped so the plan reads like a top-down map). */
export function mapGeometry(dto: CaptureCoverageDto, width = 320, height = 260, pad = 28): MapGeometry | null {
  if (!dto.available || dto.walls.length === 0) return null;
  const pts: [number, number][] = [...dto.walls.flatMap((w) => [w.p0, w.p1]), ...(dto.cameras ?? []).map((c) => c.xy)];
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1e-9);
  const s = Math.min(width - 2 * pad, height - 2 * pad) / span;
  const ox = (width - (maxX - minX) * s) / 2;
  const oy = (height - (maxY - minY) * s) / 2;
  const tx = (x: number) => ox + (x - minX) * s;
  const ty = (y: number) => height - (oy + (y - minY) * s);
  const arrow = 14;
  return {
    width,
    height,
    walls: dto.walls.map((w) => ({
      index: w.index,
      x1: tx(w.p0[0]), y1: ty(w.p0[1]), x2: tx(w.p1[0]), y2: ty(w.p1[1]),
      status: w.status, label: w.label,
      mx: tx((w.p0[0] + w.p1[0]) / 2), my: ty((w.p0[1] + w.p1[1]) / 2),
    })),
    corners: (dto.corners ?? []).map((c) => ({ label: c.label, x: tx(c.xy[0]), y: ty(c.xy[1]) })),
    cameras: (dto.cameras ?? []).map((c) => {
      const x = tx(c.xy[0]), y = ty(c.xy[1]);
      return { x, y, tx: x + c.dir[0] * arrow, ty: y - c.dir[1] * arrow };
    }),
  };
}

/** One-line status for the card header, from the API's own counts. */
export function coverageHeadline(dto: CaptureCoverageDto): string {
  if (!dto.available) return dto.reason ?? "Wall coverage is not available yet.";
  const s = dto.summary ?? { seen: 0, weak: 0, not_seen: 0 };
  if (s.weak === 0 && s.not_seen === 0) return `All ${dto.walls.length} walls were seen.`;
  const bits: string[] = [];
  if (s.not_seen) bits.push(`${s.not_seen} not seen`);
  if (s.weak) bits.push(`${s.weak} seen weakly`);
  return `${s.seen} of ${dto.walls.length} walls seen; ${bits.join(", ")}.`;
}
