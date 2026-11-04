/**
 * Pure viewport logic (no three.js): the three view modes, the dollhouse cutaway facing test, the metric scale bar
 * and the XYZ axis gizmo projection. Unit-tested under Node; three-scene.ts / World3DViewport.tsx only wire them up.
 */

import type { ShellKind, ShellRenderMode } from "./room-shell-model";

export type ViewMode = "realistic" | "wireframe" | "confidence";
export const VIEW_MODES: ReadonlyArray<{ id: ViewMode; label: string }> = [
  { id: "realistic", label: "Real" },
  { id: "wireframe", label: "Wireframe" },
  { id: "confidence", label: "Evidence" },
];

/** The three modes are materials over ONE canonical shell: realistic = measured colours/atlas, never beautified. */
export function viewModeToRender(mode: ViewMode | string): ShellRenderMode {
  return mode === "wireframe" ? "wireframe" : (mode === "confidence" || mode === "evidence") ? "confidence" : "solid";
}

export function renderToViewMode(mode: ShellRenderMode | string): ViewMode {
  // "both" is folded away: it no longer has its own tab, so it reads as realistic.
  return mode === "wireframe" ? "wireframe" : (mode === "confidence" || mode === "evidence") ? "confidence" : "realistic";
}

const MODE_KEY = "re.viewport.mode";

type StorageLike = Pick<Storage, "getItem" | "setItem">;

export function loadStoredMode(storage?: StorageLike | null): ViewMode {
  try {
    const v = (storage ?? (typeof localStorage !== "undefined" ? localStorage : null))?.getItem(MODE_KEY);
    if (v === "evidence") return "confidence";
    return v === "wireframe" || v === "confidence" || v === "realistic" ? v : "realistic";
  } catch {
    return "realistic";
  }
}

export function storeMode(mode: ViewMode, storage?: StorageLike | null): void {
  try {
    (storage ?? (typeof localStorage !== "undefined" ? localStorage : null))?.setItem(MODE_KEY, mode);
  } catch {
    /* storage blocked (private window): the choice just does not persist */
  }
}

// ---- Cutaway ("dollhouse") -------------------------------------------------------------------------------------

type V3 = [number, number, number];

const CEILING_KINDS = new Set<ShellKind>(["ceiling", "false_ceiling", "ceiling_step"]);
const WALL_KINDS = new Set<ShellKind>(["wall", "window_glass", "window_frame"]);

/**
 * Should this shell part be hidden so the interior is visible from `camera`?
 * Ceilings always; wall-borne parts (walls, window glass/frame) when they sit on the camera's side of the room
 * (horizontal outward direction from the room centre, dotted with the horizontal direction to the camera).
 * Floor and furniture never hide. A camera straight above the room hides no walls.
 * ponytail: outward = part centre minus room centre, so a very concave room can misjudge a wall; use the wall's
 * stored outward normal if that shows up.
 */
export function cutawayHides(kind: ShellKind, partCenter: V3, roomCenter: V3, camera: V3, threshold = 0.25): boolean {
  if (CEILING_KINDS.has(kind)) return true;
  if (!WALL_KINDS.has(kind)) return false;
  const ox = partCenter[0] - roomCenter[0], oz = partCenter[2] - roomCenter[2];
  const cx = camera[0] - roomCenter[0], cz = camera[2] - roomCenter[2];
  const ol = Math.hypot(ox, oz), cl = Math.hypot(cx, cz);
  if (ol < 1e-6 || cl < 1e-6) return false;
  return (ox * cx + oz * cz) / (ol * cl) > threshold;
}

export function centroid(positions: ArrayLike<number>): V3 {
  const n = Math.floor(positions.length / 3);
  if (!n) return [0, 0, 0];
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < n; i++) { x += positions[3 * i]; y += positions[3 * i + 1]; z += positions[3 * i + 2]; }
  return [x / n, y / n, z / n];
}

/** The floor centre closest (in plan) to a part: that floor's room is the part's room. */
export function nearestCenter(p: V3, centers: V3[]): V3 {
  let best = centers[0] ?? p, bd = Infinity;
  for (const c of centers) {
    const d = (c[0] - p[0]) ** 2 + (c[2] - p[2]) ** 2;
    if (d < bd) { bd = d; best = c; }
  }
  return best;
}

// ---- Scale bar -------------------------------------------------------------------------------------------------

/** Largest 1/2/5 x 10^n metre length whose bar is <= maxPx wide at `metersPerPx`. */
export function niceScale(metersPerPx: number, maxPx: number): { meters: number; px: number } | null {
  if (!(metersPerPx > 0) || !Number.isFinite(metersPerPx) || !(maxPx > 0)) return null;
  const raw = metersPerPx * maxPx;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = [5, 2, 1].map((k) => k * pow).find((v) => v <= raw) ?? pow;
  return { meters: m, px: m / metersPerPx };
}

export function formatMeters(m: number): string {
  return m >= 1 ? `${+m.toFixed(2)} m` : m >= 0.01 ? `${Math.round(m * 100)} cm` : `${Math.round(m * 1000)} mm`;
}

// ---- Axis gizmo ------------------------------------------------------------------------------------------------

/** Screen-plane direction (x right, y up, unit-length max) of the world X/Y/Z axes given the camera right/up vectors. */
export function axisGizmo(right: V3, up: V3): Record<"x" | "y" | "z", [number, number]> {
  const proj = (a: V3): [number, number] => [a[0] * right[0] + a[1] * right[1] + a[2] * right[2], a[0] * up[0] + a[1] * up[1] + a[2] * up[2]];
  return { x: proj([1, 0, 0]), y: proj([0, 1, 0]), z: proj([0, 0, 1]) };
}

// ---- Orthographic emulation ------------------------------------------------------------------------------------

export const PERSPECTIVE_FOV = 50;
/** ponytail: "orthographic" is a 3 degree-FOV perspective camera pulled back to the same framing; swap in a real
 *  THREE.OrthographicCamera if parallel lines must be exactly parallel. */
export const ORTHO_FOV = 3;

/** Distance that frames the same width after changing FOV. */
export function dollyForFov(distance: number, fromFov: number, toFov: number): number {
  const t = (f: number) => Math.tan((f * Math.PI) / 360);
  return (distance * t(fromFov)) / t(toFov);
}

// ---- CSS token colours -----------------------------------------------------------------------------------------

/** "#rrggbb" / "#rgb" (as read from a CSS custom property) -> sRGB 0-255, or null when not a hex colour. */
export function parseHexColor(v: string | null | undefined): V3 | null {
  const s = (v ?? "").trim();
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(s);
  if (!m) return null;
  const h = m[1].length === 3 ? m[1].split("").map((c) => c + c).join("") : m[1];
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as V3;
}

// ---- World up axis ---------------------------------------------------------------------------------------------

/** "z" when the world was compiled Z-up (metadata.compiled_from.up ~ +Z, e.g. the indoor synthetic generator), else "y". */
export function upAxisOf(world: { metadata?: unknown } | null | undefined): "y" | "z" {
  const up = ((world?.metadata as { compiled_from?: { up?: unknown } } | undefined)?.compiled_from?.up);
  if (Array.isArray(up) && up.length === 3 && up.every((n) => typeof n === "number")) {
    if (Math.abs(up[2]) > 0.9 && Math.abs(up[1]) < 0.5) return "z";
  }
  return "y";
}
