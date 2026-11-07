/**
 * Furniture design controls (plan T-4b.1): wording and arithmetic only, no I/O, so it can be tested.
 * Every design change is a saved version the designer can undo; the as-built survey (walls, floors, openings) is never edited here.
 */

export interface CatalogItem {
  id: string;
  name: string;
  category: string;
  length_m: number;
  width_m: number;
  height_m: number;
  basis: string;
}

type EntityLike = { type?: string; custom_properties?: Record<string, unknown> | null; transform?: { position?: { x: number; y: number; z: number } } | null };

export function isFurniture(entity: EntityLike | null | undefined): boolean {
  return entity?.type === "furniture";
}

/** The world has a real-world scale only when metadata.scale.state says "metric"; a catalog size in metres means nothing otherwise. */
export function isMetricScale(metadata: unknown): boolean {
  const scale = (metadata as { scale?: { state?: unknown } } | null | undefined)?.scale;
  return String(scale?.state ?? "").toLowerCase() === "metric";
}

export function yawDegrees(entity: EntityLike | null | undefined): number {
  const yaw = entity?.custom_properties?.yaw_rad;
  return typeof yaw === "number" && Number.isFinite(yaw) ? normaliseDegrees((yaw * 180) / Math.PI) : 0;
}

/** Into (-180, 180], rounded to 0.1 degree for display. */
export function normaliseDegrees(deg: number): number {
  let d = deg % 360;
  if (d > 180) d -= 360;
  if (d <= -180) d += 360;
  return Math.round(d * 10) / 10;
}

export function degreesToRadians(deg: number): number {
  return (normaliseDegrees(deg) * Math.PI) / 180;
}

/** Parse a number typed into a field; null when it is empty or not a finite number (never silently 0). */
export function parseNumber(text: string): number | null {
  if (!text.trim()) return null;
  const n = Number(text);
  return Number.isFinite(n) ? n : null;
}

export function catalogLabel(item: CatalogItem): string {
  return `${item.name} - ${item.length_m.toFixed(2)} x ${item.width_m.toFixed(2)} x ${item.height_m.toFixed(2)} m`;
}

export function replaceBlockedReason(metricScale: boolean): string | null {
  return metricScale ? null : "This model has no real-world scale yet. Use Set scale with a known length first, so a catalog size means metres.";
}

export function deleteConfirmation(name: string): string {
  return `Delete ${name || "this piece"} from the model? You can undo it right after.`;
}

export const REPLACE_NOTE = "Catalog sizes are nominal, not measured. The piece is drawn from boxes; the survey is not changed.";
