/**
 * Pure logic for composing a Site into a world, anchoring it to the map and
 * loading map context. No React and no `@/` imports (contract tests load it directly).
 */

export const OSM_ATTRIBUTION = "(c) OpenStreetMap contributors (ODbL)";
export const MAP_COVERAGE_NOTE =
  "Map data does not include everything: street furniture coverage varies, so your photos add the rest.";
export const CONNECTOR_NOTE =
  "Connectors are imagined placeholders, shown as Generated, until you capture that area.";
export const MAP_CONTEXT_LABEL = "Map context";

export const RADIUS_MIN_M = 100;
export const RADIUS_MAX_M = 600;
export const RADIUS_DEFAULT_M = 250;

// ---- anchor ------------------------------------------------------------------------------------

export interface Anchor {
  lat: number;
  lon: number;
  rotation_deg: number;
}

export type MaybeAnchor = { lat: number | null; lon: number | null; rotation_deg: number | null };

export interface AnchorDraft {
  lat: string;
  lon: string;
  rotation: string;
}

export function draftFromAnchor(a: MaybeAnchor | null | undefined): AnchorDraft {
  if (!a || a.lat == null || a.lon == null) return { lat: "", lon: "", rotation: "0" };
  return { lat: String(a.lat), lon: String(a.lon), rotation: String(a.rotation_deg ?? 0) };
}

function num(raw: string): number | null {
  const t = raw.trim();
  if (t === "") return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** Accepts "22.5757, 88.4312" (Google Maps copy), with optional spaces/semicolon/tab. */
export function parseLatLonPaste(raw: string): { lat: number; lon: number } | null {
  const m = /^\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*$/.exec(raw);
  if (!m) return null;
  const lat = Number(m[1]);
  const lon = Number(m[2]);
  return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180 ? { lat, lon } : null;
}

export interface AnchorCheck {
  anchor: Anchor | null;
  errors: { lat?: string; lon?: string; rotation?: string };
}

export function validateAnchor(d: AnchorDraft): AnchorCheck {
  const errors: AnchorCheck["errors"] = {};
  const lat = num(d.lat);
  const lon = num(d.lon);
  const rot = d.rotation.trim() === "" ? 0 : num(d.rotation);
  if (lat === null) errors.lat = "Enter a latitude.";
  else if (lat < -90 || lat > 90) errors.lat = "Latitude must be between -90 and 90.";
  if (lon === null) errors.lon = "Enter a longitude.";
  else if (lon < -180 || lon > 180) errors.lon = "Longitude must be between -180 and 180.";
  if (rot === null) errors.rotation = "Rotation must be a number of degrees.";
  if (Object.keys(errors).length > 0) return { anchor: null, errors };
  const r = ((rot as number) % 360 + 360) % 360;
  return { anchor: { lat: lat as number, lon: lon as number, rotation_deg: Math.round(r * 100) / 100 }, errors };
}

export function clampRadius(v: number): number {
  if (!Number.isFinite(v)) return RADIUS_DEFAULT_M;
  return Math.min(RADIUS_MAX_M, Math.max(RADIUS_MIN_M, Math.round(v)));
}

// ---- map context -------------------------------------------------------------------------------

export interface GeoContextInput {
  lat: number;
  lon: number;
  radius_m: number;
  name?: string;
}

export function buildGeoContextPayload(anchor: Anchor, radiusM: number, name?: string): GeoContextInput {
  const n = name?.trim();
  return { lat: anchor.lat, lon: anchor.lon, radius_m: clampRadius(radiusM), ...(n ? { name: n } : {}) };
}

/** Plain-language message for a failed map-context call. */
export function geoErrorMessage(status: number, detail?: string): string {
  if (status === 404) return "Map context is not available on this server yet.";
  if (status === 429) return "The map service is rate limiting requests. Wait a minute, then retry.";
  if (status === 502 || status === 503 || status === 504) return "The map service is down right now. Retry in a moment.";
  return detail && detail.trim() ? detail : `Map context failed (HTTP ${status}).`;
}

export interface MapContextResult {
  counts_by_kind: Record<string, number>;
  estimated_count: number;
  element_count: number;
  attribution: string;
  source: "overpass" | "cache";
}

/** "12 buildings, 3 roads"; kinds sorted by count then name; zero counts omitted. */
export function summariseCounts(counts: Record<string, number>): string {
  return Object.entries(counts)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([k, n]) => `${n} ${k}`)
    .join(", ");
}

// ---- compose -----------------------------------------------------------------------------------

export type ComposeKind = "house" | "neighbourhood";
export type ConnectorKind = "driveway" | "street" | "corridor";

export interface ConnectorDraft {
  kind: ConnectorKind;
  from: string;
  to: string;
  width: string;
}

export interface ConnectorPayload {
  kind: ConnectorKind;
  from_placement_id: string;
  to_placement_id: string;
  width_m: number;
}

export const DEFAULT_CONNECTOR_WIDTH_M: Record<ConnectorKind, number> = { driveway: 3, street: 6, corridor: 1.5 };
export const CONNECTOR_WIDTH_RANGE_M = { min: 0.5, max: 30 } as const;

export function newConnectorDraft(kind: ConnectorKind = "driveway"): ConnectorDraft {
  return { kind, from: "", to: "", width: String(DEFAULT_CONNECTOR_WIDTH_M[kind]) };
}

export function validateConnector(c: ConnectorDraft, placementIds: readonly string[]): string | null {
  if (!c.from || !c.to) return "Pick two placements to connect.";
  if (!placementIds.includes(c.from) || !placementIds.includes(c.to)) return "Pick placements that are on this site.";
  if (c.from === c.to) return "A connector needs two different placements.";
  const w = num(c.width);
  if (w === null) return "Enter a width in metres.";
  if (w < CONNECTOR_WIDTH_RANGE_M.min || w > CONNECTOR_WIDTH_RANGE_M.max) {
    return `Width must be between ${CONNECTOR_WIDTH_RANGE_M.min} and ${CONNECTOR_WIDTH_RANGE_M.max} m.`;
  }
  return null;
}

export interface ComposeInput {
  name: string;
  kind: ComposeKind;
  connectors?: ConnectorPayload[];
}

export interface ComposeCheck {
  payload: ComposeInput | null;
  nameError: string | null;
  /** Index-aligned with the connector drafts; null = valid. */
  connectorErrors: (string | null)[];
}

export function buildComposePayload(
  name: string,
  kind: ComposeKind,
  connectors: ConnectorDraft[],
  placementIds: readonly string[],
): ComposeCheck {
  const nameError = name.trim() ? null : "Name the composed world.";
  const connectorErrors = connectors.map((c) => validateConnector(c, placementIds));
  if (nameError || connectorErrors.some((e) => e)) return { payload: null, nameError, connectorErrors };
  const payload: ComposeInput = { name: name.trim(), kind };
  if (connectors.length > 0) {
    payload.connectors = connectors.map((c) => ({
      kind: c.kind,
      from_placement_id: c.from,
      to_placement_id: c.to,
      width_m: Number(c.width),
    }));
  }
  return { payload, nameError: null, connectorErrors };
}

export interface ComposeResultLike {
  source_count: number;
  entity_count: number;
  generated_count: number;
}

/** Counts line; the generated share is always stated, even at zero, so it is never hidden. */
export function describeComposeResult(r: ComposeResultLike): string {
  const captured = Math.max(0, r.entity_count - r.generated_count);
  return `${r.source_count} source ${r.source_count === 1 ? "world" : "worlds"} · ${r.entity_count} entities (${captured} from sources, ${r.generated_count} generated)`;
}

export function composeErrorMessage(status: number, detail?: string): string {
  if (status === 404) return "Composing is not available on this server yet.";
  if (status === 409) return detail?.trim() || "This site cannot be composed right now (conflict).";
  if (status === 422) return detail?.trim() || "The site could not be composed with these settings.";
  return detail?.trim() || `Compose failed (HTTP ${status}).`;
}

// ---- grid marking ------------------------------------------------------------------------------

/** The map-context placement is the one added by "Load map context". */
export function isMapContextPlacement(p: { label: string }): boolean {
  return p.label === MAP_CONTEXT_LABEL;
}
