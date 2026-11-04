/**
 * Pure models and helpers for Geographic Anchors (Blueprint 16 & Task 9.3).
 *
 * Keeps local metric coordinates completely separate from geographic anchors (WGS84).
 */

export interface GeoAnchor {
  latitude: number;
  longitude: number;
  altitude_m: number;
  heading_deg: number;
  crs: string;
  accuracy_m?: number;
  source: string;
}

export interface GeoJSONPolygonFeature {
  type: "Feature";
  geometry: {
    type: "Polygon";
    coordinates: number[][][]; // [[[lng, lat], ...]]
  };
  properties: {
    anchor_lat: number;
    anchor_lon: number;
    altitude_m: number;
    heading_deg: number;
    crs: string;
    source: string;
  };
}

export function formatGeoAnchorCoordinates(anchor: GeoAnchor): string {
  const latStr = `${Math.abs(anchor.latitude).toFixed(5)}° ${anchor.latitude >= 0 ? "N" : "S"}`;
  const lonStr = `${Math.abs(anchor.longitude).toFixed(5)}° ${anchor.longitude >= 0 ? "E" : "W"}`;
  return `${latStr}, ${lonStr}`;
}

export function isIndoorGpsAccuracy(accuracy_m?: number): boolean {
  // Indoors GPS is a noisy prior (>15m uncertainty) — per Blueprint 16 sec 3
  if (accuracy_m === undefined || accuracy_m === null) return false;
  return accuracy_m > 15.0;
}
