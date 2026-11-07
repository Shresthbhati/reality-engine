/**
 * GPS track for a capture session (plan T-5.1): the session holds N timestamped fixes, not one.
 *
 * Pure and dependency-free so it is testable under `node --test`. Fixes come from `navigator.geolocation.watchPosition`.
 * Nothing is invented: a fix with poor accuracy or a non-finite coordinate is counted and dropped, a device that gives no accuracy is
 * not treated as inaccurate, and a photo is matched to a fix only when one was taken close enough in time.
 */

export interface GpsFix {
  lat: number;
  lng: number;
  accuracyM: number | null;
  altitudeM: number | null;
  heading: number | null;
  speedMps: number | null;
  /** epoch milliseconds the device took the fix */
  at: number;
}

export interface GpsTrack {
  fixes: GpsFix[];
  /** how many leading fixes have already been posted to the session */
  sent: number;
  dropped: { inaccurate: number; tooClose: number; invalid: number };
}

export interface TrackOptions {
  minIntervalMs: number;
  minMoveM: number;
  /** a standing-still capture still keeps one fix per this long */
  heartbeatMs: number;
  maxAccuracyM: number;
  maxFixes: number;
}

export const DEFAULT_TRACK: TrackOptions = { minIntervalMs: 2000, minMoveM: 1.5, heartbeatMs: 15_000, maxAccuracyM: 100, maxFixes: 2000 };

export const emptyTrack = (): GpsTrack => ({ fixes: [], sent: 0, dropped: { inaccurate: 0, tooClose: 0, invalid: 0 } });

export function distanceM(a: Pick<GpsFix, "lat" | "lng">, b: Pick<GpsFix, "lat" | "lng">): number {
  const R = 6_371_000;
  const rad = (d: number) => (d * Math.PI) / 180;
  const dLat = rad(b.lat - a.lat);
  const dLng = rad(b.lng - a.lng);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
}

export function addFix(track: GpsTrack, fix: GpsFix, options: Partial<TrackOptions> = {}): GpsTrack {
  const o = { ...DEFAULT_TRACK, ...options };
  const valid = Number.isFinite(fix.lat) && Number.isFinite(fix.lng) && Number.isFinite(fix.at) && Math.abs(fix.lat) <= 90 && Math.abs(fix.lng) <= 180;
  if (!valid) return { ...track, dropped: { ...track.dropped, invalid: track.dropped.invalid + 1 } };
  if (fix.accuracyM !== null && fix.accuracyM > o.maxAccuracyM) return { ...track, dropped: { ...track.dropped, inaccurate: track.dropped.inaccurate + 1 } };
  const last = track.fixes[track.fixes.length - 1];
  if (last) {
    const dt = fix.at - last.at;
    if (dt < o.minIntervalMs || (distanceM(last, fix) < o.minMoveM && dt < o.heartbeatMs)) {
      return { ...track, dropped: { ...track.dropped, tooClose: track.dropped.tooClose + 1 } };
    }
  }
  let fixes = [...track.fixes, fix];
  let sent = track.sent;
  if (fixes.length > o.maxFixes) {
    const drop = fixes.length - o.maxFixes;
    fixes = fixes.slice(drop);
    sent = Math.max(0, sent - drop);
  }
  return { ...track, fixes, sent };
}

export const unsent = (track: GpsTrack): GpsFix[] => track.fixes.slice(track.sent);

export const markSent = (track: GpsTrack, count: number): GpsTrack => ({ ...track, sent: Math.min(track.fixes.length, track.sent + count) });

/** The fix nearest in time to `atMs`, or null when none was taken within `maxSkewMs` (a photo is never given a distant position). */
export function fixNear(track: GpsTrack, atMs: number, maxSkewMs = 30_000): GpsFix | null {
  let best: GpsFix | null = null;
  for (const f of track.fixes) {
    if (Math.abs(f.at - atMs) <= maxSkewMs && (best === null || Math.abs(f.at - atMs) < Math.abs(best.at - atMs))) best = f;
  }
  return best;
}

export interface LocationPointPayload {
  latitude: number;
  longitude: number;
  altitude?: number;
  accuracy?: number;
  heading?: number;
  speed?: number;
  source: string;
  captured_at: string;
}

/** The body of `POST /api/sessions/{id}/locations`. */
export function toLocationPoints(fixes: GpsFix[]): LocationPointPayload[] {
  return fixes.map((f) => ({
    latitude: f.lat,
    longitude: f.lng,
    ...(f.altitudeM !== null ? { altitude: f.altitudeM } : {}),
    ...(f.accuracyM !== null ? { accuracy: f.accuracyM } : {}),
    ...(f.heading !== null ? { heading: f.heading } : {}),
    ...(f.speedMps !== null ? { speed: f.speedMps } : {}),
    source: "browser_geolocation",
    captured_at: new Date(f.at).toISOString(),
  }));
}
