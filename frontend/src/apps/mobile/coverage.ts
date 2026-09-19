'use client';

/**
 * Mobile coverage & guidance — derived from real frame telemetry only.
 *
 * The field app does not have a 3D reconstruction, so coverage is computed
 * from what the device actually reported at capture time:
 *   - compass heading (viewpoint direction)
 *   - GPS position (geographic extent)
 *
 * When telemetry is absent, coverage is UNKNOWN — never invented.
 *
 * Sectors are compass octants (N, NE, E, SE, S, SW, W, NW). A sector is
 * covered when at least one USEFUL frame was captured facing that octant.
 * Redundant/rejected frames do not count toward coverage.
 */

import type { FieldSession, FrameRecord } from './types';

const OCTANTS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'] as const;

const OCTANT_BOUNDS = [
  [337.5, 360],
  [0, 22.5],
  [67.5, 112.5],
  [112.5, 157.5],
  [157.5, 202.5],
  [202.5, 247.5],
  [247.5, 292.5],
  [292.5, 337.5],
] as const;

/** Which octant a compass heading falls in. */
export function octantOf(headingDeg: number): string {
  const h = ((headingDeg % 360) + 360) % 360;
  for (let i = 0; i < OCTANTS.length; i++) {
    const [min, max] = OCTANT_BOUNDS[i];
    if (h >= min && h < max) return OCTANTS[i];
  }
  return 'N';
}

/** Per-sector coverage state derived from real USEFUL frames. */
export interface CoverageSector {
  octant: string;
  /** USEFUL frames captured facing this octant. */
  usefulCount: number;
  /** All frames captured facing this octant (including redundant/rejected). */
  totalCount: number;
  covered: boolean;
}

/** Coverage model computed from a real session. No estimation, no filling. */
export interface CoverageModel {
  /** Null when no frame has heading telemetry — coverage is genuinely unknown. */
  bySector: CoverageSector[] | null;
  /** Number of octants with at least one USEFUL frame. */
  coveredOctants: number;
  /** Total octants (always 8). */
  totalOctants: number;
  /** Fraction of octants covered by USEFUL frames; null when telemetry absent. */
  sectorCoverage: number | null;
  /** True when every octant has at least one USEFUL frame. */
  fullyCovered: boolean;
  /** GPS bounding box from real frame telemetry; null when no GPS. */
  gpsBounds: { minLat: number; maxLat: number; minLon: number; maxLon: number } | null;
  /** Number of frames with GPS; null when no GPS at all. */
  gpsFrameCount: number | null;
}

/** Build a CoverageModel from a real session. No estimation, no filling. */
export function computeCoverage(session: FieldSession): CoverageModel {
  const usefulFrames = session.frames.filter((f) => f.verdict === 'USEFUL');
  const framesWithHeading = usefulFrames.filter(
    (f) => typeof f.telemetry.headingDeg === 'number',
  );

  if (framesWithHeading.length === 0) {
    return {
      bySector: null,
      coveredOctants: 0,
      totalOctants: 8,
      sectorCoverage: null,
      fullyCovered: false,
      gpsBounds: null,
      gpsFrameCount: null,
    };
  }

  const sectors: Record<string, CoverageSector> = {};
  for (const o of OCTANTS) {
    sectors[o] = { octant: o, usefulCount: 0, totalCount: 0, covered: false };
  }

  for (const f of framesWithHeading) {
    const h = f.telemetry.headingDeg;
    if (typeof h !== 'number') continue;
    const oct = octantOf(h);
    sectors[oct].usefulCount++;
    sectors[oct].covered = true;
  }

  for (const f of session.frames) {
    if (typeof f.telemetry.headingDeg === 'number') {
      const oct = octantOf(f.telemetry.headingDeg);
      sectors[oct].totalCount++;
    }
  }

  const bySector = OCTANTS.map((o) => sectors[o]);
  const coveredOctants = bySector.filter((s) => s.covered).length;

  const gpsFrames = session.frames.filter(
    (f) => f.telemetry.geolocation != null,
  ) as FrameRecord[];
  let gpsBounds: { minLat: number; maxLat: number; minLon: number; maxLon: number } | null = null;
  if (gpsFrames.length > 0) {
    gpsBounds = {
      minLat: Math.min(...gpsFrames.map((f) => f.telemetry.geolocation!.latitude)),
      maxLat: Math.max(...gpsFrames.map((f) => f.telemetry.geolocation!.latitude)),
      minLon: Math.min(...gpsFrames.map((f) => f.telemetry.geolocation!.longitude)),
      maxLon: Math.max(...gpsFrames.map((f) => f.telemetry.geolocation!.longitude)),
    };
  }

  return {
    bySector,
    coveredOctants,
    totalOctants: 8,
    sectorCoverage: coveredOctants / 8,
    fullyCovered: coveredOctants === 8,
    gpsBounds,
    gpsFrameCount: gpsFrames.length > 0 ? gpsFrames.length : null,
  };
}

/** Coverage gaps — octants with no USEFUL frames. */
/**
 * Coverage gaps — octants with no USEFUL frames. Ordered by likelihood of being useful next.
 */
export interface CoverageGap {
  octant: string;
  /** Why this gap matters / what to do. */
  guidance: string;
}

/** Generate guidance from a CoverageModel. Empty when coverage is unknown or complete. */
export function coverageGuidance(model: CoverageModel): CoverageGap[] {
  if (model.bySector === null) return [];
  if (model.fullyCovered) return [];

  const gapMap: Record<string, string> = {
    N: 'Capture the north-facing facade — move to the opposite side and shoot back.',
    NE: 'Cover the northeast corner — step to the NE and vary your angle.',
    E: 'Capture the east side — move around to face east.',
    SE: 'Cover the southeast — a different elevation or corner may help.',
    S: 'Capture the south-facing surface — often the sunlit side, check exposure.',
    SW: 'Cover the southwest — move to the SW corner for a new viewpoint.',
    W: 'Capture the west side — move around to face west.',
    NW: 'Cover the northwest corner — a different approach angle would help here.',
  };

  return model.bySector
    .filter((s) => !s.covered)
    .map((s) => ({ octant: s.octant, guidance: gapMap[s.octant] ?? `Capture the ${s.octant} side.` }));
}

/** Human-readable coverage summary line. Empty string when telemetry is absent. */
export function coverageSummaryLine(model: CoverageModel): string {
  if (model.bySector === null) return '';
  if (model.fullyCovered) {
    return `All 8 viewing directions covered`;
  }
  const missing = 8 - model.coveredOctants;
  return `${model.coveredOctants} of 8 viewing directions covered · ${missing} gap${missing === 1 ? '' : 's'} remaining`;
}

