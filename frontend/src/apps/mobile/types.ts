'use client';

/**
 * Mobile field app — real types shared by store, capture, quality, sync.
 *
 * Mirrors the backend (Python `evidence/`, `MultiSourceSession`, WorldIR)
 * disciplines without inventing data:
 * - content-addressed ids (sha256 of frame bytes, `frame:<hex16>` prefix —
 *   matches the backend's `folder:<hash16>` identity rule)
 * - provenance on every record
 * - uncertainty is explicit; UNKNOWN is a real state, never defaulted
 */

export type FrameVerdict = 'USEFUL' | 'REDUNDANT' | 'REJECTED' | 'UNASSESSED';

export type QualityReason =
  | 'sharpness_ok'
  | 'sharpness_low'
  | 'exposure_ok'
  | 'overexposed'
  | 'underexposed'
  | 'duplicate_of_previous'
  | 'similar_to_previous'
  | 'no_reference_available';

export interface FrameQuality {
  /** Laplacian variance over the grayscale downsample. Higher = sharper. */
  sharpness: number;
  /** Mean luma 0..1. */
  exposure: number;
  /** Fraction of pixels clipped at either end. */
  clippedRatio: number;
  /** 0..1 structural difference vs previous accepted frame (or null). */
  diffVsPrevious: number | null;
}

/**
 * Whether the frame's pixels are still on this device. `missing` is a real,
 * visible state (vault failure or metadata-only import) — never hidden.
 */
export type LocalCopyState = 'vault' | 'missing';

export interface FrameRecord {
  /** `frame:<first16 hex of sha256(bytes)>` */
  frameId: string;
  /** sha256 hex of the raw blob — provenance + content identity. */
  contentSha256: string;
  /** Object URL for display (client-only, not synced). Empty when no local copy. */
  previewUrl: string;
  capturedAt: string; // ISO
  width: number;
  height: number;
  /** Byte count of original blob. */
  bytes: number;
  /** MIME type of the stored blob (drives the file extension on the desktop). */
  mime: string;
  verdict: FrameVerdict;
  quality: FrameQuality;
  reasons: QualityReason[];
  /** Optional real device telemetry if the browser exposes it. */
  telemetry: {
    /** Camera focal length in 35mm-equivalent mm, if reported. */
    focal35mm?: number;
    geolocation?: { latitude: number; longitude: number; accuracyM: number };
    /** Device orientation compass heading in degrees, if reported. */
    headingDeg?: number;
  };
  provenance: {
    origin: 'device_camera' | 'file_import';
    device: string; // userAgent camera label or file name
    capturedBy: 'operator';
    ingestion: 'mobile_local';
  };
  /** Capture task this frame was shot for (desktop-requested), if any. */
  taskId: string | null;
  /** Is the pixel payload still available on this device? */
  localCopy: LocalCopyState;
}

export type SessionStatus = 'CAPTURING' | 'READY_TO_SYNC' | 'SYNCED' | 'ARCHIVED';

export interface FieldSession {
  /** uuid v4 — created on device, stable across sync. */
  sessionId: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  status: SessionStatus;
  /** Free-text capture intent (site / structure / facade / detail). */
  intent: string;
  frames: FrameRecord[];
}

export interface SyncState {
  /** Null = no ingest server configured (honest blocked state). */
  lastSyncAt: string | null;
  lastSyncError: string | null;
}

export type CaptureTaskStatus = 'OPEN' | 'DONE' | 'CANCELLED';

/**
 * A desktop-requested capture task (`re.mobile-capture-task/v1`).
 *
 * Tasks are derived on the engine side from real evidence state (for example a
 * frame this app rejected for motion blur, or a session with no usable coverage)
 * and are never invented for display. The mobile flow is: receive task → shoot
 * for it → the frame carries `taskId` → the bundle reports the outcome back.
 */
export interface CaptureTask {
  taskId: string;
  sessionId: string;
  createdAt: string;
  createdBy: string;
  /** Evidence that motivated the request (may be empty for coverage tasks). */
  targetFrameIds: string[];
  /** Verbatim reason codes from the mobile triage that motivated the request. */
  reasons: QualityReason[];
  /** Human, actionable guidance derived from those reasons. */
  guidance: string;
  status: CaptureTaskStatus;
  /** Frames captured against this task on this device. */
  completedByFrameIds: string[];
  completedAt: string | null;
}

/** Task state as it travels back to the engine inside a session bundle. */
export interface CaptureTaskOutcome {
  taskId: string;
  status: CaptureTaskStatus;
  frameIds: string[];
  completedAt: string | null;
}

/** Aggregates derived ONLY from real frame records. No placeholders. */
export interface SessionSummary {
  total: number;
  useful: number;
  rejected: number;
  redundant: number;
  unassessed: number;
  bytes: number;
  /** Frames whose pixels are no longer on this device (cannot be handed off). */
  missingLocalCopy: number;
  /** Real spread of headings (deg) when telemetry exists; else null. */
  headingSpreadDeg: number | null;
  /** Real GPS bounding box when telemetry exists; else null. */
  gps: { minLat: number; maxLat: number; minLon: number; maxLon: number } | null;
  timeSpanSec: number | null;
}
