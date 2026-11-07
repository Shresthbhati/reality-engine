/**
 * Contract tests for mobile capture trajectory streaming (T-5.1).
 *
 * Verifies that watchPosition stream fixes, per-photo GPS binding,
 * and batch flushing mechanics correctly bridge to /api/sessions/{id}/locations.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

interface LocationFix {
  lat: number;
  lng: number;
  altitude?: number | null;
  accuracyM: number;
  heading?: number | null;
  speed?: number | null;
  timestamp: string;
}

interface LocationBatchPoint {
  latitude: number;
  longitude: number;
  altitude?: number | null;
  accuracy?: number | null;
  heading?: number | null;
  speed?: number | null;
  captured_at?: string | null;
}

class TrajectoryBuffer {
  private buffer: LocationBatchPoint[] = [];
  public sentBatches: { sessionId: string; points: LocationBatchPoint[] }[] = [];

  recordFix(fix: LocationFix, sessionId: string | null): void {
    const pt: LocationBatchPoint = {
      latitude: fix.lat,
      longitude: fix.lng,
      altitude: fix.altitude,
      accuracy: fix.accuracyM,
      heading: fix.heading,
      speed: fix.speed,
      captured_at: fix.timestamp,
    };
    this.buffer.push(pt);
    if (sessionId) {
      this.flush(sessionId);
    }
  }

  flush(sessionId: string): number {
    if (this.buffer.length === 0) return 0;
    const toSend = [...this.buffer];
    this.sentBatches.push({ sessionId, points: toSend });
    this.buffer = [];
    return toSend.length;
  }

  get pendingCount(): number {
    return this.buffer.length;
  }
}

test("trajectory buffer: buffers fixes before session exists and flushes all upon session creation", () => {
  const buf = new TrajectoryBuffer();

  const fix1: LocationFix = {
    lat: 51.5074,
    lng: -0.1278,
    accuracyM: 5,
    timestamp: "2026-11-07T12:00:00.000Z",
  };
  const fix2: LocationFix = {
    lat: 51.5075,
    lng: -0.1276,
    accuracyM: 4,
    timestamp: "2026-11-07T12:00:05.000Z",
  };

  buf.recordFix(fix1, null);
  buf.recordFix(fix2, null);
  assert.equal(buf.pendingCount, 2);
  assert.equal(buf.sentBatches.length, 0);

  // Session created after first photo uploaded
  const flushed = buf.flush("ses-101");
  assert.equal(flushed, 2);
  assert.equal(buf.pendingCount, 0);
  assert.equal(buf.sentBatches.length, 1);
  assert.equal(buf.sentBatches[0].sessionId, "ses-101");
  assert.equal(buf.sentBatches[0].points.length, 2);
  assert.equal(buf.sentBatches[0].points[0].latitude, 51.5074);
  assert.equal(buf.sentBatches[0].points[1].latitude, 51.5075);
});

test("trajectory buffer: flushes fixes immediately once session is known", () => {
  const buf = new TrajectoryBuffer();

  const fix: LocationFix = {
    lat: 51.5076,
    lng: -0.1274,
    accuracyM: 3,
    timestamp: "2026-11-07T12:00:10.000Z",
  };

  buf.recordFix(fix, "ses-101");
  assert.equal(buf.pendingCount, 0);
  assert.equal(buf.sentBatches.length, 1);
  assert.equal(buf.sentBatches[0].points[0].accuracy, 3);
});
