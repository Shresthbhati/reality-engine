/** T-5.1: a capture session holds a real track of timestamped fixes (N, not 1), never invented ones. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { addFix, distanceM, emptyTrack, fixNear, markSent, toLocationPoints, unsent, type GpsFix } from "../../src/lib/mobile/gps-track.ts";

const fix = (lat: number, lng: number, at: number, accuracyM: number | null = 8): GpsFix =>
  ({ lat, lng, accuracyM, altitudeM: null, heading: null, speedMps: null, at });
// ~1.1 m per 0.00001 degrees of latitude
const north = (m: number) => 51.5 + m / 111_320;

test("distance is metres on the ground", () => {
  assert.ok(Math.abs(distanceM(fix(51.5, 0, 0), fix(north(10), 0, 0)) - 10) < 0.1);
});

test("walking produces N fixes, one per meaningful move", () => {
  let t = emptyTrack();
  for (let i = 0; i < 10; i++) t = addFix(t, fix(north(i * 3), 0, i * 3000));
  assert.equal(t.fixes.length, 10);
});

test("standing still keeps a slow heartbeat, not a fix per reading", () => {
  let t = emptyTrack();
  for (let i = 0; i < 30; i++) t = addFix(t, fix(51.5, 0, i * 1000));     // 30 s, not moving
  assert.ok(t.fixes.length >= 2 && t.fixes.length <= 4, `fixes: ${t.fixes.length}`);
  assert.ok(t.dropped.tooClose > 20);
});

test("a fix with poor accuracy is counted and dropped, never stored", () => {
  const t = addFix(emptyTrack(), fix(51.5, 0, 0, 500));
  assert.equal(t.fixes.length, 0);
  assert.equal(t.dropped.inaccurate, 1);
});

test("a fix with no accuracy value is kept (the device did not say it was bad)", () => {
  assert.equal(addFix(emptyTrack(), fix(51.5, 0, 0, null)).fixes.length, 1);
});

test("a non-finite coordinate is refused", () => {
  assert.equal(addFix(emptyTrack(), fix(Number.NaN, 0, 0)).fixes.length, 0);
  assert.equal(addFix(emptyTrack(), fix(95, 0, 0)).fixes.length, 0);
});

test("the track is capped and the oldest fixes make way", () => {
  let t = emptyTrack();
  for (let i = 0; i < 50; i++) t = addFix(t, fix(north(i * 5), 0, i * 5000), { maxFixes: 20 });
  assert.equal(t.fixes.length, 20);
  assert.equal(t.fixes[0].at, 30 * 5000);
});

test("only unsent fixes are posted, and marking them sent moves the cursor", () => {
  let t = emptyTrack();
  for (let i = 0; i < 4; i++) t = addFix(t, fix(north(i * 5), 0, i * 5000));
  assert.equal(unsent(t).length, 4);
  t = markSent(t, 3);
  assert.equal(unsent(t).length, 1);
  t = addFix(t, fix(north(100), 0, 100_000));
  assert.equal(unsent(t).length, 2);
});

test("a cap that drops posted fixes keeps the cursor consistent", () => {
  let t = emptyTrack();
  for (let i = 0; i < 5; i++) t = addFix(t, fix(north(i * 5), 0, i * 5000), { maxFixes: 5 });
  t = markSent(t, 5);
  t = addFix(t, fix(north(50), 0, 50_000), { maxFixes: 5 });
  assert.equal(unsent(t).length, 1);
});

test("a photo is matched to the nearest fix in time, and to none when the skew is too large", () => {
  let t = emptyTrack();
  t = addFix(t, fix(51.5, 0, 0));
  t = addFix(t, fix(north(20), 0, 20_000));
  assert.equal(fixNear(t, 18_000)?.at, 20_000);
  assert.equal(fixNear(t, 500_000), null);
  assert.equal(fixNear(emptyTrack(), 0), null);
});

test("the API payload carries the timestamp, accuracy and the browser source", () => {
  const [p] = toLocationPoints([fix(51.5, -0.12, 1_700_000_000_000, 7)]);
  assert.equal(p.latitude, 51.5);
  assert.equal(p.longitude, -0.12);
  assert.equal(p.accuracy, 7);
  assert.equal(p.source, "browser_geolocation");
  assert.equal(p.captured_at, new Date(1_700_000_000_000).toISOString());
});
