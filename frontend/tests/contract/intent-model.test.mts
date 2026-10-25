/**
 * World Intent pure model: validation helpers and the estimate -> view mapping (null never becomes a number).
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CAPABILITIES,
  SCALES,
  formatBytes,
  formatDuration,
  isCapability,
  isScale,
  stepProblems,
  summarizeFiles,
  toProfileRows,
  toggleCapability,
  type ResourceProfileDto,
} from "../../src/lib/intent/model.ts";

test("there are exactly the 7 scales and 6 capabilities the API accepts", () => {
  assert.deepEqual(SCALES.map((s) => s.id), ["interior_space", "building", "street", "neighborhood", "locality", "district", "city"]);
  assert.equal(CAPABILITIES.length, 6);
  assert.equal(isScale("city"), true);
  assert.equal(isScale("galaxy"), false);
  assert.equal(isCapability("geometry"), true);
  assert.equal(isCapability("teleportation"), false);
});

test("toggleCapability keeps canonical order and never duplicates", () => {
  let c = toggleCapability([], "measurement");
  c = toggleCapability(c, "geometry");
  assert.deepEqual(c, ["geometry", "measurement"]);
  assert.deepEqual(toggleCapability(c, "geometry"), ["measurement"]);
});

test("stepProblems blocks until scale, a capability, and (from step 1) a name exist", () => {
  const empty = { scale: null, capabilities: [], name: "" } as const;
  assert.equal(stepProblems(0, empty).length, 2);
  const ok = { scale: "building", capabilities: ["geometry"], name: "  " } as const;
  assert.equal(stepProblems(0, { ...ok, capabilities: [...ok.capabilities] }).length, 0);
  assert.equal(stepProblems(1, { ...ok, capabilities: [...ok.capabilities] }).length, 1);
  assert.equal(stepProblems(1, { scale: "building", capabilities: ["geometry"], name: "Site" }).length, 0);
});

const unknownTime: ResourceProfileDto = {
  compute: { value: { cuda_gpu: false, colmap: true }, basis: "detected", kind: "detected" },
  memory: { value: null, basis: "RAM could not be read", kind: "unknown", reason: "host RAM unavailable" },
  storage: { value: 1024 * 1024 * 5, basis: "inputs only", kind: "derived" },
  time: { value: null, basis: "no previous runs on this machine", kind: "unknown", reason: "no previous runs" },
  warnings: [],
};

test("null fields render as Unknown with the API's reason, never a number", () => {
  const rows = toProfileRows(unknownTime);
  const mem = rows.find((r) => r.key === "memory")!;
  const time = rows.find((r) => r.key === "time")!;
  assert.equal(mem.text, "Unknown");
  assert.equal(mem.known, false);
  assert.equal(mem.note, "host RAM unavailable");
  assert.equal(time.text, "Unknown");
  assert.equal(time.note, "no previous runs");
  assert.equal(rows.find((r) => r.key === "storage")!.text, "5.0 MB");
  assert.match(rows.find((r) => r.key === "compute")!.text, /No CUDA GPU detected · COLMAP available/);
});

test("a derived time is formatted and keeps its basis", () => {
  const rows = toProfileRows({ ...unknownTime, time: { value: 7200, basis: "median over 3 runs", kind: "derived", sample_size: 3 } });
  const time = rows.find((r) => r.key === "time")!;
  assert.equal(time.text, "~ 2.0 h");
  assert.equal(time.kind, "derived");
  assert.equal(time.note, "median over 3 runs");
});

test("formatters", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatDuration(30), "30 s");
  assert.equal(formatDuration(600), "10 min");
  assert.equal(formatBytes(-1), "Unknown");
});

test("summarizeFiles counts videos by MIME or extension", () => {
  const s = summarizeFiles([
    { name: "a.jpg", size: 10, type: "image/jpeg" },
    { name: "b.MOV", size: 100 },
    { name: "c.bin", size: 1, type: "video/mp4" },
  ]);
  assert.deepEqual(s, { file_count: 3, total_bytes: 111, video_count: 2 });
});
