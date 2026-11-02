import { test } from "node:test";
import assert from "node:assert/strict";
import {
  readConfidence, meanConfidence, buildConfidenceFacets, TONE_TOKEN,
} from "../../src/lib/inspector/confidence-model.ts";

test("no value is unknown: neutral token, null ratio, never a colour or number", () => {
  for (const v of [undefined, null, "0.9", NaN, Infinity, {}]) {
    const r = readConfidence(v);
    assert.deepEqual(r, { label: "Not assessed", tone: "unknown", ratio: null });
  }
  assert.equal(TONE_TOKEN.unknown, "var(--conf-unknown)");
  assert.equal(TONE_TOKEN.high, "var(--conf-high)");
  assert.equal(TONE_TOKEN.low, "var(--conf-low)");
});

test("zero is a real Low reading, not unknown", () => {
  assert.deepEqual(readConfidence(0), { label: "Low", tone: "low", ratio: 0 });
});

test("thresholds and clamping", () => {
  assert.equal(readConfidence(0.75).tone, "high");
  assert.equal(readConfidence(0.7499).tone, "medium");
  assert.equal(readConfidence(0.45).tone, "medium");
  assert.equal(readConfidence(0.449).tone, "low");
  assert.equal(readConfidence(1.7).ratio, 1);
  assert.equal(readConfidence(-2).ratio, 0);
});

test("meanConfidence ignores non-numbers and returns null for nothing", () => {
  assert.equal(meanConfidence([]), null);
  assert.equal(meanConfidence([undefined, "x"]), null);
  assert.equal(meanConfidence([0.5, undefined, 1]), 0.75);
});

test("facets: only recorded values get a tone; uncertainty.confidence is never read", () => {
  const wall = { id: "w", type: "wall", confidence: 0.95, uncertainty: { confidence: 0 } } as never;
  const f = buildConfidenceFacets(wall, {});
  assert.deepEqual(f.map((x) => x.key), ["geometry", "dimensions", "materials"]);
  assert.equal(f[0].reading.tone, "high");
  assert.equal(f[1].reading.tone, "unknown");
  assert.equal(f[2].reading.ratio, null);
});

test("room facets aggregate real child confidences; absent children stay unknown", () => {
  const room = { id: "r", type: "room", confidence: 0.99, relationships: [] };
  const ents = {
    r: room,
    d: { id: "d", type: "door", confidence: 0.4, custom_properties: { room_id: "r" } },
    win: { id: "win", type: "window", confidence: 1, parent_id: "r" },
    o: { id: "o", type: "unknown", confidence: 0.9, semantic_labels: ["furniture_candidate"], custom_properties: { room_id: "r" } },
    other: { id: "x", type: "door", confidence: 0, custom_properties: { room_id: "elsewhere" } },
  };
  const f = buildConfidenceFacets(room, ents);
  const by = Object.fromEntries(f.map((x) => [x.key, x.reading]));
  assert.equal(by.openings.ratio, 0.7); // mean(0.4, 1)
  assert.equal(by.openings.tone, "medium");
  assert.equal(by.objects.ratio, 0.9);
  const empty = buildConfidenceFacets(room, { r: room });
  assert.equal(empty.find((x) => x.key === "objects")!.reading.tone, "unknown");
});

test("openings join a room through the wall that hosts them", () => {
  const room = { id: "r", type: "room", confidence: 0.9 };
  const ents = {
    r: room,
    w: { id: "w", type: "wall", confidence: 0.95, custom_properties: { room_id: "r" } },
    win: { id: "win", type: "window", confidence: 1, relationships: [{ target_id: "w", kind: "part_of" }] },
    d: { id: "d", type: "door", confidence: 0.2, custom_properties: { host_wall_id: "w" } },
    stray: { id: "stray", type: "door", confidence: 0.9, custom_properties: { host_wall_id: "elsewhere" } },
  };
  const open = buildConfidenceFacets(room, ents).find((x) => x.key === "openings")!.reading;
  assert.equal(open.ratio, 0.6); // mean(1, 0.2)
  assert.equal(open.tone, "medium");
});
