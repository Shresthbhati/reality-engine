import test from "node:test";
import assert from "node:assert/strict";
import {
  viewModeToRender, renderToViewMode, loadStoredMode, storeMode, cutawayHides, niceScale, axisGizmo, dollyForFov,
  parseHexColor, formatMeters, upAxisOf, PERSPECTIVE_FOV, ORTHO_FOV, VIEW_MODES,
} from "../../src/lib/viewport/view-model.ts";
import {
  partMaterial, confidenceRamp, DEFAULT_CONF_PALETTE, type ShellPart,
} from "../../src/lib/viewport/room-shell-model.ts";

const part = (over: Partial<ShellPart>): ShellPart => ({
  id: "p", kind: "wall", entityId: "e", positions: [], indices: [], color: [200, 194, 184], opacity: 1,
  colorMeasured: true, confidence: null, provenance: null, confidenceState: "unknown", ...over,
});

test("exactly three prominent modes; 'both' folds into realistic", () => {
  assert.deepEqual(VIEW_MODES.map((m) => m.id), ["realistic", "wireframe", "confidence"]);
  assert.equal(viewModeToRender("realistic"), "solid");
  assert.equal(viewModeToRender("wireframe"), "wireframe");
  assert.equal(viewModeToRender("confidence"), "confidence");
  assert.equal(renderToViewMode("both"), "realistic");
  assert.equal(renderToViewMode("confidence"), "confidence");
});

test("realistic keeps the measured colour untouched; confidence replaces it", () => {
  const p = part({ color: [10, 20, 30], confidence: 0.9, confidenceState: "scored" });
  assert.deepEqual(partMaterial(p, viewModeToRender("realistic")).color, [10, 20, 30]);
  assert.notDeepEqual(partMaterial(p, viewModeToRender("confidence")).color, [10, 20, 30]);
});

test("confidence: high is green, low is red, unknown stays the neutral grey token colour", () => {
  const hi = confidenceRamp(1), lo = confidenceRamp(0);
  assert.ok(hi[1] > hi[0] && hi[1] > hi[2], "high confidence reads green");
  assert.ok(lo[0] > lo[1] && lo[0] > lo[2], "low confidence reads red");
  const unknown = partMaterial(part({ confidence: null, confidenceState: "unknown" }), "confidence");
  assert.deepEqual(unknown.color, DEFAULT_CONF_PALETTE.unknown);
  // a score of 0 is data (red); null is not (grey): they must differ
  assert.notDeepEqual(partMaterial(part({ confidence: 0, confidenceState: "scored" }), "confidence").color, unknown.color);
  // a runtime (CSS token) palette is honoured, unknown included
  const pal = { low: [1, 2, 3], mid: [4, 5, 6], high: [7, 8, 9], unknown: [50, 50, 50] } as typeof DEFAULT_CONF_PALETTE;
  assert.deepEqual(partMaterial(part({}), "confidence", pal).color, [50, 50, 50]);
  assert.deepEqual(confidenceRamp(1, pal), [7, 8, 9]);
});

test("cutaway: ceilings always hide; walls hide only on the camera side; floor and furniture never", () => {
  const room: [number, number, number] = [0, 1, 0];
  const cam: [number, number, number] = [6, 5, 6];
  assert.equal(cutawayHides("ceiling", [0, 2.5, 0], room, cam), true);
  assert.equal(cutawayHides("false_ceiling", [0, 2.4, 0], room, cam), true);
  assert.equal(cutawayHides("wall", [2, 1, 0], room, cam), true, "+x wall faces a +x,+z camera");
  assert.equal(cutawayHides("wall", [0, 1, 2], room, cam), true, "+z wall faces it too");
  assert.equal(cutawayHides("wall", [-2, 1, 0], room, cam), false, "-x wall is behind the room");
  assert.equal(cutawayHides("wall", [0, 1, -2], room, cam), false);
  assert.equal(cutawayHides("window_glass", [2, 1, 0], room, cam), true, "windows follow their wall");
  assert.equal(cutawayHides("floor", [0, 0, 0], room, cam), false);
  assert.equal(cutawayHides("furniture", [1, 0.4, 1], room, cam), false);
  assert.equal(cutawayHides("wall", [2, 1, 0], room, [0, 9, 0]), false, "camera straight above hides no wall");
});

test("cutaway recomputes with the camera (orbit to the other side swaps the hidden walls)", () => {
  const room: [number, number, number] = [0, 1, 0];
  assert.equal(cutawayHides("wall", [-2, 1, 0], room, [-6, 5, -6]), true);
  assert.equal(cutawayHides("wall", [2, 1, 0], room, [-6, 5, -6]), false);
});

test("mode persistence survives broken storage", () => {
  const mem = new Map<string, string>();
  const ok = { getItem: (k: string) => mem.get(k) ?? null, setItem: (k: string, v: string) => void mem.set(k, v) };
  assert.equal(loadStoredMode(ok), "realistic");
  storeMode("confidence", ok);
  assert.equal(loadStoredMode(ok), "confidence");
  mem.set("re.viewport.mode", "garbage");
  assert.equal(loadStoredMode(ok), "realistic");
  const broken = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
  assert.equal(loadStoredMode(broken), "realistic");
  assert.doesNotThrow(() => storeMode("wireframe", broken));
});

test("scale bar picks a 1/2/5 length that fits", () => {
  const s = niceScale(0.01, 110)!; // 1.1 m fits in 110px
  assert.equal(s.meters, 1);
  assert.ok(s.px <= 110 && s.px > 55);
  assert.equal(niceScale(0.02, 110)!.meters, 2);
  assert.equal(niceScale(0.05, 110)!.meters, 5);
  assert.equal(niceScale(0, 110), null);
  assert.equal(niceScale(NaN, 110), null);
  assert.equal(formatMeters(2), "2 m");
  assert.equal(formatMeters(0.5), "50 cm");
});

test("axis gizmo projects world axes onto the camera right/up", () => {
  const g = axisGizmo([1, 0, 0], [0, 1, 0]); // looking down -z
  assert.deepEqual(g.x, [1, 0]);
  assert.deepEqual(g.y, [0, 1]);
  assert.deepEqual(g.z, [0, 0]);
});

test("orthographic emulation keeps framing; hex tokens parse", () => {
  const d = dollyForFov(5, PERSPECTIVE_FOV, ORTHO_FOV);
  assert.ok(d > 5 * 10);
  assert.ok(Math.abs(dollyForFov(d, ORTHO_FOV, PERSPECTIVE_FOV) - 5) < 1e-9);
  assert.deepEqual(parseHexColor(" #4caf62 "), [76, 175, 98]);
  assert.deepEqual(parseHexColor("#fff"), [255, 255, 255]);
  assert.equal(parseHexColor("rgba(1,2,3,0.5)"), null);
  assert.equal(parseHexColor(undefined), null);
});

test("up axis comes from the world's compiled_from.up; anything else is Y-up", () => {
  const w = (up: unknown) => ({ metadata: { compiled_from: { up } } });
  assert.equal(upAxisOf(w([0, 0, 1])), "z");
  assert.equal(upAxisOf(w([0, 1, 0])), "y");
  assert.equal(upAxisOf(w(undefined)), "y");
  assert.equal(upAxisOf(w(["a", 0, 1])), "y");
  assert.equal(upAxisOf(null), "y");
});
