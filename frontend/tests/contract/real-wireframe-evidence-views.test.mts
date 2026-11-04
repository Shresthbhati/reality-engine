import test from "node:test";
import assert from "node:assert/strict";
import {
  VIEW_MODES,
  viewModeToRender,
  renderToViewMode,
  loadStoredMode,
  storeMode,
} from "../../src/lib/viewport/view-model.ts";
import {
  partMaterial,
  legendFor,
  resolveEvidenceState,
  EVIDENCE_PALETTE,
  type ShellPart,
} from "../../src/lib/viewport/room-shell-model.ts";

const makePart = (over: Partial<ShellPart>): ShellPart => ({
  id: "part-test",
  kind: "wall",
  entityId: "entity-test",
  positions: [0, 0, 0, 1, 0, 0, 1, 2, 0],
  indices: [0, 1, 2],
  color: [200, 194, 184],
  opacity: 1,
  colorMeasured: true,
  confidence: null,
  provenance: null,
  confidenceState: "unknown",
  ...over,
});

test("VIEW_MODES exposes Real, Wireframe, Evidence modes with Blueprint 14 labels", () => {
  assert.equal(VIEW_MODES.length, 3);
  assert.deepEqual(VIEW_MODES.map((m) => m.id), ["realistic", "wireframe", "confidence"]);
  assert.deepEqual(VIEW_MODES.map((m) => m.label), ["Real", "Wireframe", "Evidence"]);
});

test("viewModeToRender and renderToViewMode support realistic, wireframe, confidence and evidence alias", () => {
  assert.equal(viewModeToRender("realistic"), "solid");
  assert.equal(viewModeToRender("wireframe"), "wireframe");
  assert.equal(viewModeToRender("confidence"), "confidence");
  assert.equal(viewModeToRender("evidence"), "confidence");

  assert.equal(renderToViewMode("solid"), "realistic");
  assert.equal(renderToViewMode("wireframe"), "wireframe");
  assert.equal(renderToViewMode("confidence"), "confidence");
  assert.equal(renderToViewMode("both"), "realistic");
});

test("mode storage supports evidence alias seamlessly", () => {
  const store = new Map<string, string>();
  const storageMock = {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => void store.set(k, v),
  };
  store.set("re.viewport.mode", "evidence");
  assert.equal(loadStoredMode(storageMock), "confidence");

  storeMode("wireframe", storageMock);
  assert.equal(loadStoredMode(storageMock), "wireframe");
});

test("resolveEvidenceState maps provenance, custom_properties, and confidence into Blueprint 14 states", () => {
  // Direct evidence_state property
  assert.equal(resolveEvidenceState("OBSERVED", null, 0.9), "OBSERVED");
  assert.equal(resolveEvidenceState("PARTIAL", null, 0.5), "PARTIAL");
  assert.equal(resolveEvidenceState("OCCLUDED", null, null), "OCCLUDED");
  assert.equal(resolveEvidenceState("INFERRED", null, null), "INFERRED");
  assert.equal(resolveEvidenceState("PRIOR_GENERATED", null, null), "PRIOR_GENERATED");
  assert.equal(resolveEvidenceState("CONFLICTED", null, null), "CONFLICTED");

  // Provenance fallbacks
  assert.equal(resolveEvidenceState(null, "RECONSTRUCTED", 0.95), "OBSERVED");
  assert.equal(resolveEvidenceState(null, "ESTIMATED", null), "INFERRED");
  assert.equal(resolveEvidenceState(null, "SYNTHESIZED", null), "PRIOR_GENERATED");
  assert.equal(resolveEvidenceState(null, "UNOBSERVED", null), "OCCLUDED");

  // Numeric confidence fallbacks
  assert.equal(resolveEvidenceState(null, null, 0.85), "OBSERVED");
  assert.equal(resolveEvidenceState(null, null, 0.55), "PARTIAL");
  assert.equal(resolveEvidenceState(null, null, 0.2), "CONFLICTED");
  assert.equal(resolveEvidenceState(null, null, null), "UNKNOWN");
});

test("EVIDENCE_PALETTE defines non-clashing distinct RGB values for all states", () => {
  const states = Object.keys(EVIDENCE_PALETTE) as Array<keyof typeof EVIDENCE_PALETTE>;
  assert.equal(states.length, 7);
  const colorStrings = new Set<string>();
  for (const st of states) {
    const c = EVIDENCE_PALETTE[st];
    assert.equal(c.length, 3);
    const key = c.join(",");
    assert.ok(!colorStrings.has(key), `Duplicate color for state ${st}`);
    colorStrings.add(key);
  }
});

test("partMaterial renders real unadorned colors in solid mode and evidence states in evidence mode", () => {
  const realPart = makePart({
    color: [120, 150, 180],
    colorMeasured: true,
    evidenceState: "OBSERVED",
  });

  // In solid/real mode: exact measured color, no hatching
  const solidMat = partMaterial(realPart, "solid");
  assert.deepEqual(solidMat.color, [120, 150, 180]);
  assert.equal(solidMat.hatch, false);

  // In evidence mode: evidence palette green for OBSERVED
  const evMat = partMaterial(realPart, "confidence");
  assert.deepEqual(evMat.color, EVIDENCE_PALETTE.OBSERVED);
  assert.equal(evMat.hatch, false);

  // Inferred part is hatched in evidence mode
  const inferredPart = makePart({
    color: [200, 200, 200],
    colorMeasured: false,
    evidenceState: "INFERRED",
  });
  const infMat = partMaterial(inferredPart, "confidence");
  assert.deepEqual(infMat.color, EVIDENCE_PALETTE.INFERRED);
  assert.equal(infMat.hatch, true);

  // Conflicted part
  const conflictedPart = makePart({
    evidenceState: "CONFLICTED",
  });
  const confMat = partMaterial(conflictedPart, "confidence");
  assert.deepEqual(confMat.color, EVIDENCE_PALETTE.CONFLICTED);
  assert.equal(confMat.hatch, false);
});

test("legendFor(confidence) supplies comprehensive evidence categories", () => {
  const legend = legendFor("confidence");
  assert.ok(legend.length >= 7);
  const labels = legend.map((l) => l.label);
  assert.ok(labels.includes("Observed"));
  assert.ok(labels.includes("Partial"));
  assert.ok(labels.includes("Occluded"));
  assert.ok(labels.includes("Inferred"));
  assert.ok(labels.includes("Prior / Generated"));
  assert.ok(labels.includes("Conflicted"));
  assert.ok(labels.includes("Unknown"));
});
