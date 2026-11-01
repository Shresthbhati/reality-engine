/** Compose / anchor / map-context pure logic. Run with: npm run test:contract */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  parseLatLonPaste, validateAnchor, draftFromAnchor, clampRadius, buildGeoContextPayload, geoErrorMessage,
  summariseCounts, buildComposePayload, newConnectorDraft, validateConnector, describeComposeResult,
  composeErrorMessage, isMapContextPlacement, RADIUS_MIN_M, RADIUS_MAX_M,
} from "../../src/lib/sites/compose-model.ts";
import { buildSceneTree, entityOrigin } from "../../src/lib/workspace/scene-tree-model.ts";
import { buildInsights } from "../../src/lib/workspace/insights-model.ts";

test("lat/lon paste accepts Google Maps style and rejects junk", () => {
  assert.deepEqual(parseLatLonPaste("22.5757, 88.4312"), { lat: 22.5757, lon: 88.4312 });
  assert.deepEqual(parseLatLonPaste(" -33.86 151.21 "), { lat: -33.86, lon: 151.21 });
  assert.equal(parseLatLonPaste("91, 10"), null);
  assert.equal(parseLatLonPaste("10, 181"), null);
  assert.equal(parseLatLonPaste("north, east"), null);
  assert.equal(parseLatLonPaste("22.5"), null);
});

test("validateAnchor: ranges, rotation normalised, empty reports errors", () => {
  const ok = validateAnchor({ lat: "22.5", lon: "88.4", rotation: "-90" });
  assert.deepEqual(ok.anchor, { lat: 22.5, lon: 88.4, rotation_deg: 270 });
  assert.equal(validateAnchor({ lat: "22.5", lon: "88.4", rotation: "" }).anchor?.rotation_deg, 0);
  const bad = validateAnchor({ lat: "95", lon: "", rotation: "x" });
  assert.equal(bad.anchor, null);
  assert.ok(bad.errors.lat && bad.errors.lon && bad.errors.rotation);
});

test("draftFromAnchor handles null and unset", () => {
  assert.deepEqual(draftFromAnchor(null), { lat: "", lon: "", rotation: "0" });
  assert.deepEqual(draftFromAnchor({ lat: null, lon: null, rotation_deg: null }), { lat: "", lon: "", rotation: "0" });
  assert.deepEqual(draftFromAnchor({ lat: 1, lon: 2, rotation_deg: 30 }), { lat: "1", lon: "2", rotation: "30" });
});

test("radius clamps to 100..600; geo payload carries name only when given", () => {
  assert.equal(clampRadius(5), RADIUS_MIN_M);
  assert.equal(clampRadius(9999), RADIUS_MAX_M);
  assert.equal(clampRadius(NaN), 250);
  const a = { lat: 1, lon: 2, rotation_deg: 0 };
  assert.deepEqual(buildGeoContextPayload(a, 300), { lat: 1, lon: 2, radius_m: 300 });
  assert.equal(buildGeoContextPayload(a, 300, " Home ").name, "Home");
});

test("geo errors map to plain messages", () => {
  assert.match(geoErrorMessage(502), /down/);
  assert.match(geoErrorMessage(429), /rate limiting/);
  assert.match(geoErrorMessage(404), /not available/);
  assert.equal(geoErrorMessage(422, "bad radius"), "bad radius");
});

test("summariseCounts sorts and drops zeros", () => {
  assert.equal(summariseCounts({ road: 3, building: 12, tree: 0 }), "12 building, 3 road");
  assert.equal(summariseCounts({}), "");
});

test("connector validation", () => {
  const ids = ["a", "b"];
  const c = newConnectorDraft("street");
  assert.equal(c.width, "6");
  assert.match(validateConnector(c, ids) ?? "", /two placements/);
  assert.match(validateConnector({ ...c, from: "a", to: "a" }, ids) ?? "", /different/);
  assert.match(validateConnector({ ...c, from: "a", to: "zz" }, ids) ?? "", /on this site/);
  assert.match(validateConnector({ ...c, from: "a", to: "b", width: "0" }, ids) ?? "", /Width/);
  assert.match(validateConnector({ ...c, from: "a", to: "b", width: "abc" }, ids) ?? "", /width/);
  assert.equal(validateConnector({ ...c, from: "a", to: "b" }, ids), null);
});

test("compose payload: name required, connectors mapped, omitted when none", () => {
  const ids = ["a", "b"];
  assert.equal(buildComposePayload("  ", "house", [], ids).payload, null);
  assert.deepEqual(buildComposePayload(" Lane ", "neighbourhood", [], ids).payload, { name: "Lane", kind: "neighbourhood" });
  const withC = buildComposePayload("X", "house", [{ kind: "driveway", from: "a", to: "b", width: "3.5" }], ids);
  assert.deepEqual(withC.payload?.connectors, [{ kind: "driveway", from_placement_id: "a", to_placement_id: "b", width_m: 3.5 }]);
  const badC = buildComposePayload("X", "house", [newConnectorDraft()], ids);
  assert.equal(badC.payload, null);
  assert.ok(badC.connectorErrors[0]);
});

test("compose result copy always states generated count", () => {
  assert.match(describeComposeResult({ source_count: 2, entity_count: 10, generated_count: 0 }), /0 generated/);
  assert.match(describeComposeResult({ source_count: 1, entity_count: 10, generated_count: 4 }), /1 source world · 10 entities \(6 from sources, 4 generated\)/);
  assert.match(composeErrorMessage(409, "busy"), /busy/);
  assert.match(composeErrorMessage(404), /not available/);
});

test("map-context placement is identified by label", () => {
  assert.equal(isMapContextPlacement({ label: "Map context" }), true);
  assert.equal(isMapContextPlacement({ label: "Kitchen" }), false);
});

// ---- badges + insights -------------------------------------------------------------------------

const ent = (id: string, type: string, cp: Record<string, unknown> = {}) =>
  ({ id, type, semantic_labels: [], custom_properties: cp }) as never;
const worldOf = (...es: ReturnType<typeof ent>[]) => ({ entities: Object.fromEntries(es.map((e: { id: string }) => [e.id, e])) }) as never;

test("entityOrigin: generated beats imported; plain = captured", () => {
  assert.equal(entityOrigin(ent("a", "wall")), "captured");
  assert.equal(entityOrigin(ent("a", "wall", { imported: true })), "imported");
  assert.equal(entityOrigin(ent("a", "wall", { source: "OpenStreetMap" })), "imported");
  assert.equal(entityOrigin(ent("a", "wall", { generated_by: "compose", imported: true })), "generated");
});

test("scene tree stamps badges only on imported/generated rows", () => {
  const w = worldOf(ent("w1", "wall"), ent("w2", "wall", { imported: true }), ent("w3", "wall", { generated_by: "connector" }));
  const flat: { entityId: string | null; badge?: string }[] = [];
  const walk = (ns: { entityId: string | null; badge?: string; children: never[] }[]) => ns.forEach((n) => (flat.push(n), walk(n.children)));
  walk(buildSceneTree(w) as never);
  const by = (id: string) => flat.find((n) => n.entityId === id);
  assert.equal(by("w1")?.badge, undefined);
  assert.equal(by("w2")?.badge, "Imported from map");
  assert.equal(by("w3")?.badge, "Generated (not captured)");
});

test("insights: one card each, only when such entities exist", () => {
  const none = buildInsights(worldOf(ent("w1", "wall")));
  assert.ok(!none.cards.some((c) => c.id === "imported-from-map" || c.id === "generated-placeholders"));
  const m = buildInsights(worldOf(ent("w2", "wall", { imported: true }), ent("w3", "wall", { imported: true }), ent("g", "wall", { generated_by: "x" })));
  const imp = m.cards.find((c) => c.id === "imported-from-map");
  const gen = m.cards.find((c) => c.id === "generated-placeholders");
  assert.match(imp?.label ?? "", /^2 objects imported from OpenStreetMap - approximate$/);
  assert.match(gen?.label ?? "", /^1 generated placeholder - capture this area to replace it$/);
  assert.equal(gen?.tone, "warning");
});
