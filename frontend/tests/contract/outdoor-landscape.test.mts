/**
 * Outdoor landscape + roofs: roof geometry, area polygons, hedges, trees (round/cone) and the new scene-tree groups.
 * fixtures/gardens-roofs-world.json is the entity output of geo.osm_context.osm_to_world over
 * tests/fixtures/osm/synthetic_gardens_roofs.json (regenerate it with that function when the importer changes).
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  RECT_FILL_MIN,
  buildOutdoor,
  meshBounds,
  orientedBox,
  parseRoof,
  partMesh,
  roofGeometry,
  type OutdoorPart,
  type RoofSpec,
} from "../../src/lib/viewport/outdoor-model.ts";
import { buildSceneTree, type SceneNode } from "../../src/lib/workspace/scene-tree-model.ts";

type E = { id: string; type: string; name?: string; confidence?: number; custom_properties: Record<string, unknown> } & Record<string, unknown>;
const ents = JSON.parse(readFileSync(new URL("./fixtures/gardens-roofs-world.json", import.meta.url), "utf8")) as E[];
const world = (es: E[]) => ({ id: "w", entities: Object.fromEntries(es.map((e) => [e.id, e])), metadata: {} }) as never;
const res = buildOutdoor(world(ents));
const part = (id: string) => res.parts.find((p) => p.entityId === id)!;

const RECT: Array<[number, number]> = [[0, 0], [10, 0], [10, 6], [0, 6]];
const roof = (shape: RoofSpec["shape"], height = 3, orientation: RoofSpec["orientation"] = "along"): RoofSpec => ({ shape, height, orientation });

function assertSound(g: { positions: number[]; indices: number[]; tint: number[] }) {
  assert.ok(g.positions.every(Number.isFinite), "no NaN / Infinity");
  assert.equal(g.positions.length / 3, g.tint.length, "one tint per vertex");
  assert.ok(g.indices.length % 3 === 0);
  assert.ok(g.indices.every((i) => Number.isInteger(i) && i >= 0 && i < g.positions.length / 3), "indices in range");
  for (let t = 0; t < g.indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => g.indices[t + k]);
    const v = (i: number) => [g.positions[3 * i], g.positions[3 * i + 1], g.positions[3 * i + 2]];
    const [pa, pb, pc] = [v(a), v(b), v(c)];
    const u = pb.map((x, k) => x - pa[k]), w = pc.map((x, k) => x - pa[k]);
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    assert.ok(Math.hypot(...n) > 1e-9, `triangle ${t / 3} is degenerate`);
  }
}

test("every roof shape on a rectangle gives sound geometry within the footprint and the roof height", () => {
  for (const shape of ["gabled", "hipped", "pyramidal", "skillion", "dome"] as const) {
    const g = roofGeometry(RECT, roof(shape));
    assert.equal(g.shape, shape);
    assert.equal(g.fallback, null);
    assert.ok(g.indices.length > 0, shape);
    assertSound(g);
    const hs = g.positions.filter((_, i) => i % 3 === 1);
    assert.ok(Math.max(...hs) <= 3 + 1e-9 && Math.min(...hs) >= -1e-9, `${shape} rise within 0..H`);
    assert.ok(Math.max(...hs) > 2.5, `${shape} reaches the ridge / apex`);
    const xs = g.positions.filter((_, i) => i % 3 === 0), zs = g.positions.filter((_, i) => i % 3 === 2);
    assert.ok(Math.min(...xs) >= -1e-6 && Math.max(...xs) <= 10 + 1e-6 && Math.min(...zs) >= -1e-6 && Math.max(...zs) <= 6 + 1e-6, `${shape} stays on the footprint`);
  }
});

test("gabled ridge runs along the long side by default and across it when asked", () => {
  const ridgeSpan = (o: RoofSpec["orientation"]) => {
    const g = roofGeometry(RECT, roof("gabled", 3, o));
    const top: number[][] = [];
    for (let i = 0; i < g.tint.length; i++) if (Math.abs(g.positions[3 * i + 1] - 3) < 1e-9) top.push([g.positions[3 * i], g.positions[3 * i + 2]]);
    const xs = top.map((p) => p[0]), zs = top.map((p) => p[1]);
    return [Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs)];
  };
  const [ax, az] = ridgeSpan("along");
  assert.ok(Math.abs(ax - 10) < 1e-6 && az < 1e-6);
  const [bx, bz] = ridgeSpan("across");
  assert.ok(bx < 1e-6 && Math.abs(bz - 6) < 1e-6);
});

test("a rotated rectangle is detected and roofed", () => {
  const a = 0.6, rot = RECT.map(([x, z]): [number, number] => [x * Math.cos(a) - z * Math.sin(a) + 40, x * Math.sin(a) + z * Math.cos(a) - 7]);
  const ob = orientedBox(rot)!;
  assert.ok(Math.abs(ob.a * ob.b * 4 - 60) < 1e-6);
  const g = roofGeometry(rot, roof("hipped"));
  assert.equal(g.shape, "hipped");
  assertSound(g);
});

test("non-rectangular footprints fall back: convex -> pyramidal, concave -> flat; never broken triangles", () => {
  const hexagon: Array<[number, number]> = [[0, 0], [8, 0], [12, 5], [8, 10], [0, 10], [-4, 5]];
  const L: Array<[number, number]> = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]];
  const h = roofGeometry(hexagon, roof("gabled"));
  assert.equal(h.shape, "pyramidal");
  assert.match(h.fallback ?? "", /rectangle/);
  assertSound(h);
  const l = roofGeometry(L, roof("hipped"));
  assert.equal(l.shape, "flat");
  assert.equal(l.indices.length, 0);
  assert.ok(l.fallback);
  const domeL = roofGeometry(L, roof("dome"));
  assert.equal(domeL.shape, "flat");
  assert.ok(RECT_FILL_MIN > 0.8 && RECT_FILL_MIN < 1);
});

test("degenerate polygons and heights produce no geometry rather than garbage", () => {
  const bad: Array<Array<[number, number]>> = [[], [[0, 0], [1, 1]], [[0, 0], [5, 0], [10, 0]], [[1, 1], [1, 1], [1, 1]], [[0, 0], [NaN, 0], [3, 3]]];
  for (const poly of bad) {
    const g = roofGeometry(poly, roof("gabled"));
    assert.equal(g.indices.length, 0);
    assert.ok(g.positions.every(Number.isFinite));
  }
  for (const height of [0, -2, NaN, Infinity]) assert.equal(roofGeometry(RECT, roof("gabled", height)).indices.length, 0);
  assert.equal(roofGeometry(RECT, roof("flat")).indices.length, 0);
  // partMesh on a degenerate roofed building still returns finite arrays
  const p = { ...part("osm-way-7004"), polygon: [[0, 0], [5, 0], [10, 0]] as Array<[number, number]> } as OutdoorPart;
  const m = partMesh(p, "y");
  assert.ok(m.positions.every(Number.isFinite));
});

test("parseRoof maps OSM shapes: approximations are named, unknowns and non-positive heights draw flat", () => {
  assert.deepEqual(parseRoof({ shape: "half-hipped", height_m: 2 }), { shape: "hipped", height: 2, orientation: "along", approximates: "half-hipped" });
  assert.equal(parseRoof({ shape: "gambrel", height_m: 2 })!.shape, "gabled");
  assert.equal(parseRoof({ shape: "round", height_m: 2 })!.shape, "flat");
  assert.equal(parseRoof({ shape: "zigzag", height_m: 2 })!.shape, "flat");
  assert.equal(parseRoof({ shape: "gabled", height_m: 0 })!.shape, "flat");
  assert.equal(parseRoof({ shape: "gabled", height_m: 2, orientation: "across" })!.orientation, "across");
  assert.equal(parseRoof(undefined), null);
});

test("buildings: roof raises the top above the eave, in the roof colour; flat roofs keep the old massing", () => {
  const hip = part("osm-way-7001"), gab = part("osm-way-7004"), apt = part("osm-way-7005"), odd = part("osm-way-7003");
  assert.equal(hip.roof!.shape, "hipped");
  assert.equal(hip.height, 6); // walls stop at the eave (2 levels * 3 m)
  assert.notDeepEqual(hip.color2, null);
  assert.equal(gab.roof!.shape, "gabled");
  assert.match(gab.note ?? "", /default for building=house \(not mapped\)/);
  assert.equal(gab.estimated, true);
  assert.equal(apt.roof, null);
  assert.equal(odd.roof, null);
  const top = (p: OutdoorPart) => meshBounds(partMesh(p, "y").positions)!.max[1];
  assert.ok(Math.abs(top(hip) - 9) < 1e-6, "eave 6 + roof 3");
  assert.ok(Math.abs(top(apt) - 18) < 1e-6, "flat roof: top is the wall height");
  const m = partMesh(hip, "y");
  assert.ok(m.tint.some((t) => t === 1) && m.tint.some((t) => t === 0));
  assertSound({ ...m, tint: m.tint });
  const mz = partMesh(hip, "z");
  assert.ok(Math.abs(meshBounds(mz.positions)!.max[2] - 9) < 1e-6, "Z-up frame keeps height on z");
});

test("areas are flat meshes just above the ground, layered, with distinct land-cover colours", () => {
  const garden = part("osm-way-7010"), bed = part("osm-way-7011"), wood = part("osm-way-7014"), pond = part("osm-way-7015");
  for (const p of [garden, bed, wood, pond]) {
    assert.equal(p.shape, "area");
    const m = partMesh(p, "y");
    assertSound(m);
    const bb = meshBounds(m.positions)!;
    assert.equal(bb.min[1], bb.max[1]);
    assert.ok(bb.max[1] > 0 && bb.max[1] < 0.1);
    assert.equal(p.imported, true);
  }
  assert.ok(bed.baseY > garden.baseY, "flowerbed sits over its garden");
  const cols = new Set([garden, bed, wood, pond].map((p) => p.color.join()));
  assert.equal(cols.size, 4);
  assert.deepEqual(res.diagnostics.skipped, []);
});

test("areas with no or untriangulable outline are skipped with a reason, not drawn", () => {
  const noPoly: E = { id: "a1", type: "vegetation", custom_properties: { imported: true, kind: "garden" } };
  const bow: E = { id: "a2", type: "vegetation", custom_properties: { imported: true, kind: "lawn", footprint_polygon_xyz_m: [[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]] } }; // collinear: no triangles
  const r = buildOutdoor(world([noPoly, bow]));
  assert.equal(r.parts.length, 0);
  assert.deepEqual(r.diagnostics.skipped.map((s) => s.reason), ["area has no footprint polygon", "area outline is not a simple polygon"]);
});

test("hedge is an extruded green strip; fence is thinner and a different colour", () => {
  const hedge = part("osm-way-7012"), fence = part("osm-way-7016");
  assert.equal(hedge.shape, "strip");
  assert.equal(hedge.group, "hedges");
  assert.equal(hedge.height, 1.5);
  assert.equal(hedge.width, 0.8);
  assert.ok(hedge.color[1] > hedge.color[0] && hedge.color[1] > hedge.color[2], "green");
  assert.ok(fence.width <= 0.12);
  assert.notDeepEqual(fence.color, hedge.color);
  const bb = meshBounds(partMesh(hedge, "y").positions)!;
  assert.ok(Math.abs(bb.max[1] - 1.5) < 1e-9);
});

test("trees: round vs cone canopy honour the entity radius and height; shrubs have no trunk; generated trees stay flagged", () => {
  const broad = part("osm-node-7021"), pine = part("osm-node-7022"), shrub = part("osm-node-7024");
  assert.equal(broad.canopy!.style, "round");
  assert.equal(broad.canopy!.radius, 4);
  assert.equal(pine.canopy!.style, "cone");
  const bb = (p: OutdoorPart) => meshBounds(partMesh(p, "y").positions)!;
  const b = bb(broad), c = bb(pine);
  assert.ok(Math.abs(b.max[1] - 12) < 0.01, "round canopy reaches the tagged height");
  assert.ok(Math.abs(c.max[1] - 15) < 1e-9, "cone apex at the tagged height");
  assert.ok(b.max[0] - b.min[0] > 7.5, "canopy diameter ~ 2 * radius");
  const pm = partMesh(pine, "y");
  assertSound(pm);
  assert.ok(pm.tint.some((t) => t === 1), "trunk tinted");
  assert.ok(!partMesh(shrub, "y").tint.some((t) => t === 1), "shrub has no trunk");
  const gen = res.parts.filter((p) => p.generated);
  assert.ok(gen.length > 20);
  assert.ok(gen.every((p) => !p.imported && p.shape === "tree" && p.group === "trees" && (p.confidence ?? 1) <= 0.3));
  assert.ok(res.parts.filter((p) => p.imported).every((p) => !p.generated));
});

test("scene tree: new groups with counts and Imported / Generated badges", () => {
  const tree = buildSceneTree(world(ents));
  const label = (s: string) => tree.find((n) => n.label.startsWith(s))!;
  assert.equal(label("Gardens & lawns").label, "Gardens & lawns (1)");
  assert.equal(label("Flowerbeds").label, "Flowerbeds (1)");
  assert.equal(label("Hedges & fences").label, "Hedges & fences (2)");
  assert.equal(label("Water").label, "Water (1)");
  assert.equal(label("Woodland & parks").label, "Woodland & parks (1)");
  const leaves = (n: SceneNode): SceneNode[] => (n.entityId ? [n] : n.children.flatMap(leaves));
  assert.ok(leaves(label("Gardens & lawns")).every((n) => n.badge === "Imported from map"));
  const trees = leaves(label("Trees"));
  assert.equal(trees.filter((n) => n.badge === "Generated (not captured)").length, res.parts.filter((p) => p.generated).length);
  assert.equal(trees.filter((n) => n.badge === "Imported from map").length, 5); // 3 tagged trees + shrub + tree row line
  assert.equal(res.diagnostics.byGroup.woodland, 1);
});
