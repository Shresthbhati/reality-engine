/**
 * Scene tree: built only from entities that exist, no fabricated groups, honest names, eye toggles.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ancestorKeys,
  buildSceneTree,
  classifyEntity,
  defaultExpandedKeys,
  filterTree,
  toggleHidden,
  visibilityOf,
  type SceneNode,
} from "../../src/lib/workspace/scene-tree-model.ts";
import { buildInsights, insightsStatus, withKnownPhotos } from "../../src/lib/workspace/insights-model.ts";

type E = Record<string, unknown> & { id: string; type: string };
const world = (entities: E[], metadata: Record<string, unknown> = {}) =>
  ({ id: "w", entities: Object.fromEntries(entities.map((e) => [e.id, e])), metadata }) as never;

const room: E = { id: "room-plane-000", type: "room", name: "room-plane-000", custom_properties: { wall_slab_ids: ["wall-1", "wall-2"] } };
const wall = (id: string, name?: string, extra: Record<string, unknown> = {}): E => ({ id, type: "wall", name, relationships: [], custom_properties: { room_id: "room-plane-000", ...extra } });
const find = (nodes: SceneNode[], pred: (n: SceneNode) => boolean): SceneNode | null => {
  for (const n of nodes) {
    if (pred(n)) return n;
    const c = find(n.children, pred);
    if (c) return c;
  }
  return null;
};

test("an empty or missing world yields an empty tree (nothing invented)", () => {
  assert.deepEqual(buildSceneTree(null), []);
  assert.deepEqual(buildSceneTree(world([])), []);
});

test("no building/level entity means no Building or Floor row", () => {
  const tree = buildSceneTree(world([room, wall("wall-1", "Wall 1")]));
  assert.equal(tree.length, 1);
  assert.equal(tree[0].kind, "room");
  assert.equal(find(tree, (n) => n.kind === "building"), null);
  assert.equal(find(tree, (n) => n.kind === "level"), null);
});

test("groups only exist when they have members, and carry counts", () => {
  const tree = buildSceneTree(
    world([
      room,
      wall("wall-1", "Wall 1"),
      wall("wall-2"),
      { id: "win-1", type: "window", name: "Window 001", custom_properties: { room_id: "room-plane-000" } },
    ]),
  );
  const labels = tree[0].children.map((c) => c.label);
  assert.deepEqual(labels, ["Walls (2)", "Windows (1)"]);
  assert.equal(find(tree, (n) => n.label.startsWith("Doors")), null);
  assert.equal(find(tree, (n) => n.label.startsWith("Lighting")), null);
  assert.equal(find(tree, (n) => n.label.startsWith("Furniture")), null);
});

test("names: real names are kept, id-like names fall back to numbered labels with the id as hint", () => {
  const tree = buildSceneTree(world([room, wall("struct-plane-001", "struct-plane-001"), wall("w-b", "Wall 1")]));
  const r = tree[0];
  assert.equal(r.label, "Room 001");
  assert.equal(r.hint, "room-plane-000");
  const walls = r.children[0].children;
  assert.deepEqual(walls.map((w) => w.label).sort(), ["Wall 1", "Wall 001"].sort());
  assert.equal(walls.find((w) => w.label === "Wall 001")?.hint, "struct-plane-001");
  assert.equal(walls.find((w) => w.label === "Wall 1")?.hint, undefined);
});

test("a single floor / ceiling is a plain leaf row under its room", () => {
  const tree = buildSceneTree(
    world([
      room,
      { id: "f", type: "floor", name: "floor-slab-room-plane-000", custom_properties: { room_id: "room-plane-000" } },
      { id: "c", type: "ceiling", custom_properties: { room_id: "room-plane-000" } },
    ]),
  );
  assert.deepEqual(tree[0].children.map((c) => c.label), ["Floor", "Ceiling"]);
  assert.ok(tree[0].children.every((c) => c.entityId !== null));
});

test("membership is resolved through relationships and host walls", () => {
  const tree = buildSceneTree(
    world([
      room,
      wall("wall-1", "Wall 1"),
      { id: "door-1", type: "door", name: "Door 01", custom_properties: { host_wall_id: "wall-1" } },
      { id: "plane-9", type: "wall", name: "Wall 9", relationships: [], custom_properties: {} },
      { id: "wall-attach", type: "wall", name: "Wall A", relationships: [{ kind: "attached_to", target_id: "plane-9" }], custom_properties: { room_id: "room-plane-000" } },
    ]),
  );
  const doors = find(tree, (n) => n.label === "Doors (1)");
  assert.ok(doors, "door resolved into the room via its host wall");
  assert.equal(find(tree, (n) => n.kind === "unassigned"), null, "plane-9 is reached via the reverse attached_to link");
});

test("entities that cannot be tied to a room go to Unassigned, never into a guessed room", () => {
  const tree = buildSceneTree(world([room, wall("wall-1", "Wall 1"), { id: "stray", type: "wall", name: "Stray wall" }]));
  const un = tree.find((n) => n.kind === "unassigned");
  assert.ok(un);
  assert.deepEqual(un.children.map((c) => c.label), ["Walls (1)"]);
  assert.equal(find([tree[0]], (n) => n.entityId === "stray"), null);
});

test("with no rooms at all, groups sit at the root without an Unassigned wrapper", () => {
  const tree = buildSceneTree(world([wall("a", "Wall A"), { id: "p", type: "window", name: "Window P" }]));
  assert.deepEqual(tree.map((n) => n.label), ["Walls (1)", "Windows (1)"]);
});

test("building > level > room hierarchy appears only from real entities", () => {
  const tree = buildSceneTree(
    world(
      [
        { id: "b", type: "building", name: "Block A" },
        { id: "s1", type: "storey", name: "Ground" },
        { ...room, parent_id: "s1" },
        wall("wall-1", "Wall 1"),
      ],
      { interior_space_graph: { levels: [{ level_id: "ghost-level", room_ids: ["ghost-room"] }] } },
    ),
  );
  assert.equal(tree[0].kind, "building");
  assert.equal(tree[0].children[0].kind, "level");
  assert.equal(tree[0].children[0].children[0].kind, "room");
  assert.equal(find(tree, (n) => n.entityId === "ghost-level" || n.entityId === "ghost-room"), null);
});

test("classification covers opening kinds and furniture candidates", () => {
  assert.equal(classifyEntity({ id: "a", type: "opening", custom_properties: { kind: "doorway" } } as never), "door");
  assert.equal(classifyEntity({ id: "a", type: "unknown", semantic_labels: ["furniture_candidate"] } as never), "furniture");
  assert.equal(classifyEntity({ id: "a", type: "lamp" } as never), "lighting");
  assert.equal(classifyEntity({ id: "a", type: "mystery" } as never), "other");
});

test("eye toggles: group hides/shows all children, state is derived", () => {
  const tree = buildSceneTree(world([room, wall("wall-1", "Wall 1"), wall("wall-2", "Wall 2")]));
  const walls = tree[0].children[0];
  const hidden0 = new Set<string>();
  assert.equal(visibilityOf(walls.leafIds, hidden0), "all");
  const hidden1 = toggleHidden(hidden0, walls.leafIds);
  assert.equal(visibilityOf(walls.leafIds, hidden1), "none");
  assert.equal(hidden0.size, 0, "input set is not mutated");
  const hidden2 = toggleHidden(hidden1, [walls.children[0].entityId!]);
  assert.equal(visibilityOf(walls.leafIds, hidden2), "some");
  assert.equal(visibilityOf(tree[0].leafIds, hidden1), "some", "room eye reflects partly-hidden children");
  assert.equal(visibilityOf(walls.leafIds, toggleHidden(hidden2, walls.leafIds)), "none", "partly hidden -> hide all");
});

test("search keeps matching rows and their ancestors; expansion helpers", () => {
  const tree = buildSceneTree(world([room, wall("wall-1", "North wall"), wall("wall-2", "South wall")]));
  const f = filterTree(tree, "north");
  assert.equal(f.length, 1);
  assert.deepEqual(f[0].children[0].children.map((c) => c.label), ["North wall"]);
  assert.deepEqual(filterTree(tree, "zzz"), []);
  assert.equal(filterTree(tree, "  ").length, tree.length);
  const exp = defaultExpandedKeys(tree);
  assert.ok(exp.has(tree[0].key) && !exp.has(tree[0].children[0].key), "rooms open, groups closed");
  assert.deepEqual(ancestorKeys(tree, "wall-1"), [tree[0].key, tree[0].children[0].key]);
  assert.deepEqual(ancestorKeys(tree, "nope"), []);
});

// ---- insights ----

test("insights: nothing real -> no cards, no photos, diagnostics not evaluated", () => {
  const m = buildInsights(world([room]));
  assert.deepEqual(m.cards, []);
  assert.deepEqual(m.photoIds, []);
  assert.equal(m.diagnosticsEvaluated, false);
  assert.equal(m.hasInsights, false);
  assert.equal(insightsStatus(m), "unknown", "never claims 'all clear' without diagnostics");
});

test("insights: detected counts, wall colour and photos come from entities", () => {
  const m = buildInsights(
    world([
      { ...room, custom_properties: { wall_color_rgb: [141, 118, 103] } },
      { id: "win-1", type: "window", custom_properties: {} },
      { id: "win-2", type: "window", custom_properties: {} },
      { id: "d", type: "door", custom_properties: {} },
      { id: "p", type: "wall", custom_properties: { supporting_evidence_ids: ["ev-1", "ev-2"] } },
      { id: "q", type: "wall", custom_properties: { supporting_evidence_ids: ["ev-2", "ev-3"] } },
    ]),
  );
  assert.deepEqual(m.cards.map((c) => c.label), ["2 windows detected", "1 door detected", "Wall colour estimate available"]);
  assert.deepEqual(m.cards[2].rgb, [141, 118, 103]);
  assert.deepEqual(m.photoIds, ["ev-1", "ev-2", "ev-3"]);
  assert.equal(m.cards.find((c) => c.id === "windows")!.entityIds.length, 2);
});

test("insights: estimated dimensions and unobserved walls are warnings; MEASURED is not", () => {
  const dim = (state: string) => ({ value: 1, state });
  const m = buildInsights(
    world([
      { id: "w1", type: "wall", custom_properties: { dimensions: { a: dim("MEASURED"), b: dim("ESTIMATED"), c: dim("RECONSTRUCTED") }, observed_fraction: 0.6 } },
      { id: "w2", type: "wall", custom_properties: { dimensions: { a: dim("MEASURED") }, observed_fraction: 1, unobserved_spans_m: [] } },
    ]),
  );
  const unc = m.cards.find((c) => c.id === "uncertain-dimensions");
  assert.equal(unc?.label, "2 of 4 dimensions estimated, not measured");
  assert.equal(unc?.tone, "warning");
  assert.deepEqual(unc?.entityIds, ["w1"]);
  assert.deepEqual(m.cards.find((c) => c.id === "missing-geometry")?.entityIds, ["w1"]);
  assert.equal(insightsStatus(m), "attention");
});

test("insights: 'clear' only when diagnostics ran and reported nothing", () => {
  const clean = buildInsights(world([room], { dimension_consistency: { rooms: [{ room_id: "r", checks: [{ ok: true }], warnings: [] }], warnings: [] } }));
  assert.equal(clean.diagnosticsEvaluated, true);
  assert.equal(insightsStatus(clean), "clear");
  assert.equal(clean.cards.find((c) => c.id === "review"), undefined);

  const bad = buildInsights(world([room], { dimension_consistency: { rooms: [{ room_id: "r", checks: [{ ok: false, detail: "area mismatch" }], warnings: ["tolerance exceeded"] }], warnings: [] } }));
  const review = bad.cards.find((c) => c.id === "review");
  assert.equal(review?.label, "2 areas need review");
  assert.match(review?.detail ?? "", /area mismatch/);
  assert.equal(insightsStatus(bad), "attention");
});

test("insights: photographs without stored evidence are dropped, never shown as dead links", () => {
  const m = buildInsights(world([{ id: "p", type: "wall", custom_properties: { supporting_evidence_ids: ["ev-1", "ev-2"] } }]));
  assert.equal(m.hasInsights, true);
  const kept = withKnownPhotos(m, new Set(["ev-2"]));
  assert.deepEqual(kept.photoIds, ["ev-2"]);
  const none = withKnownPhotos(m, new Set());
  assert.deepEqual(none.photoIds, []);
  assert.equal(none.hasInsights, false, "no cards and no real photos -> nothing to show");
});
