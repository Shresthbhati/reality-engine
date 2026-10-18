/**
 * Apartment builder pure logic: templates, validation, request body, plan -> SVG, adjacency summary.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  APARTMENT_TEMPLATES,
  adjacencySummary,
  effectiveEntrance,
  templateAdjacency,
  nextRoomId,
  planToSvg,
  templateRows,
  toRequestBody,
  validateApartment,
  type ApartmentDraft,
  type ApartmentPlan,
} from "../../src/lib/procedural/apartment-model.ts";
import { apartmentSummary, isEditable, summarize } from "../../src/lib/procedural/generated-model.ts";

function draft(over: Partial<ApartmentDraft> = {}): ApartmentDraft {
  const rows = templateRows("2-bedroom");
  return {
    name: "flat", rows, entrance: effectiveEntrance(rows, ""), footprint: null,
    adjacency: [], seed: 0, ...over,
  };
}

test("templates fill rows only and return independent copies", () => {
  const a = templateRows("2-bedroom");
  assert.deepEqual(a.map((r) => r.id), ["living", "kitchen", "bedroom-1", "bedroom-2", "bathroom", "hallway"]);
  a[0].min_area = 99;
  assert.equal(templateRows("2-bedroom")[0].min_area, 18);
  assert.equal(templateRows("studio").length, 4);
  assert.equal(templateRows("3-bedroom").filter((r) => r.kind === "bedroom").length, 3);
  assert.deepEqual(templateRows("nope"), []);
});

test("entrance defaults to the hallway and survives only while it exists", () => {
  const rows = templateRows("2-bedroom");
  assert.equal(effectiveEntrance(rows, ""), "hallway");
  assert.equal(effectiveEntrance(rows, "kitchen"), "kitchen");
  assert.equal(effectiveEntrance(rows.filter((r) => r.id !== "kitchen"), "kitchen"), "hallway");
  assert.equal(effectiveEntrance([], "x"), "");
  assert.equal(nextRoomId(rows, "office"), "office");
  const withBed = [...rows, { id: "bedroom", kind: "bedroom" as const, min_area: 9, max_area: 12 }];
  assert.equal(nextRoomId(withBed, "bedroom"), "bedroom-3"); // -1 and -2 already taken
});

test("every template prefills required adjacency over its own rooms only; entrance is the hallway", () => {
  for (const t of APARTMENT_TEMPLATES) {
    const rows = templateRows(t.key);
    const ids = new Set(rows.map((r) => r.id));
    const adj = templateAdjacency(rows);
    assert.ok(adj.length >= 3, t.key);
    for (const a of adj) {
      assert.ok(ids.has(a.a) && ids.has(a.b), `${t.key}: ${a.a}-${a.b}`);
      assert.equal(a.level, "required");
    }
    const kindOf = (id: string) => rows.find((r) => r.id === id)!.kind;
    for (const r of rows.filter((x) => ["bedroom", "bathroom", "living"].includes(x.kind))) {
      assert.ok(adj.some((a) => a.a === "hallway" && a.b === r.id), `${t.key}: hallway-${r.id}`);
    }
    assert.ok(adj.some((a) => kindOf(a.a) === "living" && kindOf(a.b) === "kitchen"));
    assert.equal(effectiveEntrance(rows, ""), "hallway");
    assert.deepEqual(validateApartment({ name: "x", rows, entrance: "hallway", footprint: null, adjacency: adj, seed: 0 }), []);
  }
  assert.equal(templateAdjacency(templateRows("2-bedroom")).length, 5);
});

test("a template draft is valid", () => assert.deepEqual(validateApartment(draft()), []));

test("validation catches the documented problems", () => {
  const rows = templateRows("studio");
  assert.match(validateApartment(draft({ name: " " }))[0], /name/);
  assert.match(validateApartment(draft({ rows: [], entrance: "" })).join("|"), /at least one room/);
  const tooMany = Array.from({ length: 21 }, (_, i) => ({ id: `r${i}`, kind: "storage" as const, min_area: 1, max_area: 2 }));
  assert.match(validateApartment(draft({ rows: tooMany, entrance: "r0" })).join("|"), /limited to 20/);
  const bad = rows.map((r) => ({ ...r }));
  bad[0].min_area = 30; bad[0].max_area = 20;
  bad[1].min_area = 0;
  bad[2].max_area = NaN;
  const msgs = validateApartment(draft({ rows: bad })).join("|");
  assert.match(msgs, /living: min area \(30\) is larger than max area \(20\)/);
  assert.match(msgs, /kitchen: areas must be greater than 0/);
  assert.match(msgs, /bathroom: enter numbers/);
  const dup = [...rows, { ...rows[0] }];
  assert.match(validateApartment(draft({ rows: dup })).join("|"), /unique \(repeated: living\)/);
  assert.match(validateApartment(draft({ entrance: "ghost" })).join("|"), /entrance/);
  assert.match(validateApartment(draft({ footprint: { width: 0, depth: 5 } })).join("|"), /Footprint/);
  assert.match(validateApartment(draft({ footprint: { width: 5, depth: 101 } })).join("|"), /Footprint/);
  assert.match(validateApartment(draft({ seed: 1.5 })).join("|"), /Seed/);
});

test("adjacency validation: unknown room, self, duplicate (either order)", () => {
  const a = (x: string, y: string) => ({ a: x, b: y, level: "required" as const });
  assert.match(validateApartment(draft({ adjacency: [a("living", "ghost")] })).join("|"), /not in the list/);
  assert.match(validateApartment(draft({ adjacency: [a("living", "living")] })).join("|"), /itself/);
  assert.match(
    validateApartment(draft({ adjacency: [a("living", "kitchen"), a("kitchen", "living")] })).join("|"),
    /listed twice/,
  );
  assert.deepEqual(validateApartment(draft({ adjacency: [a("living", "kitchen")] })), []);
});

test("request body matches the API shape (arrays for footprint and adjacency)", () => {
  const body = toRequestBody(draft({
    name: " flat ", footprint: { width: 10, depth: 8 },
    adjacency: [{ a: "living", b: "kitchen", level: "preferred" }], seed: 3,
  }));
  assert.equal(body.name, "flat");
  assert.deepEqual(body.footprint, [10, 8]);
  assert.deepEqual(body.adjacency, [["living", "kitchen", "preferred"]]);
  assert.equal(body.entrance, "hallway");
  assert.equal(body.seed, 3);
  assert.deepEqual(body.rooms[0], { id: "living", kind: "living", min_area: 18, max_area: 26 });
  assert.equal("aspect_max" in body.rooms[0], false);
  assert.equal(toRequestBody(draft()).footprint, null);
});

const plan: ApartmentPlan = {
  rects: { living: [0, 0, 6, 4], hallway: [6, 0, 2, 4], bath: [0, 4, 1, 1] },
  score: 12.345,
  satisfied: [["living", "hallway", "required"], ["living", "bath", "preferred"]],
  unsatisfied: [],
};

test("plan -> svg scales uniformly to the width, tints the hallway, hides labels in tiny rooms", () => {
  const svg = planToSvg(plan, { living: "living", hallway: "hallway", bath: "bathroom" }, [8, 5], 160);
  assert.equal(svg.width, 160);
  assert.equal(svg.height, 100); // 5 m * (160 / 8)
  const living = svg.rooms.find((r) => r.id === "living")!;
  assert.deepEqual([living.x, living.y, living.w, living.h], [0, 0, 120, 80]);
  assert.equal(living.areaText, "24 m²");
  assert.equal(living.circulation, false);
  assert.equal(svg.rooms.find((r) => r.id === "hallway")!.circulation, true);
  assert.equal(svg.rooms.find((r) => r.id === "bath")!.tiny, true);
  assert.equal(living.tiny, false);
  // no footprint -> bounding box of the rects
  assert.equal(planToSvg(plan, {}, null, 80).height, 50);
});

test("adjacency summary counts required adjacencies from the spec against the plan", () => {
  const adj = [
    { a: "living", b: "hallway", level: "required" as const },
    { a: "living", b: "bath", level: "preferred" as const },
    { a: "hallway", b: "bath", level: "required" as const },
  ];
  assert.equal(adjacencySummary(plan, adj), "1 of 2 required adjacencies met");
  assert.equal(adjacencySummary(plan, []), "0 of 0 required adjacencies met");
});

test("stored apartments summarize as 'N rooms, W x D m' and are read-only", () => {
  const params = {
    rooms: [{}, {}, {}, {}, {}, {}],
    rects: { a: [0, 0, 6, 8], b: [6, 0, 4, 8] },
  };
  assert.equal(apartmentSummary(params), "6 rooms, 10 x 8 m");
  const item = { entity_id: "f-apartment", kind: "apartment" as const, name: "f", params, origin: [0, 0, 0] as [number, number, number], attach: null, depends_on: [] };
  assert.equal(summarize(item), "6 rooms, 10 x 8 m");
  assert.equal(isEditable(item), false);
  assert.equal(isEditable({ ...item, kind: "room" }), true);
  assert.equal(apartmentSummary({ rooms: [{}] }), "1 room, 0 x 0 m");
});
