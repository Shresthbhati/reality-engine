/**
 * Room shell model: an L-shaped room with a false ceiling, a window and furniture becomes ONE shell with the right
 * areas. Fixture mirrors what perception/architecture/room_semantics.py writes (up = +Z here, plan basis a=y, b=-x).
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { buildRoomShellParts, triangulate2D, type ShellPart } from "../../src/lib/viewport/room-shell-model.ts";

const area = (p: ShellPart): number => {
  let a = 0;
  for (let i = 0; i < p.indices.length; i += 3) {
    const v = (k: number): [number, number, number] => [p.positions[p.indices[i + k] * 3], p.positions[p.indices[i + k] * 3 + 1], p.positions[p.indices[i + k] * 3 + 2]];
    const [a0, b0, c0] = [v(0), v(1), v(2)];
    const u = [b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]];
    const w = [c0[0] - a0[0], c0[1] - a0[1], c0[2] - a0[2]];
    a += Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]) / 2;
  }
  return a;
};
const sum = (parts: ShellPart[]) => parts.reduce((s, p) => s + area(p), 0);

function lRoomWorld() {
  const poly = [[6, 0, 0], [0, 0, 0], [0, 5, 0], [3, 5, 0], [3, 3, 0], [6, 3, 0]];
  return {
    id: "w", entities: {
      "wall-1": { id: "wall-1", type: "wall", custom_properties: { mean_color_rgb: [180, 170, 160] } },
      "floor-1": { id: "floor-1", type: "floor", custom_properties: { mean_color_rgb: [150, 120, 90] } },
      "room-1": {
        id: "room-1", type: "room", custom_properties: {
          footprint_polygon_xyz_m: poly, floor_height_m: 0, footprint_basis: { up: [0, 0, 1] },
          footprint_edge_heights_m: [2.5, 2.8, 2.8, 2.8, 2.5, 2.5], boundary_element_ids: ["wall-1", "floor-1"],
          ceiling_height_min_m: 2.5,
          ceiling_layers: [
            { role: "false_ceiling", height_above_floor_m: 2.5, entity_ids: ["c1"],
              polygon_xyz_m: [[0, 0, 2.5], [6, 0, 2.5], [6, 3, 2.5], [0, 3, 2.5]] },
            { role: "structural_ceiling", height_above_floor_m: 2.8, entity_ids: ["c2"],
              polygon_xyz_m: [[0, 3, 2.8], [3, 3, 2.8], [3, 5, 2.8], [0, 5, 2.8]] },
          ],
        },
      },
      "win-1": { id: "win-1", type: "window", custom_properties: { bounds_min: [1.5, 0, 0.9], bounds_max: [3.5, 0.15, 2.1] } },
      "obj-1": {
        id: "obj-1", type: "unknown", semantic_labels: ["furniture_candidate"], custom_properties: {
          room_id: "room-1", plan_obb_size_m: [1.2, 0.8], plan_obb_centre_m: [1.2, -4.2], plan_obb_angle_deg: 0,
          base_height_above_floor_m: 0, height_m: 0.75, mean_color_rgb: [128, 96, 60],
        },
      },
    },
  } as never;
}

test("ear clipping triangulates a concave L polygon into n-2 triangles", () => {
  const tri = triangulate2D([[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]]);
  assert.equal(tri.length, 12); // 6 vertices -> 4 triangles
});

test("floor is the true L footprint (24 m^2), not the 30 m^2 bounding box", () => {
  const floor = buildRoomShellParts(lRoomWorld()).filter((p) => p.kind === "floor");
  assert.equal(floor.length, 1);
  assert.ok(Math.abs(area(floor[0]) - 24) < 1e-9, `area ${area(floor[0])}`);
  assert.deepEqual(floor[0].color, [150, 120, 90]); // measured colour, not a type default
});

test("walls follow the per-edge heights and the window is a real opening in its wall", () => {
  const parts = buildRoomShellParts(lRoomWorld());
  const edge0 = parts.filter((p) => p.id.startsWith("room-1:wall0:"));
  assert.equal(edge0.length, 4, "left, right, below, above the window");
  assert.ok(Math.abs(sum(edge0) - (6 * 2.5 - 2 * 1.2)) < 1e-9, `edge0 wall area ${sum(edge0)}`);
  const glass = parts.filter((p) => p.kind === "window_glass");
  assert.equal(glass.length, 1);
  assert.ok(Math.abs(area(glass[0]) - 2.4) < 1e-9);
  assert.ok(glass[0].opacity < 1, "glass is translucent");
  assert.equal(parts.filter((p) => p.kind === "window_frame").length, 4);
  const edge1 = parts.filter((p) => p.id.startsWith("room-1:wall1:"));
  assert.ok(Math.abs(sum(edge1) - 5 * 2.8) < 1e-9, "edge 1 has no window and its own height");
});

test("false ceiling and structural ceiling are two layers of ONE room, with a drop face between them", () => {
  const parts = buildRoomShellParts(lRoomWorld());
  assert.equal(parts.filter((p) => p.kind === "false_ceiling").length, 1);
  assert.equal(parts.filter((p) => p.kind === "ceiling").length, 1);
  const step = parts.filter((p) => p.kind === "ceiling_step");
  assert.ok(step.length >= 1);
  assert.ok(Math.abs(sum(step) - 3 * 0.3) < 0.03, `drop face area ${sum(step)} (3 m long x 0.3 m)`);
});

test("furniture candidate becomes an oriented box of the measured size and colour", () => {
  const f = buildRoomShellParts(lRoomWorld()).filter((p) => p.kind === "furniture");
  assert.equal(f.length, 1);
  assert.equal(f[0].positions.length, 24);
  assert.deepEqual(f[0].color, [128, 96, 60]);
  const xs = f[0].positions.filter((_, i) => i % 3 === 0);
  const ys = f[0].positions.filter((_, i) => i % 3 === 1);
  // plan basis a=y, b=-x: size [1.2 along a(y), 0.8 along b(x)] at centre y=1.2, x=4.2
  assert.ok(Math.abs(Math.max(...xs) - Math.min(...xs) - 0.8) < 1e-9);
  assert.ok(Math.abs(Math.max(...ys) - Math.min(...ys) - 1.2) < 1e-9);
  assert.ok(Math.abs((Math.max(...xs) + Math.min(...xs)) / 2 - 4.2) < 1e-9);
});

test("a room without a measured footprint produces no shell (viewer keeps its legacy boxes)", () => {
  const w = { id: "w", entities: { r: { id: "r", type: "room", custom_properties: {} } } } as never;
  assert.deepEqual(buildRoomShellParts(w), []);
  assert.deepEqual(buildRoomShellParts(null), []);
});

// ---- V1: one shell, three modes that differ only in material -----------------------------------------------------

import { partMaterial, legendFor, confidenceRamp, type ShellRenderMode } from "../../src/lib/viewport/room-shell-model.ts";

const MODES: ShellRenderMode[] = ["solid", "wireframe", "both", "confidence"];

function withConfidence() {
  const w = lRoomWorld() as unknown as { entities: Record<string, Record<string, unknown>> };
  w.entities["wall-1"].confidence = 0.9;
  w.entities["wall-1"].provenance = "OBSERVED";
  w.entities["floor-1"].confidence = 0.3;
  w.entities["floor-1"].provenance = "INFERRED";
  w.entities["win-1"].provenance = "UNOBSERVED";
  return w as never;
}

test("all modes share the same ShellPart ids and positions; materials never touch geometry", () => {
  const parts = buildRoomShellParts(withConfidence());
  const before = JSON.stringify(parts.map((p) => [p.id, p.positions, p.indices]));
  const ids = parts.map((p) => p.id);
  for (const mode of MODES) {
    parts.forEach((p) => partMaterial(p, mode));
    assert.equal(JSON.stringify(parts.map((p) => [p.id, p.positions, p.indices])), before, `mode ${mode} altered geometry`);
  }
  // rebuilding for the same world version gives the same id set and positions
  const again = buildRoomShellParts(withConfidence());
  assert.deepEqual(again.map((p) => p.id), ids);
  assert.deepEqual(again.map((p) => p.positions), parts.map((p) => p.positions));
  assert.equal(new Set(ids).size, ids.length, "part ids are unique");
});

test("confidence mode: ramp from entity confidence, hatch for inferred, distinct unknown/unobserved, nothing invented", () => {
  const parts = buildRoomShellParts(withConfidence());
  const wall = parts.find((p) => p.kind === "wall")!;
  const floor = parts.find((p) => p.kind === "floor")!;
  const glass = parts.find((p) => p.kind === "window_glass")!;
  const furniture = parts.find((p) => p.kind === "furniture")!;
  assert.deepEqual(partMaterial(wall, "confidence").color, confidenceRamp(0.9));
  assert.equal(partMaterial(wall, "confidence").hatch, false);
  assert.deepEqual(partMaterial(floor, "confidence").color, confidenceRamp(0.3));
  assert.equal(partMaterial(floor, "confidence").hatch, true, "INFERRED provenance is hatched");
  assert.equal(glass.confidenceState, "unobserved");
  assert.equal(furniture.confidence, null, "entity without confidence stays null, not a guessed value");
  assert.equal(furniture.confidenceState, "unknown");
  const unknown = partMaterial(furniture, "confidence").color;
  const unobserved = partMaterial(glass, "confidence").color;
  assert.notDeepEqual(unknown, unobserved);
  for (const c of [0, 0.5, 1]) assert.notDeepEqual(confidenceRamp(c), unknown);
  assert.ok(legendFor("confidence").length >= 5);
});

test("fallback colours are flagged colorMeasured=false and hatched in coloured mode; measured ones are not", () => {
  const parts = buildRoomShellParts(lRoomWorld());
  const floor = parts.find((p) => p.kind === "floor")!;
  assert.equal(floor.colorMeasured, true);
  assert.equal(partMaterial(floor, "solid").hatch, false);
  const unmeasured = parts.filter((p) => ["window_glass", "window_frame", "ceiling_step"].includes(p.kind));
  assert.ok(unmeasured.length >= 3);
  for (const p of unmeasured) {
    assert.equal(p.colorMeasured, false, p.id);
    assert.equal(partMaterial(p, "solid").hatch, true, p.id);
  }
  // a world with no measured colours anywhere: walls fall back and are flagged
  const w = lRoomWorld() as unknown as { entities: Record<string, { custom_properties: Record<string, unknown> }> };
  delete w.entities["wall-1"].custom_properties.mean_color_rgb;
  const wall = buildRoomShellParts(w as never).find((p) => p.kind === "wall")!;
  assert.equal(wall.colorMeasured, false);
  assert.ok(legendFor("solid").some((l) => l.hatch));
});

test("rooms with and without footprint polygons: with -> shell, without -> empty for every mode (control disabled, not hidden)", () => {
  assert.ok(buildRoomShellParts(lRoomWorld()).length > 0);
  const none = { id: "w", entities: { r: { id: "r", type: "room", custom_properties: { floor_height_m: 0 } } } } as never;
  assert.equal(buildRoomShellParts(none).length, 0);
});

// ---- S3/S4: solid wall slabs drive the shell; openings are real holes; no footprint polygon needed ----------------

import { wallPieces } from "../../src/lib/viewport/room-shell-model.ts";

function slabWorld(withFootprint: boolean) {
  const slab = (id: string, start: number[], dir: number[], out: number[], length: number, openings: unknown[], extra = {}) => ({
    id, type: "wall", semantic_labels: ["wall_slab"], confidence: 0.9, provenance: "RECONSTRUCTED",
    custom_properties: {
      room_id: "room-1", mean_color_rgb: [190, 180, 170],
      wall_slab: { start_xyz_m: start, direction: dir, outward: out, up: [0, 0, 1], length_m: length, height_m: 2.6, thickness_m: 0.1, openings, ...extra },
    },
  });
  const room: Record<string, unknown> = { id: "room-1", type: "room", custom_properties: { wall_color_rgb: [190, 180, 170] } };
  if (withFootprint) {
    room.custom_properties = {
      footprint_polygon_xyz_m: [[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]], floor_height_m: 0, footprint_basis: { up: [0, 0, 1] },
      footprint_edge_heights_m: [2.6, 2.6, 2.6, 2.6], ceiling_height_min_m: 2.6, ceiling_layers: [],
    };
  }
  return {
    id: "w", entities: {
      "room-1": room,
      "wall-1": slab("wall-1", [0, 0, 0], [1, 0, 0], [0, -1, 0], 5, [
        { entity_id: "door-1", kind: "door", offset_m: 1, width_m: 0.9, sill_m: 0, height_m: 2.05 },
        { entity_id: "win-1", kind: "window", offset_m: 3, width_m: 1.2, sill_m: 0.9, height_m: 1.2 }]),
      "wall-2": slab("wall-2", [5, 0, 0], [0, 1, 0], [1, 0, 0], 4, [], { unobserved_gaps: [{ entity_id: "gap-1", kind: "opening" }] }),
      "floor-slab": { id: "floor-slab", type: "floor", custom_properties: { room_id: "room-1", mean_color_rgb: [150, 110, 70],
        slab: { polygon_xyz_m: [[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]], up: [0, 0, 1], thickness_m: 0.15, direction: -1 } } },
    },
  } as never;
}

test("wallPieces tiles the wall minus its openings", () => {
  const p = wallPieces(5, 2.6, [{ offset: 1, width: 0.9, sill: 0, height: 2.05 }, { offset: 3, width: 1.2, sill: 0.9, height: 1.2 }]);
  const a = p.reduce((s, [s0, s1, v0, v1]) => s + (s1 - s0) * (v1 - v0), 0);
  assert.ok(Math.abs(a - (5 * 2.6 - 0.9 * 2.05 - 1.2 * 1.2)) < 1e-9, `area ${a}`);
});

for (const withFootprint of [false, true]) {
  test(`solid wall entities render the room (footprint polygon ${withFootprint ? "present" : "ABSENT"}): holes, door frame, window glass`, () => {
    const parts = buildRoomShellParts(slabWorld(withFootprint));
    assert.ok(parts.length > 0, "a room with wall slabs is drawn even without a footprint polygon");
    assert.equal(parts.filter((p) => p.id.startsWith("room-1:wall")).length, 0, "no legacy footprint walls when slabs exist");
    const w1 = parts.filter((p) => p.kind === "wall" && p.entityId === "wall-1");
    assert.equal(w1.length, wallPieces(5, 2.6, [{ offset: 1, width: 0.9, sill: 0, height: 2.05 }, { offset: 3, width: 1.2, sill: 0.9, height: 1.2 }]).length);
    // volume of the carved slab = (L*H - holes) * thickness, via the divergence theorem on each outward-wound box
    const vol = w1.reduce((s, p) => {
      let v = 0;
      for (let i = 0; i < p.indices.length; i += 3) {
        const q = (k: number) => [0, 1, 2].map((a) => p.positions[p.indices[i + k] * 3 + a]);
        const [a, b, c] = [q(0), q(1), q(2)];
        v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
      }
      return s + v;
    }, 0);
    assert.ok(Math.abs(vol - (5 * 2.6 - 0.9 * 2.05 - 1.2 * 1.2) * 0.1) < 1e-9, `volume ${vol}`);
    assert.equal(parts.filter((p) => p.kind === "window_glass").length, 1, "only the window has glass");
    assert.equal(parts.filter((p) => p.kind === "window_frame" && p.entityId === "door-1").length, 3, "door frame: head + two jambs");
    assert.equal(parts.filter((p) => p.kind === "window_frame" && p.entityId === "win-1").length, 4);
    // wall-2 has an unobserved gap recorded, never carved: it is one solid box
    assert.equal(parts.filter((p) => p.kind === "wall" && p.entityId === "wall-2").length, 1);
    assert.equal(parts.filter((p) => p.kind === "floor").length, 1);
    assert.deepEqual(parts.find((p) => p.kind === "wall")!.color, [190, 180, 170]);
    assert.equal(parts.find((p) => p.kind === "wall")!.colorMeasured, true);
    assert.equal(new Set(parts.map((p) => p.id)).size, parts.length, "ids unique");
  });
}
