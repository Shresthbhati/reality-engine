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
