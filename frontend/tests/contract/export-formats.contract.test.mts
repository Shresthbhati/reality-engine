import test from "node:test";
import assert from "node:assert/strict";
import type { ExportFormat } from "../../src/lib/api/export.ts";

test("ExportFormat type supports all standard 3D, CAD, BIM, GIS, and Floor Plan formats", () => {
  const formats: ExportFormat[] = [
    "gltf",
    "glb",
    "usda",
    "obj",
    "ply",
    "stl",
    "ifc",
    "blender",
    "cityjson",
    "citygml",
    "geojson",
    "godot",
    "floorplan_svg",
    "floorplan_dxf",
    "room_report_pdf",
  ];

  assert.equal(formats.length, 15);
  assert.ok(formats.includes("glb"));
  assert.ok(formats.includes("obj"));
  assert.ok(formats.includes("ply"));
  assert.ok(formats.includes("stl"));
  assert.ok(formats.includes("floorplan_svg"));
  assert.ok(formats.includes("floorplan_dxf"));
  assert.ok(formats.includes("room_report_pdf"));
});
