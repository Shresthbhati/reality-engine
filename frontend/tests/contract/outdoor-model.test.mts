/**
 * Outdoor model: map-imported / generated entities become honest massing proxies; nothing is invented.
 * FX is the output of geo.osm_context.osm_to_world over tests/fixtures/osm/synthetic_street_furniture.json (syn) and 4
 * entities of city_centre_kolkata_buildings.json (kol), embedded as entities so the test needs no Python.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildOutdoor,
  buildOutdoorParts,
  derivePalette,
  meshBounds,
  outdoorDisplayName,
  outdoorGroupOf,
  outdoorSummary,
  partMesh,
} from "../../src/lib/viewport/outdoor-model.ts";
import { buildSceneTree, type SceneNode } from "../../src/lib/workspace/scene-tree-model.ts";

type E = { id: string; type: string; name?: string; confidence?: number; custom_properties: Record<string, unknown> } & Record<string, unknown>;
const FX = {"syn":[{"id":"osm-node-9001","type":"infrastructure","name":"Lamp Post","confidence":0.6,"provenance":"INFERRED","semantic_labels":["lamp_post"],"custom_properties":{"osm_id":9001,"osm_type":"node","osm_tags":{"highway":"street_lamp"},"source":"OpenStreetMap","imported":true,"kind":"lamp_post","dimensions":{"height":{"value":5.0,"unit":"m","state":"ESTIMATED","basis":"default_lamp_post_height"},"footprint_side":{"value":0.3,"unit":"m","state":"ESTIMATED","basis":"default_lamp_post_size"}},"position_xyz_m":[5.142,0.0,-5.537],"bounds_min":[4.992,0.0,-5.687],"bounds_max":[5.292000000000001,5.0,-5.387]}},{"id":"osm-node-9002","type":"infrastructure","name":"Bin","confidence":0.6,"provenance":"INFERRED","semantic_labels":["bin"],"custom_properties":{"osm_id":9002,"osm_type":"node","osm_tags":{"amenity":"waste_basket"},"source":"OpenStreetMap","imported":true,"kind":"bin","dimensions":{"height":{"value":0.9,"unit":"m","state":"ESTIMATED","basis":"default_bin_height"},"footprint_side":{"value":0.5,"unit":"m","state":"ESTIMATED","basis":"default_bin_size"}},"position_xyz_m":[6.17,0.0,-6.644],"bounds_min":[5.92,0.0,-6.894],"bounds_max":[6.42,0.9,-6.394]}},{"id":"osm-node-9003","type":"infrastructure","name":"Post Box","confidence":0.6,"provenance":"INFERRED","semantic_labels":["post_box"],"custom_properties":{"osm_id":9003,"osm_type":"node","osm_tags":{"amenity":"post_box"},"source":"OpenStreetMap","imported":true,"kind":"post_box","dimensions":{"height":{"value":1.2,"unit":"m","state":"ESTIMATED","basis":"default_post_box_height"},"footprint_side":{"value":0.5,"unit":"m","state":"ESTIMATED","basis":"default_post_box_size"}},"position_xyz_m":[7.199,0.0,-7.752],"bounds_min":[6.949,0.0,-8.001999999999999],"bounds_max":[7.449,1.2,-7.502]}},{"id":"osm-node-9004","type":"structure","name":"Police barricade","confidence":0.6,"provenance":"INFERRED","semantic_labels":["jersey_barrier"],"custom_properties":{"osm_id":9004,"osm_type":"node","osm_tags":{"name":"Police barricade","barrier":"jersey_barrier"},"source":"OpenStreetMap","imported":true,"kind":"jersey_barrier","dimensions":{"height":{"value":1.0,"unit":"m","state":"ESTIMATED","basis":"default_barrier_height"},"footprint_side":{"value":1.5,"unit":"m","state":"ESTIMATED","basis":"default_barrier_size"}},"position_xyz_m":[8.227,0.0,-8.859],"bounds_min":[7.477,0.0,-9.609],"bounds_max":[8.977,1.0,-8.109],"barrier_kind":"jersey_barrier"}},{"id":"osm-node-9005","type":"vegetation","name":"Tree","confidence":0.6,"provenance":"INFERRED","semantic_labels":["tree"],"custom_properties":{"osm_id":9005,"osm_type":"node","osm_tags":{"height":"9 m","natural":"tree"},"source":"OpenStreetMap","imported":true,"kind":"tree","dimensions":{"height":{"value":9.0,"unit":"m","state":"INFERRED","basis":"osm_tag:height"},"footprint_side":{"value":3.0,"unit":"m","state":"ESTIMATED","basis":"default_tree_size"}},"position_xyz_m":[9.256,0.0,-9.966],"bounds_min":[7.756,0.0,-11.466],"bounds_max":[10.756,9.0,-8.466]}},{"id":"osm-node-9006","type":"vegetation","name":"Tree","confidence":0.6,"provenance":"INFERRED","semantic_labels":["tree"],"custom_properties":{"osm_id":9006,"osm_type":"node","osm_tags":{"natural":"tree"},"source":"OpenStreetMap","imported":true,"kind":"tree","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_tree_height"},"footprint_side":{"value":3.0,"unit":"m","state":"ESTIMATED","basis":"default_tree_size"}},"position_xyz_m":[10.284,0.0,-11.074],"bounds_min":[8.784,0.0,-12.574],"bounds_max":[11.784,6.0,-9.574]}},{"id":"osm-node-9007","type":"infrastructure","name":"Test Station","confidence":0.6,"provenance":"INFERRED","semantic_labels":["station"],"custom_properties":{"osm_id":9007,"osm_type":"node","osm_tags":{"name":"Test Station","railway":"station"},"source":"OpenStreetMap","imported":true,"kind":"station","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_station_height"},"footprint_side":{"value":20.0,"unit":"m","state":"ESTIMATED","basis":"default_station_size"}},"position_xyz_m":[-10.284,0.0,11.074],"bounds_min":[-20.284,0.0,1.0739999999999998],"bounds_max":[-0.2840000000000007,6.0,21.073999999999998]}},{"id":"osm-node-9008","type":"infrastructure","name":"Station Entrance","confidence":0.6,"provenance":"INFERRED","semantic_labels":["station_entrance"],"custom_properties":{"osm_id":9008,"osm_type":"node","osm_tags":{"railway":"subway_entrance"},"source":"OpenStreetMap","imported":true,"kind":"station_entrance","dimensions":{"height":{"value":3.0,"unit":"m","state":"ESTIMATED","basis":"default_station_entrance_height"},"footprint_side":{"value":3.0,"unit":"m","state":"ESTIMATED","basis":"default_station_entrance_size"}},"position_xyz_m":[-9.256,0.0,9.966],"bounds_min":[-10.756,0.0,8.466],"bounds_max":[-7.756,3.0,11.466]}},{"id":"osm-node-9009","type":"infrastructure","name":"Bench","confidence":0.6,"provenance":"INFERRED","semantic_labels":["bench"],"custom_properties":{"osm_id":9009,"osm_type":"node","osm_tags":{"amenity":"bench"},"source":"OpenStreetMap","imported":true,"kind":"bench","dimensions":{"height":{"value":0.8,"unit":"m","state":"ESTIMATED","basis":"default_bench_height"},"footprint_side":{"value":1.5,"unit":"m","state":"ESTIMATED","basis":"default_bench_size"}},"position_xyz_m":[-8.227,0.0,8.859],"bounds_min":[-8.977,0.0,8.109],"bounds_max":[-7.477,0.8,9.609]}},{"id":"osm-node-9010","type":"infrastructure","name":"Traffic Sign","confidence":0.6,"provenance":"INFERRED","semantic_labels":["traffic_sign"],"custom_properties":{"osm_id":9010,"osm_type":"node","osm_tags":{"highway":"traffic_signals"},"source":"OpenStreetMap","imported":true,"kind":"traffic_sign","dimensions":{"height":{"value":2.5,"unit":"m","state":"ESTIMATED","basis":"default_traffic_sign_height"},"footprint_side":{"value":0.3,"unit":"m","state":"ESTIMATED","basis":"default_traffic_sign_size"}},"position_xyz_m":[-7.199,0.0,7.752],"bounds_min":[-7.349,0.0,7.601999999999999],"bounds_max":[-7.0489999999999995,2.5,7.902]}},{"id":"osm-node-9011","type":"infrastructure","name":"Fire Hydrant","confidence":0.6,"provenance":"INFERRED","semantic_labels":["fire_hydrant"],"custom_properties":{"osm_id":9011,"osm_type":"node","osm_tags":{"emergency":"fire_hydrant"},"source":"OpenStreetMap","imported":true,"kind":"fire_hydrant","dimensions":{"height":{"value":0.8,"unit":"m","state":"ESTIMATED","basis":"default_fire_hydrant_height"},"footprint_side":{"value":0.3,"unit":"m","state":"ESTIMATED","basis":"default_fire_hydrant_size"}},"position_xyz_m":[-6.17,0.0,6.644],"bounds_min":[-6.32,0.0,6.494],"bounds_max":[-6.02,0.8,6.7940000000000005]}},{"id":"osm-way-8001","type":"infrastructure","name":"Subway","confidence":0.45,"provenance":"INFERRED","semantic_labels":["metro_line"],"custom_properties":{"osm_id":8001,"osm_type":"way","osm_tags":{"railway":"subway","tunnel":"yes","layer":"-2"},"source":"OpenStreetMap","imported":true,"kind":"metro_line","dimensions":{"width":{"value":3.0,"unit":"m","state":"ESTIMATED","basis":"default_rail_width"}},"railway":"subway","tunnel":true,"layer":-2.0,"underground":true,"polyline_xyz_m":[[-123.409,0.0,77.516],[82.272,0.0,-33.222]],"bounds_min":[-124.909,0.0,-34.722],"bounds_max":[83.772,0.3,79.016]}},{"id":"osm-way-8002","type":"infrastructure","name":"Bridge","confidence":0.45,"provenance":"INFERRED","semantic_labels":["bridge"],"custom_properties":{"osm_id":8002,"osm_type":"way","osm_tags":{"highway":"primary","bridge":"yes","layer":"1"},"source":"OpenStreetMap","imported":true,"kind":"bridge","dimensions":{"width":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_bridge_width"},"deck_elevation":{"value":5.0,"unit":"m","state":"ESTIMATED","basis":"layer*5m"}},"carries":"primary","layer":1.0,"polyline_xyz_m":[[-20.568,5.0,22.148],[30.852,5.0,22.148]],"bounds_min":[-23.568,5.0,19.148],"bounds_max":[33.852000000000004,6.0,25.148]}},{"id":"osm-way-8003","type":"stairs","name":"Steps","confidence":0.5,"provenance":"INFERRED","semantic_labels":["steps"],"custom_properties":{"osm_id":8003,"osm_type":"way","osm_tags":{"highway":"steps"},"source":"OpenStreetMap","imported":true,"kind":"steps","dimensions":{"width":{"value":1.5,"unit":"m","state":"ESTIMATED","basis":"default_steps_width"}},"polyline_xyz_m":[[10.284,0.0,-11.074],[10.284,0.0,-16.611]],"bounds_min":[9.534,0.0,-17.361],"bounds_max":[11.034,2.0,-10.324]}},{"id":"osm-way-8004","type":"road","name":"Road","confidence":0.6,"provenance":"INFERRED","semantic_labels":["road"],"custom_properties":{"osm_id":8004,"osm_type":"way","osm_tags":{"highway":"residential","width":"7"},"source":"OpenStreetMap","imported":true,"kind":"road","dimensions":{"width":{"value":7.0,"unit":"m","state":"INFERRED","basis":"osm_tag:width"}},"polyline_xyz_m":[[-123.408,0.0,-0.0],[82.272,0.0,-0.0]],"bounds_min":[-126.908,0.0,-3.5],"bounds_max":[85.772,0.2,3.5]}},{"id":"osm-way-8005","type":"sidewalk","name":"Sidewalk","confidence":0.5,"provenance":"INFERRED","semantic_labels":["sidewalk"],"custom_properties":{"osm_id":8005,"osm_type":"way","osm_tags":{"highway":"footway"},"source":"OpenStreetMap","imported":true,"kind":"sidewalk","dimensions":{"width":{"value":2.0,"unit":"m","state":"ESTIMATED","basis":"default_sidewalk_width"}},"polyline_xyz_m":[[-123.408,0.0,-2.215],[82.272,0.0,-2.215]],"bounds_min":[-124.408,0.0,-3.215],"bounds_max":[83.272,0.2,-1.2149999999999999]}},{"id":"osm-way-8006","type":"building","name":"Building","confidence":0.7,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":8006,"osm_type":"way","osm_tags":{"building":"yes","building:levels":"5","height":"20"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":20.0,"unit":"m","state":"INFERRED","basis":"osm_tag:height"}},"footprint_polygon_xyz_m":[[-20.568,0.0,-33.221],[-10.284,0.0,-33.221],[-10.284,0.0,-44.295],[-20.568,0.0,-44.295]],"footprint_estimated":false,"bounds_min":[-20.568,0.0,-44.295],"bounds_max":[-10.284,20.0,-33.221]}},{"id":"osm-way-8007","type":"building","name":"Building","confidence":0.55,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":8007,"osm_type":"way","osm_tags":{"building":"yes","building:levels":"3"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":9.0,"unit":"m","state":"ESTIMATED","basis":"building:levels*3m"}},"footprint_polygon_xyz_m":[[-20.568,0.0,-55.369],[-10.284,0.0,-55.369],[-10.284,0.0,-66.443],[-20.568,0.0,-66.443]],"footprint_estimated":false,"bounds_min":[-20.568,0.0,-66.443],"bounds_max":[-10.284,9.0,-55.369]}},{"id":"osm-way-8008","type":"building","name":"Building","confidence":0.4,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":8008,"osm_type":"way","osm_tags":{"building":"yes"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_building_height"}},"footprint_polygon_xyz_m":[[-20.568,0.0,-77.517],[-10.284,0.0,-77.517],[-10.284,0.0,-88.591],[-20.568,0.0,-88.591]],"footprint_estimated":false,"bounds_min":[-20.568,0.0,-88.591],"bounds_max":[-10.284,6.0,-77.517]}},{"id":"osm-way-8009","type":"structure","name":"Barrier","confidence":0.5,"provenance":"INFERRED","semantic_labels":["fence"],"custom_properties":{"osm_id":8009,"osm_type":"way","osm_tags":{"barrier":"fence"},"source":"OpenStreetMap","imported":true,"kind":"fence","dimensions":{"width":{"value":0.3,"unit":"m","state":"ESTIMATED","basis":"default_barrier_width"}},"polyline_xyz_m":[[-20.568,0.0,-99.664],[0.0,0.0,-99.664]],"bounds_min":[-20.718,0.0,-99.81400000000001],"bounds_max":[0.15,1.0,-99.514]}}],"kol":[{"id":"osm-way-134101085","type":"building","name":"PWC Office","confidence":0.3,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":134101085,"osm_type":"way","osm_tags":{"name":"PWC Office","building":"yes"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_building_height"},"footprint_side":{"value":12.0,"unit":"m","state":"ESTIMATED","basis":"centre_point_only"}},"footprint_polygon_xyz_m":[[-163.364,0.0,-208.751],[-151.364,0.0,-208.751],[-151.364,0.0,-196.751],[-163.364,0.0,-196.751]],"footprint_estimated":true,"bounds_min":[-163.364,0.0,-208.751],"bounds_max":[-151.364,6.0,-196.751]}},{"id":"osm-way-134101096","type":"building","name":"Haldia Petrochemicals","confidence":0.3,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":134101096,"osm_type":"way","osm_tags":{"name":"Haldia Petrochemicals","building":"yes"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_building_height"},"footprint_side":{"value":12.0,"unit":"m","state":"ESTIMATED","basis":"centre_point_only"}},"footprint_polygon_xyz_m":[[-8.602,0.0,-243.611],[3.398,0.0,-243.611],[3.398,0.0,-231.611],[-8.602,0.0,-231.611]],"footprint_estimated":true,"bounds_min":[-8.602,0.0,-243.611],"bounds_max":[3.398,6.0,-231.611]}},{"id":"osm-way-134101300","type":"building","name":"Institute of Engineering and Management","confidence":0.3,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":134101300,"osm_type":"way","osm_tags":{"name":"Institute of Engineering and Management","building":"college","amenity":"college"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_building_height"},"footprint_side":{"value":12.0,"unit":"m","state":"ESTIMATED","basis":"centre_point_only"}},"footprint_polygon_xyz_m":[[249.581,0.0,158.477],[261.581,0.0,158.477],[261.581,0.0,170.477],[249.581,0.0,170.477]],"footprint_estimated":true,"bounds_min":[249.581,0.0,158.477],"bounds_max":[261.581,6.0,170.477]}},{"id":"osm-way-134101262","type":"building","name":"Building","confidence":0.3,"provenance":"INFERRED","semantic_labels":["building"],"custom_properties":{"osm_id":134101262,"osm_type":"way","osm_tags":{"building":"college"},"source":"OpenStreetMap","imported":true,"kind":"building","dimensions":{"height":{"value":6.0,"unit":"m","state":"ESTIMATED","basis":"default_building_height"},"footprint_side":{"value":12.0,"unit":"m","state":"ESTIMATED","basis":"centre_point_only"}},"footprint_polygon_xyz_m":[[-302.992,0.0,-14.884],[-290.992,0.0,-14.884],[-290.992,0.0,-2.8840000000000003],[-302.992,0.0,-2.8840000000000003]],"footprint_estimated":true,"bounds_min":[-302.992,0.0,-14.884],"bounds_max":[-290.992,6.0,-2.8840000000000003]}}]} as { syn: E[]; kol: E[] };
const world = (entities: E[], metadata: Record<string, unknown> = {}) =>
  ({ id: "w", entities: Object.fromEntries(entities.map((e) => [e.id, e])), metadata }) as never;
const byId = (id: string) => FX.syn.find((e) => e.id === id)!;
const syn = world(FX.syn, { geo: { source: "OpenStreetMap" } });
const parts = buildOutdoorParts(syn);
const part = (id: string) => parts.find((p) => p.entityId === id)!;

test("every synthetic entity maps to a part: kind -> shape", () => {
  assert.equal(parts.length, FX.syn.length);
  const shape = (id: string) => part(id).shape;
  assert.equal(shape("osm-way-8006"), "extrusion"); // building
  assert.equal(shape("osm-way-8004"), "strip"); // road
  assert.equal(shape("osm-way-8005"), "strip"); // sidewalk
  assert.equal(shape("osm-way-8001"), "strip"); // metro line
  assert.equal(shape("osm-way-8002"), "strip"); // bridge deck
  assert.equal(shape("osm-way-8003"), "stairs");
  assert.equal(shape("osm-way-8009"), "strip"); // fence along its way
  assert.equal(shape("osm-node-9001"), "pole"); // lamp post
  assert.equal(shape("osm-node-9010"), "pole"); // traffic sign
  assert.equal(shape("osm-node-9002"), "box"); // bin
  assert.equal(shape("osm-node-9003"), "box"); // post box
  assert.equal(shape("osm-node-9004"), "box"); // jersey barrier on a node
  assert.equal(shape("osm-node-9005"), "tree");
  assert.equal(shape("osm-node-9007"), "box"); // station
});

test("groups follow the kind", () => {
  assert.equal(part("osm-way-8006").group, "buildings");
  assert.equal(part("osm-way-8004").group, "roads");
  assert.equal(part("osm-way-8001").group, "rail");
  assert.equal(part("osm-way-8002").group, "bridges");
  assert.equal(part("osm-node-9001").group, "furniture");
  assert.equal(part("osm-node-9004").group, "furniture");
  assert.equal(part("osm-node-9005").group, "trees");
});

test("buildings extrude their footprint to the tagged height", () => {
  const b = part("osm-way-8006");
  assert.equal(b.height, 20);
  assert.ok(b.polygon && b.polygon.length >= 3);
  assert.equal(b.imported, true);
  assert.equal(b.generated, false);
  assert.equal(b.confidence, byId("osm-way-8006").confidence);
  const m = partMesh(b, "y");
  const bb = meshBounds(m.positions)!;
  assert.ok(Math.abs(bb.max[1] - bb.min[1] - 20) < 1e-6, "mesh is exactly 20 m tall");
  assert.ok(m.indices.length > 0 && m.indices.every((i) => i < m.positions.length / 3));
});

test("real fixture buildings: centre-only outline is flagged estimated, default height flagged", () => {
  const out = buildOutdoor(world(FX.kol));
  assert.equal(out.parts.length, 4);
  for (const p of out.parts) {
    assert.equal(p.shape, "extrusion");
    assert.equal(p.height, 6);
    assert.equal(p.estimated, true);
    assert.match(p.note ?? "", /footprint estimated/);
    assert.equal(p.polygon!.length, 4);
  }
  assert.equal(outdoorDisplayName(FX.kol[0] as never, 1), "PWC Office"); // OSM name tag wins
});

test("a building without a footprint falls back to a flagged box, or is skipped when it has no bounds", () => {
  const b = structuredClone(FX.kol[0]);
  delete b.custom_properties.footprint_polygon_xyz_m;
  const out = buildOutdoor(world([b]));
  assert.equal(out.parts[0].shape, "box");
  assert.equal(out.parts[0].estimated, true);
  assert.match(out.parts[0].note ?? "", /footprint missing/);
  delete b.custom_properties.bounds_min;
  delete b.custom_properties.bounds_max;
  const none = buildOutdoor(world([b]));
  assert.equal(none.parts.length, 0);
  assert.deepEqual(none.diagnostics.skipped.map((s) => s.id), [b.id]);
  assert.match(none.diagnostics.skipped[0].reason, /no footprint/);
});

test("roads, sidewalks, rail are flat strips of the polyline with its width", () => {
  const road = part("osm-way-8004");
  assert.equal(road.width, 7);
  assert.ok(road.polyline && road.polyline.length >= 2);
  assert.ok(road.height < 0.1);
  assert.ok(part("osm-way-8005").height > road.height, "sidewalk sits above the carriageway (no z-fighting)");
  assert.ok(part("osm-way-8001").height > part("osm-way-8005").height);
  const bb = meshBounds(partMesh(road, "y").positions)!;
  assert.ok(bb.max[1] <= 0.05 + 1e-9);
});

test("underground metro is drawn below ground and flagged", () => {
  const metro = part("osm-way-8001");
  assert.equal(metro.baseY < 0, true);
  assert.equal(metro.estimated, true);
  assert.match(metro.note ?? "", /underground/);
});

test("bridge: deck raised to its elevation with two flagged piers", () => {
  const br = part("osm-way-8002");
  assert.equal(br.baseY, 5);
  assert.equal(br.piers, true);
  assert.equal(br.estimated, true);
  const withPiers = partMesh(br, "y");
  const noPiers = partMesh({ ...br, piers: false }, "y");
  assert.ok(withPiers.indices.length > noPiers.indices.length);
  assert.equal(meshBounds(withPiers.positions)!.min[1], 0, "piers reach the ground");
  assert.equal(meshBounds(noPiers.positions)!.min[1], 5);
});

test("poles, boxes and trees are proxies with plausible, flagged sizes", () => {
  const lamp = part("osm-node-9001");
  assert.equal(lamp.height, 5);
  assert.equal(lamp.estimated, true);
  const lb = meshBounds(partMesh(lamp, "y").positions)!;
  assert.ok(lb.max[1] > 5, "lamp head sits on top of the pole");
  assert.ok(lb.max[0] - lb.min[0] <= 0.6);
  assert.equal(part("osm-node-9002").height, 0.9);
  assert.equal(part("osm-node-9003").height, 1.2);
  const tall = part("osm-node-9005"), dflt = part("osm-node-9006");
  assert.equal(tall.height, 9); // tagged
  assert.equal(dflt.height, 6); // default
  assert.equal(dflt.estimated, true);
  const tb = meshBounds(partMesh(tall, "y").positions)!;
  assert.ok(Math.abs(tb.max[1] - 9) < 1e-6 && tb.min[1] === 0);
  assert.ok(partMesh(tall, "y").indices.length > 100, "trunk + canopy");
});

test("Z-up worlds keep ground = (x, y) and height = z", () => {
  const driveway: E = {
    id: "gen/0-driveway", type: "road", name: "Driveway (generated)", confidence: 0.3,
    custom_properties: {
      generated_by: "connector", road_kind: "driveway", dimensions: { width_m: { value: 3.5, state: "ESTIMATED", unit: "m" } },
      footprint_polygon_xyz_m: [[27, -22, 0], [51, 72, 0], [47, 73, 0], [24, -22, 0]],
    },
  };
  const out = buildOutdoor(world([driveway], { compiled_from: { composed: true, up: [0, 0, 1] } }));
  assert.equal(out.up, "z");
  const p = out.parts[0];
  assert.equal(p.generated, true);
  assert.equal(p.imported, false);
  assert.equal(p.shape, "strip");
  assert.equal(p.group, "roads");
  assert.deepEqual(p.polygon![1], [51, 72]);
  const bb = meshBounds(partMesh(p, "z").positions)!;
  assert.ok(bb.max[2] > 0 && bb.max[2] <= 0.05 + 1e-9, "height on z");
  assert.ok(bb.min[1] < -21 && bb.max[1] > 72, "ground y preserved");
});

test("no fabricated parts: non-outdoor, organizational and geometry-less entities draw nothing", () => {
  const room: E = { id: "r1", type: "room", custom_properties: { footprint_polygon_xyz_m: [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]] } };
  const container: E = { id: "compose/building/x", type: "building", custom_properties: { organizational: true, composed_by: "compose" } };
  const indoorBuilding: E = { id: "b", type: "building", custom_properties: {} };
  const bare: E = { id: "osm-node-1", type: "infrastructure", custom_properties: { imported: true, kind: "lamp_post" } };
  const out = buildOutdoor(world([room, container, indoorBuilding, bare]));
  assert.equal(out.parts.length, 0);
  assert.deepEqual(out.diagnostics.skipped, [{ id: "osm-node-1", reason: "no footprint, polyline, position or bounds" }]);
  assert.equal(out.diagnostics.drawn, 0);
});

test("unknown kinds become a small neutral box so nothing is silently missing; counted in diagnostics", () => {
  const odd: E = {
    id: "osm-node-77", type: "infrastructure", name: "Thing", confidence: 0.6,
    custom_properties: { imported: true, kind: "weird_thing", position_xyz_m: [5, 0, 7], dimensions: {} },
  };
  const out = buildOutdoor(world([odd]));
  assert.equal(out.parts.length, 1);
  const p = out.parts[0];
  assert.equal(p.shape, "box");
  assert.equal(p.estimated, true);
  assert.match(p.note ?? "", /unknown kind/);
  assert.ok(p.size[0] <= 1 && p.height <= 1);
  assert.equal(out.diagnostics.byGroup.other, 1);
});

test("diagnostics count what was drawn per group and kind; summary is ordered and honest", () => {
  const d = buildOutdoor(syn).diagnostics;
  assert.equal(d.drawn, 20);
  assert.equal(d.skipped.length, 0);
  assert.equal(d.byKind.tree, 2);
  assert.equal(d.byGroup.trees, 2);
  const s = outdoorSummary(parts, 3);
  assert.match(s, /^3 buildings, 2 trees, /);
  assert.match(s, /\+\d+ more kinds$/);
});

test("palette is token-derived and keeps kinds apart", () => {
  const pal = derivePalette({ bg: [23, 25, 24], gridMajor: [59, 65, 59], text: [236, 237, 230], textSecondary: [162, 166, 154], accent: [197, 138, 96], high: [76, 175, 98] });
  assert.deepEqual(pal.rail, [197, 138, 96]);
  assert.notDeepEqual(pal.road, pal.sidewalk);
  assert.notDeepEqual(pal.building, pal.road);
  assert.deepEqual(buildOutdoorParts(syn, pal).find((p) => p.kind === "metro_line")!.color, pal.rail);
});

test("scene tree: outdoor kinds get typed groups, OSM names, honest counts, and keep the imported badge", () => {
  const tree = buildSceneTree(syn);
  const labels = tree.map((n) => n.label);
  assert.deepEqual(labels, ["Buildings (3)", "Roads & paths (3)", "Rail & metro (3)", "Bridges (1)", "Street furniture (8)", "Trees (2)"]);
  const leaves = (n: SceneNode): SceneNode[] => (n.entityId ? [n] : n.children.flatMap(leaves));
  const all = tree.flatMap(leaves);
  assert.equal(all.length, 20);
  assert.ok(all.every((n) => n.badge === "Imported from map"));
  const furniture = tree.find((n) => n.label.startsWith("Street furniture"))!;
  assert.ok(furniture.children.some((c) => c.label === "Police barricade")); // OSM name tag
  assert.ok(furniture.children.some((c) => c.label === "Lamp post 001"));
  assert.equal(new Set(all.map((n) => n.label)).size, all.length, "labels are unique");
  assert.equal(tree[0].leafIds.length, 3);
});

test("scene tree: many buildings are all listed (not collapsed to one building root)", () => {
  const many = Array.from({ length: 95 }, (_, i) => {
    const e = structuredClone(FX.kol[0]);
    e.id = `osm-way-${i}`;
    e.name = "Building";
    e.custom_properties.osm_tags = {};
    return e;
  });
  const tree = buildSceneTree(world(many));
  assert.equal(tree.length, 1);
  assert.equal(tree[0].label, "Buildings (95)");
  assert.equal(tree[0].children[94].label, "Building 095");
});

test("outdoorGroupOf is null for captured entities", () => {
  assert.equal(outdoorGroupOf({ id: "w1", type: "wall", custom_properties: {} } as never), null);
  assert.equal(outdoorGroupOf(byId("osm-way-8006") as never), "buildings");
});
