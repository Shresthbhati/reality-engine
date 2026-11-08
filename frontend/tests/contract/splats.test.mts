import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseSplatBuffer,
  parsePlySplatBuffer,
  formatSplatCount,
  formatByteSize,
} from "../../src/lib/viewport/splat-model.ts";

test("formatSplatCount formats counts with metric suffixes", () => {
  assert.equal(formatSplatCount(0), "0");
  assert.equal(formatSplatCount(450), "450");
  assert.equal(formatSplatCount(1200), "1.2k");
  assert.equal(formatSplatCount(18500), "18.5k");
  assert.equal(formatSplatCount(1500000), "1.5M");
});

test("formatByteSize formats byte values into human-readable memory units", () => {
  assert.equal(formatByteSize(500), "500 B");
  assert.equal(formatByteSize(2048), "2 KB");
  assert.equal(formatByteSize(3 * 1024 * 1024), "3.0 MB");
  assert.equal(formatByteSize(2.5 * 1024 * 1024 * 1024), "2.5 GB");
});

test("parseSplatBuffer parses standard 32-byte chunk binary .splat buffer", () => {
  // Construct 2 splats = 64 bytes
  const buffer = new ArrayBuffer(64);
  const view = new DataView(buffer);
  const uint8 = new Uint8Array(buffer);

  // Splat 0: pos (1.0, 2.0, 3.0), scale (0.05, 0.05, 0.05), color (255, 128, 64, 255), quat identity (0, 0, 0, 1 -> 128, 128, 128, 255)
  view.setFloat32(0, 1.0, true);
  view.setFloat32(4, 2.0, true);
  view.setFloat32(8, 3.0, true);
  view.setFloat32(12, 0.05, true);
  view.setFloat32(16, 0.05, true);
  view.setFloat32(20, 0.05, true);
  uint8[24] = 255; // r
  uint8[25] = 128; // g
  uint8[26] = 64;  // b
  uint8[27] = 255; // a (opacity 1.0)
  uint8[28] = 128; // qx (0.0)
  uint8[29] = 128; // qy (0.0)
  uint8[30] = 128; // qz (0.0)
  uint8[31] = 255; // qw (1.0)

  // Splat 1: pos (-2.0, 4.0, -1.0), scale (0.1, 0.2, 0.03), color (10, 20, 30, 128), quat (128, 128, 128, 255)
  view.setFloat32(32, -2.0, true);
  view.setFloat32(36, 4.0, true);
  view.setFloat32(40, -1.0, true);
  view.setFloat32(44, 0.1, true);
  view.setFloat32(48, 0.2, true);
  view.setFloat32(52, 0.03, true);
  uint8[56] = 10;
  uint8[57] = 20;
  uint8[58] = 30;
  uint8[59] = 128; // a (opacity ~0.5)
  uint8[60] = 128;
  uint8[61] = 128;
  uint8[62] = 128;
  uint8[63] = 255;

  const cloud = parseSplatBuffer(buffer);
  assert.equal(cloud.count, 2);

  // Check positions
  assert.equal(cloud.positions[0], 1.0);
  assert.equal(cloud.positions[1], 2.0);
  assert.equal(cloud.positions[2], 3.0);
  assert.equal(cloud.positions[3], -2.0);
  assert.equal(cloud.positions[4], 4.0);
  assert.equal(cloud.positions[5], -1.0);

  // Check bounds
  assert.deepEqual(cloud.bounds.min, [-2.0, 2.0, -1.0]);
  assert.deepEqual(cloud.bounds.max, [1.0, 4.0, 3.0]);

  // Check scales
  assert.ok(Math.abs(cloud.scales[0] - 0.05) < 1e-5);
  assert.ok(Math.abs(cloud.scales[3] - 0.1) < 1e-5);

  // Check colors
  assert.equal(cloud.colors[0], 255);
  assert.equal(cloud.colors[1], 128);
  assert.equal(cloud.colors[2], 64);
  assert.equal(cloud.colors[3], 10);

  // Check opacity
  assert.equal(cloud.opacities[0], 1.0);
  assert.ok(Math.abs(cloud.opacities[1] - 128 / 255) < 1e-4);

  // Check rotation
  assert.ok(Math.abs(cloud.rotations[3] - 1.0) < 1e-3);
});

test("parseSplatBuffer gracefully handles empty buffer", () => {
  const empty = new ArrayBuffer(0);
  const cloud = parseSplatBuffer(empty);
  assert.equal(cloud.count, 0);
  assert.equal(cloud.positions.length, 0);
  assert.deepEqual(cloud.bounds.min, [0, 0, 0]);
});

test("parsePlySplatBuffer parses binary little-endian 3DGS PLY format", () => {
  const header =
    "ply\n" +
    "format binary_little_endian 1.0\n" +
    "element vertex 1\n" +
    "property float x\n" +
    "property float y\n" +
    "property float z\n" +
    "property float f_dc_0\n" +
    "property float f_dc_1\n" +
    "property float f_dc_2\n" +
    "property float opacity\n" +
    "property float scale_0\n" +
    "property float scale_1\n" +
    "property float scale_2\n" +
    "property float rot_0\n" +
    "property float rot_1\n" +
    "property float rot_2\n" +
    "property float rot_3\n" +
    "end_header\n";

  const headerBytes = new TextEncoder().encode(header);
  const recordSize = 14 * 4; // 14 float32 properties = 56 bytes
  const totalLength = headerBytes.length + recordSize;

  const buffer = new ArrayBuffer(totalLength);
  const u8 = new Uint8Array(buffer);
  u8.set(headerBytes, 0);

  const view = new DataView(buffer, headerBytes.length);
  // x, y, z
  view.setFloat32(0, 5.0, true);
  view.setFloat32(4, 6.0, true);
  view.setFloat32(8, 7.0, true);
  // f_dc_0, 1, 2 (degree 0 spherical harmonics)
  // For gray ~128: f_dc = 0.0 -> (0.5 + 0) * 255 = ~128
  view.setFloat32(12, 0.0, true);
  view.setFloat32(16, 0.0, true);
  view.setFloat32(20, 0.0, true);
  // opacity: logit 2.0 -> sigmoid(2.0) = ~0.8808
  view.setFloat32(24, 2.0, true);
  // scales: log scales (-3.0) -> exp(-3.0) = ~0.0498
  view.setFloat32(28, -3.0, true);
  view.setFloat32(32, -3.0, true);
  view.setFloat32(36, -3.0, true);
  // rot_0=qw, rot_1=qx, rot_2=qy, rot_3=qz
  view.setFloat32(40, 1.0, true);
  view.setFloat32(44, 0.0, true);
  view.setFloat32(48, 0.0, true);
  view.setFloat32(52, 0.0, true);

  const cloud = parsePlySplatBuffer(buffer);
  assert.equal(cloud.count, 1);
  assert.equal(cloud.positions[0], 5.0);
  assert.equal(cloud.positions[1], 6.0);
  assert.equal(cloud.positions[2], 7.0);

  // Check SH DC to RGB conversion
  assert.ok(Math.abs(cloud.colors[0] - 128) <= 2);

  // Check scale exp
  assert.ok(Math.abs(cloud.scales[0] - Math.exp(-3.0)) < 1e-4);

  // Check opacity sigmoid
  assert.ok(Math.abs(cloud.opacities[0] - (1 / (1 + Math.exp(-2.0)))) < 1e-4);

  // Check quaternion
  assert.equal(cloud.rotations[3], 1.0); // qw
});
