/**
 * Reality Engine — PLY Binary Parser
 * Minimal binary little endian float32 PLY reader for real point clouds.
 * Matches the format emitted by engine/reconstruction/pipeline.
 */

export interface ParsedPointCloud {
  positions: Float32Array; // [x0, y0, z0, x1, y1, z1, ...]
  colors?: Float32Array;    // [r0, g0, b0, r1, g1, b1, ...] normalized 0..1
  pointCount: number;
  bounds: {
    min: [number, number, number];
    max: [number, number, number];
  };
}

export function parsePlyBuffer(buffer: ArrayBuffer): ParsedPointCloud {
  const bytes = new Uint8Array(buffer);
  const head = new TextDecoder().decode(bytes.subarray(0, 4096));
  const marker = 'end_header\n';
  const crMarker = 'end_header\r\n';

  let idx = head.indexOf(marker);
  let markerLen = marker.length;
  if (idx < 0) {
    idx = head.indexOf(crMarker);
    markerLen = crMarker.length;
  }
  if (idx < 0) {
    throw new Error('PLY: end_header not found');
  }

  const header = head.slice(0, idx);
  if (!/format binary_little_endian 1\.0/.test(header)) {
    throw new Error('PLY: only binary_little_endian format 1.0 supported');
  }

  const m = header.match(/element vertex (\d+)/);
  if (!m) {
    throw new Error('PLY: element vertex count not found');
  }
  const n = parseInt(m[1], 10);
  if (n <= 0) {
    throw new Error('PLY: point count is 0');
  }

  // Parse properties and compute stride
  const lines = header.split(/\r?\n/);
  let stride = 0;
  let hasRgb = false;
  let inVertex = false;

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'element') {
      inVertex = parts[1] === 'vertex';
      continue;
    }
    if (inVertex && parts[0] === 'property') {
      const ptype = parts[1];
      const pname = parts[2];
      if (pname === 'red' || pname === 'diffuse_red') hasRgb = true;
      if (ptype === 'float' || ptype === 'float32' || ptype === 'int' || ptype === 'uint') {
        stride += 4;
      } else if (ptype === 'double' || ptype === 'float64') {
        stride += 8;
      } else if (ptype === 'uchar' || ptype === 'uint8' || ptype === 'char' || ptype === 'int8') {
        stride += 1;
      } else if (ptype === 'ushort' || ptype === 'uint16' || ptype === 'short' || ptype === 'int16') {
        stride += 2;
      }
    }
  }

  if (stride === 0) stride = 12; // default 3 x float32

  const dataOffset = idx + markerLen;
  const dv = new DataView(buffer, dataOffset);

  const positions = new Float32Array(n * 3);
  const colors = hasRgb ? new Float32Array(n * 3) : undefined;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let i = 0; i < n; i++) {
    const base = i * stride;
    if (dataOffset + base + 12 > buffer.byteLength) break;

    const x = dv.getFloat32(base, true);
    const y = dv.getFloat32(base + 4, true);
    const z = dv.getFloat32(base + 8, true);

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;

    if (hasRgb && colors && dataOffset + base + 15 <= buffer.byteLength) {
      colors[i * 3] = dv.getUint8(base + 12) / 255;
      colors[i * 3 + 1] = dv.getUint8(base + 13) / 255;
      colors[i * 3 + 2] = dv.getUint8(base + 14) / 255;
    }
  }

  return {
    positions,
    colors,
    pointCount: n,
    bounds: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    },
  };
}
