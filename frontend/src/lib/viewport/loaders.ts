/**
 * High-performance binary loaders for Reality Engine pipeline artifacts.
 * Consumes authoritative binary float32 PLY point clouds and meshes.
 */

export function parsePly(bytes: Uint8Array): Float32Array {
  // Minimal binary_little_endian float32 xyz reader produced by SfM
  const head = new TextDecoder().decode(bytes.subarray(0, 2048));
  const marker = "end_header\n";
  const idx = head.indexOf(marker);
  if (idx < 0) throw new Error("PLY: end_header not found");
  const header = head.slice(0, idx);
  if (!/format binary_little_endian 1\.0/.test(header)) {
    throw new Error("PLY: only binary_little_endian supported");
  }
  const m = header.match(/element vertex (\d+)/);
  if (!m) throw new Error("PLY: vertex count not found");
  const n = parseInt(m[1], 10);
  const data = new Float32Array(
    bytes.buffer,
    bytes.byteOffset + idx + marker.length,
    n * 3
  );
  return data;
}

export function parseMeshPly(
  bytes: Uint8Array
): { positions: Float32Array; indices: Uint32Array } | null {
  const head = new TextDecoder().decode(bytes.subarray(0, 4096));
  const marker = "end_header\n";
  const idx = head.indexOf(marker);
  if (idx < 0) throw new Error("PLY: end_header not found");
  const header = head.slice(0, idx);
  if (!/format binary_little_endian 1\.0/.test(header)) {
    throw new Error("PLY: only binary_little_endian supported");
  }
  const vm = header.match(/element vertex (\d+)/);
  const fm = header.match(/element face (\d+)/);
  if (!vm) throw new Error("PLY: vertex count not found");
  const nv = parseInt(vm[1], 10);
  const nf = fm ? parseInt(fm[1], 10) : 0;

  const sizes: Record<string, number> = {
    float: 4,
    double: 8,
    uchar: 1,
    char: 1,
    ushort: 2,
    short: 2,
    uint: 4,
    int: 4,
  };
  let stride = 0;
  let inVertex = false;
  let countFmt = "uchar";
  let indexType = "uint";

  for (const line of header.split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      continue;
    }
    if (!inVertex && parts[0] === "property" && parts[1] === "list") {
      countFmt = parts[2];
      indexType = parts[3];
      continue;
    }
    if (inVertex && parts[0] === "property" && sizes[parts[1]]) {
      stride += sizes[parts[1]];
    }
  }

  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = idx + marker.length;
  const positions = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    const base = off + i * stride;
    positions[i * 3] = dv.getFloat32(base, true);
    positions[i * 3 + 1] = dv.getFloat32(base + 4, true);
    positions[i * 3 + 2] = dv.getFloat32(base + 8, true);
  }
  off += nv * stride;
  const countSize = sizes[countFmt] || 1;
  const indexSize = sizes[indexType] || 4;
  const indices = new Uint32Array(nf * 3);
  let fi = 0;

  for (let f = 0; f < nf && off + countSize <= bytes.byteLength; f++) {
    let count: number;
    if (countFmt === "uchar") count = dv.getUint8(off);
    else if (countFmt === "ushort") count = dv.getUint16(off, true);
    else count = dv.getInt32(off, true);
    off += countSize;

    if (count === 3 && off + indexSize * 3 <= bytes.byteLength) {
      for (let k = 0; k < 3; k++) {
        indices[fi++] =
          indexSize === 2 ? dv.getUint16(off, true) : dv.getUint32(off, true);
        off += indexSize;
      }
    } else {
      off += indexSize * count;
    }
  }
  if (fi === 0) return null;
  return { positions, indices: indices.subarray(0, fi) };
}

export function pointsBounds(pts: Float32Array): {
  min: [number, number, number];
  max: [number, number, number];
} {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pts.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = pts[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, max };
}

export function robustPointsBounds(pts: Float32Array): {
  min: [number, number, number];
  max: [number, number, number];
  center: [number, number, number];
  extent: number;
} {
  // 2nd/98th percentile per axis on a strided sample: depth-noise
  // outliers must not stretch the default framing.
  const stride = Math.max(1, Math.floor(pts.length / 3 / 20000));
  const cols: [number[], number[], number[]] = [[], [], []];
  for (let i = 0; i < pts.length; i += 3 * stride) {
    for (let k = 0; k < 3; k++) cols[k].push(pts[i + k]);
  }
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const c = cols[k].sort((a, b) => a - b);
    min[k] = c[Math.floor(0.02 * (c.length - 1))] ?? 0;
    max[k] = c[Math.ceil(0.98 * (c.length - 1))] ?? 0;
  }
  const center: [number, number, number] = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const extent = Math.max(
    max[0] - min[0],
    max[1] - min[1],
    max[2] - min[2]
  );
  return { min, max, center, extent };
}
