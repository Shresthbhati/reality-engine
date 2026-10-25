/**
 * High-performance binary loaders for Reality Engine pipeline artifacts.
 * Consumes authoritative binary float32 PLY point clouds and meshes.
 */

const PLY_SIZES: Record<string, number> = {
  float: 4, float32: 4, double: 8, float64: 8,
  uchar: 1, uint8: 1, char: 1, int8: 1,
  ushort: 2, uint16: 2, short: 2, int16: 2,
  uint: 4, uint32: 4, int: 4, int32: 4,
};

/**
 * Vertex positions from a PLY point cloud.
 *
 * Accepts `binary_little_endian` (the SfM/pipeline artifacts) AND `ascii`
 * (what the world API's `/points` stream currently emits -- the loader used
 * to reject it, so no world's points ever rendered). Position columns are
 * found by property NAME (x, y, z), so extra per-vertex properties are fine.
 * The result is always a fresh, aligned Float32Array, except for the
 * aligned x/y/z-only binary layout which stays zero-copy.
 */
export function parsePly(bytes: Uint8Array): Float32Array {
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 8192)));
  const marker = "end_header\n";
  const idx = head.indexOf(marker);
  if (idx < 0) throw new Error("PLY: end_header not found");
  const header = head.slice(0, idx);
  const isBinary = /format binary_little_endian 1\.0/.test(header);
  const isAscii = /format ascii 1\.0/.test(header);
  if (!isBinary && !isAscii) throw new Error("PLY: only ascii and binary_little_endian are supported");
  const m = header.match(/element vertex (\d+)/);
  if (!m) throw new Error("PLY: vertex count not found");
  const n = parseInt(m[1], 10);
  const bodyStart = idx + marker.length; // header is ASCII: char index === byte index

  // vertex properties, in file order
  const props: { name: string; type: string }[] = [];
  let inVertex = false;
  for (const line of header.split("\n")) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "element") inVertex = parts[1] === "vertex";
    else if (inVertex && parts[0] === "property" && parts[1] !== "list") {
      props.push({ type: parts[1], name: parts[2] });
    }
  }
  const ix = props.findIndex((q) => q.name === "x");
  const iy = props.findIndex((q) => q.name === "y");
  const iz = props.findIndex((q) => q.name === "z");
  if (ix < 0 || iy < 0 || iz < 0) throw new Error("PLY: vertex x/y/z properties not found");

  if (isAscii) {
    const text = new TextDecoder().decode(bytes.subarray(bodyStart));
    const out = new Float32Array(n * 3);
    let pos = 0;
    let count = 0;
    while (count < n && pos < text.length) {
      let eol = text.indexOf("\n", pos);
      if (eol < 0) eol = text.length;
      const cols = text.slice(pos, eol).trim().split(/\s+/);
      pos = eol + 1;
      if (cols.length <= Math.max(ix, iy, iz)) continue; // blank/short line
      out[count * 3] = parseFloat(cols[ix]);
      out[count * 3 + 1] = parseFloat(cols[iy]);
      out[count * 3 + 2] = parseFloat(cols[iz]);
      count += 1;
    }
    if (count < n) throw new Error(`PLY: expected ${n} vertices, found ${count}`);
    return out;
  }

  // binary_little_endian
  const offsets: number[] = [];
  let stride = 0;
  for (const q of props) {
    const size = PLY_SIZES[q.type];
    if (!size) throw new Error(`PLY: unsupported vertex property type ${q.type}`);
    offsets.push(stride);
    stride += size;
  }
  const abs = bytes.byteOffset + bodyStart;
  const xyzOnlyFloat32 =
    props.length === 3 && ix === 0 && iy === 1 && iz === 2 &&
    props.every((q) => PLY_SIZES[q.type] === 4 && q.type.startsWith("float"));
  if (xyzOnlyFloat32 && abs % 4 === 0) {
    return new Float32Array(bytes.buffer, abs, n * 3); // zero-copy
  }
  if (bytes.byteLength < bodyStart + n * stride) throw new Error("PLY: truncated vertex data");
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const read = (off: number, type: string) =>
    PLY_SIZES[type] === 8 ? dv.getFloat64(off, true) : dv.getFloat32(off, true);
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const base = bodyStart + i * stride;
    out[i * 3] = read(base + offsets[ix], props[ix].type);
    out[i * 3 + 1] = read(base + offsets[iy], props[iy].type);
    out[i * 3 + 2] = read(base + offsets[iz], props[iz].type);
  }
  return out;
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
