/**
 * Zero-dependency pure TypeScript decoders and data model for 3D Gaussian Splats.
 *
 * Implements decoding for:
 * 1. Standard 32-byte chunk .splat binary format
 * 2. Inria 3DGS .ply binary (little-endian) and ASCII format
 */

export interface ParsedSplatCloud {
  count: number;
  positions: Float32Array; // 3 * count
  scales: Float32Array; // 3 * count
  rotations: Float32Array; // 4 * count (qx, qy, qz, qw)
  colors: Uint8Array; // 3 * count (r, g, b)
  opacities: Float32Array; // count (0..1)
  bounds: { min: [number, number, number]; max: [number, number, number] };
}

export type HybridRenderMode = "hybrid" | "splats_only" | "cad_only";

const SH_C0 = 0.28209479177387814;

/**
 * Format splat count for display (e.g. 15,200 -> "15.2k", 1,200,000 -> "1.2M").
 */
export function formatSplatCount(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}k`;
  }
  return String(count);
}

/**
 * Format bytes into human-readable memory/file size (KB, MB, GB).
 */
export function formatByteSize(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(0)} KB`;
  }
  return `${bytes} B`;
}

/**
 * Parse standard 32-byte chunk .splat binary buffer into typed arrays.
 */
export function parseSplatBuffer(buffer: ArrayBuffer): ParsedSplatCloud {
  const byteLength = buffer.byteLength;
  const count = Math.floor(byteLength / 32);

  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const rotations = new Float32Array(count * 4);
  const colors = new Uint8Array(count * 3);
  const opacities = new Float32Array(count);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  const dataView = new DataView(buffer);
  const uint8View = new Uint8Array(buffer);

  for (let i = 0; i < count; i++) {
    const offset = i * 32;

    // Positions (Float32 x, y, z)
    const x = dataView.getFloat32(offset, true);
    const y = dataView.getFloat32(offset + 4, true);
    const z = dataView.getFloat32(offset + 8, true);
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;

    // Scales (Float32 sx, sy, sz)
    scales[i * 3] = dataView.getFloat32(offset + 12, true);
    scales[i * 3 + 1] = dataView.getFloat32(offset + 16, true);
    scales[i * 3 + 2] = dataView.getFloat32(offset + 20, true);

    // Color & Opacity (Uint8 r, g, b, a)
    colors[i * 3] = uint8View[offset + 24];
    colors[i * 3 + 1] = uint8View[offset + 25];
    colors[i * 3 + 2] = uint8View[offset + 26];
    opacities[i] = uint8View[offset + 27] / 255.0;

    // Rotations (Uint8 qx, qy, qz, qw where q = (byte - 128) / 128.0)
    const qx = (uint8View[offset + 28] - 128) / 128.0;
    const qy = (uint8View[offset + 29] - 128) / 128.0;
    const qz = (uint8View[offset + 30] - 128) / 128.0;
    const qw = (uint8View[offset + 31] - 128) / 128.0;

    const norm = Math.hypot(qx, qy, qz, qw) || 1.0;
    rotations[i * 4] = qx / norm;
    rotations[i * 4 + 1] = qy / norm;
    rotations[i * 4 + 2] = qz / norm;
    rotations[i * 4 + 3] = qw / norm;
  }

  if (count === 0) {
    minX = minY = minZ = maxX = maxY = maxZ = 0;
  }

  return {
    count,
    positions,
    scales,
    rotations,
    colors,
    opacities,
    bounds: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    },
  };
}

/**
 * Parse Inria 3DGS .ply buffer (binary little endian or ASCII).
 */
export function parsePlySplatBuffer(buffer: ArrayBuffer): ParsedSplatCloud {
  const textDecoder = new TextDecoder("ascii");
  // Check header within first 4096 bytes
  const headerPreview = textDecoder.decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 4096)));

  if (!headerPreview.startsWith("ply")) {
    throw new Error("Invalid PLY format: missing 'ply' signature");
  }

  const endHeaderIdx = headerPreview.indexOf("end_header");
  if (endHeaderIdx === -1) {
    throw new Error("Invalid PLY format: missing 'end_header'");
  }

  const headerStr = headerPreview.substring(0, endHeaderIdx);
  const isBinary = headerStr.includes("format binary_little_endian");
  const isAscii = headerStr.includes("format ascii");

  if (!isBinary && !isAscii) {
    throw new Error("Unsupported PLY format: only binary_little_endian and ascii are supported");
  }

  let vertexCount = 0;
  const properties: Array<{ name: string; type: string; byteSize: number }> = [];

  let inVertex = false;
  for (const line of headerStr.split(/\r?\n/)) {
    const tokens = line.trim().split(/\s+/);
    if (!tokens[0]) continue;
    if (tokens[0] === "element" && tokens[1] === "vertex") {
      inVertex = true;
      vertexCount = parseInt(tokens[2], 10) || 0;
    } else if (tokens[0] === "element" && tokens[1] !== "vertex") {
      inVertex = false;
    } else if (tokens[0] === "property" && inVertex) {
      const type = tokens[1].toLowerCase();
      const name = tokens[2];
      const byteSize = type === "double" || type === "float64" ? 8 : type === "uchar" || type === "uint8" ? 1 : 4;
      properties.push({ name, type, byteSize });
    }
  }

  if (vertexCount <= 0 || properties.length === 0) {
    return {
      count: 0,
      positions: new Float32Array(0),
      scales: new Float32Array(0),
      rotations: new Float32Array(0),
      colors: new Uint8Array(0),
      opacities: new Float32Array(0),
      bounds: { min: [0, 0, 0], max: [0, 0, 0] },
    };
  }

  // Find exact byte offset of the end of "end_header\n"
  const rawUint8 = new Uint8Array(buffer);
  let headerEndByte = 0;
  const endHeaderToken = [101, 110, 100, 95, 104, 101, 97, 100, 101, 114]; // "end_header"
  for (let i = 0; i < Math.min(rawUint8.length - 10, 8192); i++) {
    let match = true;
    for (let j = 0; j < endHeaderToken.length; j++) {
      if (rawUint8[i + j] !== endHeaderToken[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      headerEndByte = i + endHeaderToken.length;
      while (headerEndByte < rawUint8.length && (rawUint8[headerEndByte] === 10 || rawUint8[headerEndByte] === 13)) {
        headerEndByte++;
      }
      break;
    }
  }

  const propIndexMap = new Map<string, { offset: number; type: string; byteSize: number }>();
  let recordStride = 0;
  for (const p of properties) {
    propIndexMap.set(p.name, { offset: recordStride, type: p.type, byteSize: p.byteSize });
    recordStride += p.byteSize;
  }

  const positions = new Float32Array(vertexCount * 3);
  const scales = new Float32Array(vertexCount * 3);
  const rotations = new Float32Array(vertexCount * 4);
  const colors = new Uint8Array(vertexCount * 3);
  const opacities = new Float32Array(vertexCount);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  if (isBinary) {
    const dataView = new DataView(buffer, headerEndByte);

    const hasLogScale = propIndexMap.has("scale_0") && propIndexMap.has("scale_1") && propIndexMap.has("scale_2");
    const hasLinearScale = propIndexMap.has("sx") && propIndexMap.has("sy") && propIndexMap.has("sz");
    const hasSH = propIndexMap.has("f_dc_0") && propIndexMap.has("f_dc_1") && propIndexMap.has("f_dc_2");
    const hasRGB = propIndexMap.has("red") && propIndexMap.has("green") && propIndexMap.has("blue");
    const hasInriaRot = propIndexMap.has("rot_0") && propIndexMap.has("rot_1") && propIndexMap.has("rot_2") && propIndexMap.has("rot_3");
    const hasQuat = propIndexMap.has("qx") && propIndexMap.has("qy") && propIndexMap.has("qz") && propIndexMap.has("qw");

    const xOff = propIndexMap.get("x")?.offset ?? 0;
    const yOff = propIndexMap.get("y")?.offset ?? 4;
    const zOff = propIndexMap.get("z")?.offset ?? 8;

    for (let i = 0; i < vertexCount; i++) {
      const recOff = i * recordStride;

      const x = dataView.getFloat32(recOff + xOff, true);
      const y = dataView.getFloat32(recOff + yOff, true);
      const z = dataView.getFloat32(recOff + zOff, true);
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;

      // Scales
      if (hasLogScale) {
        const s0 = dataView.getFloat32(recOff + propIndexMap.get("scale_0")!.offset, true);
        const s1 = dataView.getFloat32(recOff + propIndexMap.get("scale_1")!.offset, true);
        const s2 = dataView.getFloat32(recOff + propIndexMap.get("scale_2")!.offset, true);
        scales[i * 3] = Math.exp(Math.max(-20, Math.min(10, s0)));
        scales[i * 3 + 1] = Math.exp(Math.max(-20, Math.min(10, s1)));
        scales[i * 3 + 2] = Math.exp(Math.max(-20, Math.min(10, s2)));
      } else if (hasLinearScale) {
        scales[i * 3] = dataView.getFloat32(recOff + propIndexMap.get("sx")!.offset, true);
        scales[i * 3 + 1] = dataView.getFloat32(recOff + propIndexMap.get("sy")!.offset, true);
        scales[i * 3 + 2] = dataView.getFloat32(recOff + propIndexMap.get("sz")!.offset, true);
      } else {
        scales[i * 3] = 0.03;
        scales[i * 3 + 1] = 0.03;
        scales[i * 3 + 2] = 0.03;
      }

      // Rotations
      if (hasInriaRot) {
        // rot_0=qw, rot_1=qx, rot_2=qy, rot_3=qz
        const qw = dataView.getFloat32(recOff + propIndexMap.get("rot_0")!.offset, true);
        const qx = dataView.getFloat32(recOff + propIndexMap.get("rot_1")!.offset, true);
        const qy = dataView.getFloat32(recOff + propIndexMap.get("rot_2")!.offset, true);
        const qz = dataView.getFloat32(recOff + propIndexMap.get("rot_3")!.offset, true);
        const norm = Math.hypot(qx, qy, qz, qw) || 1.0;
        rotations[i * 4] = qx / norm;
        rotations[i * 4 + 1] = qy / norm;
        rotations[i * 4 + 2] = qz / norm;
        rotations[i * 4 + 3] = qw / norm;
      } else if (hasQuat) {
        const qx = dataView.getFloat32(recOff + propIndexMap.get("qx")!.offset, true);
        const qy = dataView.getFloat32(recOff + propIndexMap.get("qy")!.offset, true);
        const qz = dataView.getFloat32(recOff + propIndexMap.get("qz")!.offset, true);
        const qw = dataView.getFloat32(recOff + propIndexMap.get("qw")!.offset, true);
        const norm = Math.hypot(qx, qy, qz, qw) || 1.0;
        rotations[i * 4] = qx / norm;
        rotations[i * 4 + 1] = qy / norm;
        rotations[i * 4 + 2] = qz / norm;
        rotations[i * 4 + 3] = qw / norm;
      } else {
        rotations[i * 4] = 0.0;
        rotations[i * 4 + 1] = 0.0;
        rotations[i * 4 + 2] = 0.0;
        rotations[i * 4 + 3] = 1.0;
      }

      // Colors
      if (hasSH) {
        const f0 = dataView.getFloat32(recOff + propIndexMap.get("f_dc_0")!.offset, true);
        const f1 = dataView.getFloat32(recOff + propIndexMap.get("f_dc_1")!.offset, true);
        const f2 = dataView.getFloat32(recOff + propIndexMap.get("f_dc_2")!.offset, true);
        colors[i * 3] = Math.max(0, Math.min(255, Math.round((0.5 + SH_C0 * f0) * 255)));
        colors[i * 3 + 1] = Math.max(0, Math.min(255, Math.round((0.5 + SH_C0 * f1) * 255)));
        colors[i * 3 + 2] = Math.max(0, Math.min(255, Math.round((0.5 + SH_C0 * f2) * 255)));
      } else if (hasRGB) {
        const rInfo = propIndexMap.get("red")!;
        const gInfo = propIndexMap.get("green")!;
        const bInfo = propIndexMap.get("blue")!;
        colors[i * 3] = rInfo.byteSize === 1 ? rawUint8[headerEndByte + recOff + rInfo.offset] : Math.round(dataView.getFloat32(recOff + rInfo.offset, true));
        colors[i * 3 + 1] = gInfo.byteSize === 1 ? rawUint8[headerEndByte + recOff + gInfo.offset] : Math.round(dataView.getFloat32(recOff + gInfo.offset, true));
        colors[i * 3 + 2] = bInfo.byteSize === 1 ? rawUint8[headerEndByte + recOff + bInfo.offset] : Math.round(dataView.getFloat32(recOff + bInfo.offset, true));
      } else {
        colors[i * 3] = 200;
        colors[i * 3 + 1] = 200;
        colors[i * 3 + 2] = 200;
      }

      // Opacity
      if (propIndexMap.has("opacity")) {
        const rawOp = dataView.getFloat32(recOff + propIndexMap.get("opacity")!.offset, true);
        if (rawOp < 0.0 || rawOp > 1.0) {
          // Logit opacity
          opacities[i] = 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, rawOp))));
        } else {
          opacities[i] = rawOp;
        }
      } else {
        opacities[i] = 0.95;
      }
    }
  }

  return {
    count: vertexCount,
    positions,
    scales,
    rotations,
    colors,
    opacities,
    bounds: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
    },
  };
}
