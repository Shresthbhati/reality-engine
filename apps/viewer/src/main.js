// Reality Engine Studio Viewer (minimum inspection surface, Phase 21).
//
// Consumes ONLY real pipeline artifacts produced by
// scripts/run_vertical_slice.py:
//   worldir.json  - canonical WorldIR (entities, geometry, provenance)
//   points.ply    - real reconstructed points (binary float32 xyz, meters)
//   cameras.json  - registered camera poses (camera-to-world wxyz quats)
//
// When built with build_viewer.py --data, the artifacts are embedded as
// base64 and no network fetch happens at all. Otherwise the viewer tries
// to fetch the three files next to index.html, and falls back to a
// drag-and-drop / file picker so any dataset can be inspected offline.
//
// Honesty rules mirrored from the engine:
//   - only persisted data is shown; nothing is synthesized
//   - confidence is visible (entity shading + inspector), never hidden
//   - planes with absurd bounds render as absurd boxes ON PURPOSE:
//     reconstruction noise should be visible, not prettified away

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const EMBED = window.REALITY_EMBED || null; // { worldir, points_ply, cameras } base64 or null

// ---------------------------------------------------------------- state
const state = {
  world: null,
  points: null,          // Float32Array
  cameras: null,
  selected: null,        // entity id
  show: { points: true, entities: true, cameras: true },
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------- loaders
function b64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function parsePly(bytes) {
  // minimal binary_little_endian float32 xyz reader produced by the runner
  const head = new TextDecoder().decode(bytes.subarray(0, 2048));
  const marker = 'end_header\n';
  const idx = head.indexOf(marker);
  if (idx < 0) throw new Error('PLY: end_header not found');
  const header = head.slice(0, idx);
  if (!/format binary_little_endian 1\.0/.test(header)) {
    throw new Error('PLY: only binary_little_endian supported');
  }
  const m = header.match(/element vertex (\d+)/);
  if (!m) throw new Error('PLY: vertex count not found');
  const n = parseInt(m[1], 10);
  const data = new Float32Array(bytes.buffer, bytes.byteOffset + idx + marker.length, n * 3);
  return data;
}

// Triangle-mesh PLY (MeshData.to_ply_bytes output): binary float xyz
// vertices + `element face` triangle indices. Returns {positions, indices}
// or null when the payload has no faces.
function parseMeshPly(bytes) {
  const head = new TextDecoder().decode(bytes.subarray(0, 4096));
  const marker = 'end_header\n';
  const idx = head.indexOf(marker);
  if (idx < 0) throw new Error('PLY: end_header not found');
  const header = head.slice(0, idx);
  if (!/format binary_little_endian 1\.0/.test(header)) {
    throw new Error('PLY: only binary_little_endian supported');
  }
  const vm = header.match(/element vertex (\d+)/);
  const fm = header.match(/element face (\d+)/);
  if (!vm) throw new Error('PLY: vertex count not found');
  const nv = parseInt(vm[1], 10);
  const nf = fm ? parseInt(fm[1], 10) : 0;
  // stride from declared vertex property sizes (nx/ny/nz, rgb possible)
  const sizes = { float: 4, double: 8, uchar: 1, char: 1, ushort: 2, short: 2, uint: 4, int: 4 };
  let stride = 0;
  let inVertex = false;
  let countFmt = 'uchar';
  let indexType = 'uint';
  for (const line of header.split('\n')) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === 'element') { inVertex = parts[1] === 'vertex'; continue; }
    if (!inVertex && parts[0] === 'property' && parts[1] === 'list') { countFmt = parts[2]; indexType = parts[3]; continue; }
    if (inVertex && parts[0] === 'property' && sizes[parts[1]]) stride += sizes[parts[1]];
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
    let count;
    if (countFmt === 'uchar') count = dv.getUint8(off);
    else if (countFmt === 'ushort') count = dv.getUint16(off, true);
    else count = dv.getInt32(off, true);
    off += countSize;
    if (count === 3 && off + indexSize * 3 <= bytes.byteLength) {
      for (let k = 0; k < 3; k++) {
        indices[fi++] = indexSize === 2 ? dv.getUint16(off, true) : dv.getUint32(off, true);
        off += indexSize;
      }
    } else {
      off += indexSize * count; // skip non-triangle faces honestly
    }
  }
  if (fi === 0) return null;
  return { positions, indices: indices.subarray(0, fi) };
}

async function loadArtifacts() {
  setProgress('decoding world…', 15);
  if (EMBED) {
    state.world = JSON.parse(new TextDecoder().decode(b64ToBytes(EMBED.worldir)));
    setProgress('decoding point cloud…', 45);
    state.points = EMBED.points_ply ? parsePly(b64ToBytes(EMBED.points_ply)) : null;
    state.cameras = EMBED.cameras ? JSON.parse(new TextDecoder().decode(b64ToBytes(EMBED.cameras))) : null;
    state.mesh = EMBED.mesh_ply ? parseMeshPly(b64ToBytes(EMBED.mesh_ply)) : null;
  } else {
    state.world = await (await fetch('./worldir.json')).json();
    setProgress('fetching point cloud…', 45);
    try {
      const r = await fetch('./points.ply');
      state.points = r.ok ? parsePly(new Uint8Array(await r.arrayBuffer())) : null;
    } catch { state.points = null; }
    try {
      const r = await fetch('./cameras.json');
      state.cameras = r.ok ? await r.json() : null;
    } catch { state.cameras = null; }
    try {
      const r = await fetch('./mesh.ply');
      state.mesh = r.ok ? parseMeshPly(new Uint8Array(await r.arrayBuffer())) : null;
    } catch { state.mesh = null; }
  }
  setProgress('building scene…', 75);
}

// --------------------------------------------------------------- scene
let renderer, scene, cameraCtl, persp;
const layers = { points: null, entities: null, cameras: null, mesh: null };
const entityMeshes = new Map(); // entity id -> mesh

const TYPE_COLORS = { floor: 0x4da3ff, wall: 0xffb84d, ceiling: 0xb28dff };

function pointsBounds(pts) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pts.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const v = pts[i + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, max };
}

function robustPointsBounds(pts) {
  // 2nd/98th percentile per axis on a strided sample: depth-noise
  // outliers must not stretch the default framing.
  const stride = Math.max(1, Math.floor(pts.length / 3 / 20000));
  const cols = [[], [], []];
  for (let i = 0; i < pts.length; i += 3 * stride) {
    for (let k = 0; k < 3; k++) cols[k].push(pts[i + k]);
  }
  const min = [], max = [];
  for (let k = 0; k < 3; k++) {
    const c = cols[k].sort((a, b) => a - b);
    min.push(c[Math.floor(0.02 * (c.length - 1))]);
    max.push(c[Math.ceil(0.98 * (c.length - 1))]);
  }
  return { min, max };
}

function buildScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101418);
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(5, 12, 3);
  scene.add(sun);

  persp = new THREE.PerspectiveCamera(55, 1.0, 0.05, 4000);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  $('viewport').appendChild(renderer.domElement);

  cameraCtl = new OrbitControls(persp, renderer.domElement);
  cameraCtl.enableDamping = true;

  // evidence anchor: the real reconstructed points, colored by height
  if (state.points && state.points.length) {
    const b = pointsBounds(state.points);
    const span = Math.max(1e-6, b.max[1] - b.min[1]);
    const colors = new Float32Array(state.points.length);
    for (let i = 0; i < state.points.length; i += 3) {
      const t = (state.points[i + 1] - b.min[1]) / span; // y in [0,1]
      colors[i] = 0.35 + 0.60 * t;      // r
      colors[i + 1] = 0.85;             // g
      colors[i + 2] = 1.0 - 0.45 * t;   // b
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(state.points, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    layers.points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.05, vertexColors: true, sizeAttenuation: true,
    }));
    scene.add(layers.points);
    // frame the EVIDENCE robustly (percentile bounds: outliers excluded),
    // not the (possibly noisy) entity boxes
    const rb = robustPointsBounds(state.points);
    const c = [(rb.min[0] + rb.max[0]) / 2, (rb.min[1] + rb.max[1]) / 2, (rb.min[2] + rb.max[2]) / 2];
    const r = Math.max(rb.max[0] - rb.min[0], rb.max[1] - rb.min[1], rb.max[2] - rb.min[2]);
    state.robustExtent = r;
    persp.position.set(c[0] + r * 1.4, c[1] + r * 0.8, c[2] + r * 1.4);
    cameraCtl.target.set(c[0], c[1], c[2]);
  } else {
    persp.position.set(4, 3, 6);
  }

  // real reconstructed surface mesh (when the run produced one)
  if (state.mesh && state.mesh.positions.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(state.mesh.positions, 3));
    geo.setIndex(new THREE.BufferAttribute(state.mesh.indices, 1));
    layers.mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({
      color: 0x8fb8d8, side: THREE.DoubleSide, transparent: true, opacity: 0.55,
    }));
    scene.add(layers.mesh);
  }

  buildEntityLayer();
  buildCameraLayer();

  const grid = new THREE.GridHelper(20, 20, 0x2a3340, 0x1a212b);
  scene.add(grid);

  window.addEventListener('resize', onResize);
  // the grid layout can change the viewport size without a window resize
  if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(onResize).observe($('viewport'));
  }
  onResize();
  renderer.setAnimationLoop(() => { cameraCtl.update(); renderer.render(scene, persp); });
}

function buildEntityLayer() {
  layers.entities = new THREE.Group();
  const entities = Object.values(state.world.entities || {});
  for (const e of entities) {
    const g = (state.world.geometries || {})[(e.geometry_ids || [])[0]];
    if (!g || !g.bounds_min || !g.bounds_max) continue;
    const mn = [g.bounds_min.x, g.bounds_min.y, g.bounds_min.z];
    const mx = [g.bounds_max.x, g.bounds_max.y, g.bounds_max.z];
    const size = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
    if (size.some((s) => !isFinite(s) || s <= 0)) continue; // invalid: skip box, still listable
    const color = TYPE_COLORS[e.type] || 0x9aa7b5;
    const conf = typeof e.confidence === 'number' ? e.confidence : 0.5;
    // Reconstruction noise stays visible but must not dominate: boxes
    // far larger than the robust evidence extent render much fainter.
    const extent = Math.max(size[0], size[1], size[2]);
    const outlierScale = state.robustExtent && extent > 4 * state.robustExtent ? 0.3 : 1.0;
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(size[0], size[1], size[2]),
      new THREE.MeshLambertMaterial({
        color, transparent: true,
        opacity: (0.10 + 0.28 * conf) * outlierScale, // low-confidence geometry is fainter
        depthWrite: false,
      }),
    );
    mesh.position.set((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2);
    mesh.userData.entityId = e.id;
    // Depth-noise planes (extent >> robust evidence extent) start hidden:
    // reconstruction noise must be inspectable, but not bury the scene.
    // The 'oversized' toolbar toggle reveals them.
    mesh.visible = outlierScale !== 0.3; // oversized -> hidden by default
    layers.entities.add(mesh);
    entityMeshes.set(e.id, mesh);
  }
  scene.add(layers.entities);

  const ray = new THREE.Raycaster();
  renderer.domElement.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0) return;
    const rect = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((ev.clientX - rect.left) / rect.width) * 2 - 1,
      -((ev.clientY - rect.top) / rect.height) * 2 + 1,
    );
    ray.setFromCamera(ndc, persp);
    const hits = ray.intersectObjects(layers.entities.children, false);
    selectEntity(hits.length ? hits[0].object.userData.entityId : null);
  });
}

function buildCameraLayer() {
  if (!state.cameras || !state.cameras.cameras) return;
  layers.cameras = new THREE.Group();
  const positions = [];
  const markerGeo = new THREE.SphereGeometry(0.03, 10, 10);
  const markerMat = new THREE.MeshBasicMaterial({ color: 0x35d07f });
  const quat = new THREE.Quaternion();
  const imgSize = state.cameras.image_size || [1280, 960];
  const cam = new THREE.PerspectiveCamera(55, imgSize[0] / imgSize[1], 0.01, 10);
  const corner = new THREE.Vector3();
  const dist = 0.6;
  for (const c of state.cameras.cameras) {
    const p = c.position_m;
    quat.set(c.rotation_wxyz[0], c.rotation_wxyz[1], c.rotation_wxyz[2], c.rotation_wxyz[3]);
    cam.position.set(p[0], p[1], p[2]);
    cam.quaternion.copy(quat);
    cam.updateMatrixWorld(true);
    const frustum = [];
    for (const [nx, ny] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      corner.set(nx, ny, 0.5).unproject(cam);
      const d = corner.sub(cam.position).normalize().multiplyScalar(dist);
      frustum.push([p[0] + d.x, p[1] + d.y, p[2] + d.z]);
    }
    for (let i = 0; i < 4; i++) {
      positions.push(p[0], p[1], p[2], frustum[i][0], frustum[i][1], frustum[i][2]);
      const j = (i + 1) % 4;
      positions.push(frustum[i][0], frustum[i][1], frustum[i][2], frustum[j][0], frustum[j][1], frustum[j][2]);
    }
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.position.set(p[0], p[1], p[2]);
    layers.cameras.add(marker);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  layers.cameras.add(new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ color: 0x35d07f, transparent: true, opacity: 0.5 })));
  scene.add(layers.cameras);
}

function onResize() {
  const el = $('viewport');
  const w = el.clientWidth, h = el.clientHeight;
  renderer.setSize(w, h, false);
  persp.aspect = w / Math.max(1, h);
  persp.updateProjectionMatrix();
}

// ------------------------------------------------------------------ UI
function setProgress(msg, pct) {
  $('loading-msg').textContent = msg;
  $('loading-bar').style.width = `${pct}%`;
}

function selectEntity(id) {
  state.selected = id;
  for (const [eid, mesh] of entityMeshes) {
    const sel = eid === id;
    mesh.material.emissive = new THREE.Color(sel ? 0xffffff : 0x000000);
    mesh.material.emissiveIntensity = sel ? 0.55 : 0;
  }
  renderInspector();
  renderEntityList(); // re-render for highlight
  if (id) {
    const mesh = entityMeshes.get(id);
    if (mesh) {
      cameraCtl.target.copy(mesh.position);
    }
  }
}

function flyToEntity(id) {
  const mesh = entityMeshes.get(id);
  if (!mesh) return;
  const box = new THREE.Box3().setFromObject(mesh);
  const size = box.getSize(new THREE.Vector3()).length();
  const dir = new THREE.Vector3(0.8, 0.55, 0.8).normalize();
  const d = Math.max(1.2, size * 1.4);
  persp.position.copy(mesh.position).addScaledVector(dir, d);
  cameraCtl.target.copy(mesh.position);
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function renderWorldSummary() {
  const w = state.world;
  const meta = w.metadata || {};
  const nEnt = Object.keys(w.entities || {}).length;
  const nPts = state.points ? state.points.length / 3 : 0;
  const nCam = state.cameras ? (state.cameras.cameras || []).length : 0;
  $('world-summary').innerHTML = `
    <div class="k">world</div><div class="v mono">${esc(w.id)}</div>
    <div class="k">entities</div><div class="v">${nEnt}</div>
    <div class="k">points</div><div class="v">${nPts.toLocaleString()}</div>
    <div class="k">cameras</div><div class="v">${nCam}</div>
    <div class="k">scale</div><div class="v">${esc(meta.scale?.state ?? '?')}</div>
    <div class="k">frame</div><div class="v small">${esc(meta.frame?.note ?? '—')}</div>
  `;
  $('world-provenance').innerHTML = `
    <div class="prov-row"><span class="tag tag-scale">scale</span> ${esc(meta.scale?.note ?? '—')}</div>
    <div class="prov-row"><span class="tag tag-recon">recon</span> ${esc(meta.reconstruction?.backend ?? '?')} · ${esc(meta.reconstruction?.registration_status ?? '?')} · ${esc(meta.reconstruction?.cameras_registered)}/${esc(meta.reconstruction?.cameras_input)} cameras · ${esc(meta.reconstruction?.points)} pts</div>
    <div class="prov-row"><span class="tag tag-depth">depth</span> ${esc(meta.depth?.status ?? '—')} ${esc(meta.depth?.model ?? '')} · ${esc(meta.depth?.metricized)}/${esc(meta.depth?.maps)} maps · ${esc(meta.depth?.dense_points)} dense pts</div>
  `;
}

function renderEntityList() {
  const filter = ($('entity-filter').value || '').toLowerCase();
  const entities = Object.values(state.world.entities || {})
    .filter((e) => !filter || e.id.toLowerCase().includes(filter) || e.type.toLowerCase().includes(filter))
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  $('entity-list').innerHTML = entities.map((e) => `
    <div class="entity-row ${e.id === state.selected ? 'selected' : ''}" data-id="${esc(e.id)}">
      <span class="dot" style="background:#${(TYPE_COLORS[e.type] || 0x9aa7b5).toString(16).padStart(6, '0')}"></span>
      <span class="eid mono">${esc(e.id)}</span>
      <span class="etype">${esc(e.type)}</span>
      <span class="conf">${e.confidence != null ? (e.confidence * 100).toFixed(0) + '%' : '—'}</span>
    </div>`).join('');
  for (const row of $('entity-list').querySelectorAll('.entity-row')) {
    row.addEventListener('click', () => { selectEntity(row.dataset.id); flyToEntity(row.dataset.id); });
  }
}

function renderInspector() {
  const el = $('inspector');
  if (!state.selected) {
    el.innerHTML = '<div class="empty">Select an entity in the viewport or the list to inspect it — including <b>where it came from</b>.</div>';
    return;
  }
  const e = state.world.entities[state.selected];
  const g = (state.world.geometries || {})[(e.geometry_ids || [])[0]];
  const obs = e.observations || [];
  const conf = e.confidence;
  el.innerHTML = `
    <div class="insp-head">
      <span class="dot" style="background:#${(TYPE_COLORS[e.type] || 0x9aa7b5).toString(16).padStart(6, '0')}"></span>
      <b class="mono">${esc(e.id)}</b> <span class="etype">${esc(e.type)}</span>
    </div>
    <div class="confbar"><div style="width:${((conf ?? 0) * 100).toFixed(0)}%"></div></div>
    <div class="grid">
      <div class="k">confidence</div><div class="v">${conf != null ? (conf * 100).toFixed(1) + '%' : '—'}</div>
      <div class="k">provenance</div><div class="v"><span class="tag tag-${esc((e.provenance || '').toLowerCase())}">${esc(e.provenance || '—')}</span></div>
      <div class="k">position</div><div class="v mono small">${e.transform?.position ? [e.transform.position.x, e.transform.position.y, e.transform.position.z].map((v) => v.toFixed(3)).join(', ') : '—'}</div>
      ${g ? `<div class="k">bounds ext.</div><div class="v mono small">${['x', 'y', 'z'].map((k) => (g.bounds_max[k] - g.bounds_min[k]).toFixed(2)).join(' × ')} m</div>` : ''}
      ${g ? `<div class="k">geom conf.</div><div class="v">${g.confidence != null ? (g.confidence * 100).toFixed(0) + '%' : '—'} <span class="dim">(${esc(g.provenance || '')})</span></div>` : ''}
    </div>
    <div class="sect">observations (${obs.length}) — where this came from</div>
    ${obs.map((o) => `
      <div class="obs">
        <div class="obs-head mono">${esc(o.sensor_type)}</div>
        <div class="grid">
          ${Object.entries(o.metadata || {}).map(([k, v]) => `<div class="k">${esc(k)}</div><div class="v mono small">${esc(typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v)}</div>`).join('')}
        </div>
        ${o.confidence != null ? `<div class="dim">obs confidence ${(o.confidence * 100).toFixed(0)}%</div>` : ''}
      </div>`).join('')}
  `;
}

function wireToolbar() {
  const toggles = [['toggle-points', 'points'], ['toggle-entities', 'entities'], ['toggle-cameras', 'cameras']];
  // the surface mesh follows the points toggle when no dedicated toggle exists
  if (layers.mesh) { layers.mesh.visible = state.show.points !== false; }
  for (const [btn, key] of toggles) {
    $(btn).addEventListener('change', () => {
      state.show[key] = $(btn).checked;
      if (layers[key]) layers[key].visible = state.show[key];
    });
  }
  $('toggle-oversized').addEventListener('change', () => {
    const show = $('toggle-oversized').checked;
    for (const mesh of entityMeshes.values()) {
      const s = [mesh.geometry.parameters.width, mesh.geometry.parameters.height, mesh.geometry.parameters.depth];
      const extent = Math.max(...s);
      const oversized = state.robustExtent && extent > 4 * state.robustExtent;
      mesh.visible = state.show.entities && (!oversized || show);
    }
  });
  $('entity-filter').addEventListener('input', renderEntityList);
  $('btn-frame-points').addEventListener('click', () => {
    if (!state.points) return;
    const b = robustPointsBounds(state.points);
    const c = [(b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2];
    const r = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    persp.position.set(c[0] + r * 0.9, c[1] + r * 0.55, c[2] + r * 0.9);
    cameraCtl.target.set(c[0], c[1], c[2]);
  });
}

// ---------------------------------------------------------------- boot
async function boot() {
  try {
    await loadArtifacts();
    setProgress('done', 100);
    buildScene();
    renderWorldSummary();
    renderEntityList();
    renderInspector();
    wireToolbar();
    $('loading').classList.add('hidden');
  } catch (err) {
    $('loading-msg').textContent = `failed to load artifacts: ${err.message}`;
    $('loading-bar').style.background = '#e05252';
  }
}
boot();
