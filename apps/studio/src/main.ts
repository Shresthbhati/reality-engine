import { RealityStudioClient, StudioApiError } from './api/client';
import type {
  EntityConfidenceResponse,
  EntityProvenanceResponse,
  WorkflowStage,
  WorldDiffResponse,
  WorldIRPayload,
  WorldSummary,
  WorldVersionInfo,
} from './types/studio';
import { StudioScene3D, type LayerVisibility } from './viewer/scene3d';

const STAGES: Array<{ id: WorkflowStage; label: string }> = [
  { id: 'project', label: '1. Project' },
  { id: 'upload', label: '2. Upload' },
  { id: 'reconstruct', label: '3. Reconstruct' },
  { id: 'inspect', label: '4. Inspect' },
  { id: 'correct', label: '5. Correct' },
  { id: 'compare', label: '6. Compare' },
  { id: 'export', label: '7. Export' },
];

const client = new RealityStudioClient(
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE_URL ?? '/api',
);

const state: {
  stage: WorkflowStage;
  worlds: WorldSummary[];
  worldId: string;
  world: WorldIRPayload | null;
  versions: WorldVersionInfo[];
  selectedEntityId: string | null;
  provenance: EntityProvenanceResponse | null;
  confidence: EntityConfidenceResponse | null;
  diff: WorldDiffResponse | null;
  statusMessage: string;
  layers: LayerVisibility;
} = {
  stage: 'project',
  worlds: [],
  worldId: '',
  world: null,
  versions: [],
  selectedEntityId: null,
  provenance: null,
  confidence: null,
  diff: null,
  statusMessage: 'Connect to Reality Engine API and select a world.',
  layers: {
    geometry: true,
    pointCloud: true,
    cameras: true,
    uncertaintyColoring: false,
  },
};

let scene3d: StudioScene3D | null = null;

function esc(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatConfidence(val: number | null | undefined): string {
  return val == null ? 'Not measured' : `${(val * 100).toFixed(1)}%`;
}

async function selectEntity(entityId: string | null): Promise<void> {
  state.selectedEntityId = entityId;
  state.provenance = null;
  state.confidence = null;
  if (entityId && state.worldId) {
    try {
      const [prov, conf] = await Promise.all([
        client.getEntityProvenance(state.worldId, entityId),
        client.getEntityConfidence(state.worldId, entityId),
      ]);
      state.provenance = prov;
      state.confidence = conf;
    } catch (err) {
      state.statusMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
    }
  }
  renderSidebar();
}

async function loadWorld(worldId: string): Promise<void> {
  state.worldId = worldId;
  state.selectedEntityId = null;
  try {
    const [world, versions] = await Promise.all([
      client.getWorldIR(worldId),
      client.getVersions(worldId),
    ]);
    state.world = world;
    state.versions = versions;
    scene3d?.setWorld(world, state.layers.uncertaintyColoring);
    scene3d?.setLayerVisibility(state.layers);

    // Optional point cloud and camera overlays
    client.getPoints(worldId).then((pts) => scene3d?.setPointCloud(pts)).catch(() => {});
    client.getCameras(worldId).then((cams) => scene3d?.setCameras(cams.cameras ?? [])).catch(() => {});

    state.statusMessage = `Loaded ${Object.keys(world.entities ?? {}).length} entities (${versions.length} versions).`;
  } catch (err) {
    state.statusMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
  }
  renderSidebar();
}

function renderWorkflowBar(): void {
  const bar = document.getElementById('workflow-bar');
  if (!bar) return;
  bar.innerHTML = STAGES.map(
    (s) =>
      `<button class="stage-btn ${state.stage === s.id ? 'active' : ''}" data-stage="${s.id}">${esc(s.label)}</button>`,
  ).join('');
  bar.querySelectorAll<HTMLButtonElement>('.stage-btn').forEach((btn) => {
    btn.onclick = () => {
      state.stage = (btn.dataset.stage as WorkflowStage) || 'inspect';
      renderWorkflowBar();
      renderSidebar();
    };
  });
}

function renderSidebar(): void {
  const panel = document.getElementById('side-panel');
  const statusEl = document.getElementById('status-bar');
  if (statusEl) statusEl.textContent = state.statusMessage;
  if (!panel) return;

  const entity = state.selectedEntityId && state.world ? state.world.entities[state.selectedEntityId] : null;

  panel.innerHTML = `
    <section class="card">
      <h3>Stage: ${esc(state.stage.toUpperCase())}</h3>
      <div class="row">
        <select id="world-select">
          <option value="">-- Select World --</option>
          ${state.worlds.map((w) => `<option value="${esc(w.id)}" ${w.id === state.worldId ? 'selected' : ''}>${esc(w.name)} (${esc(w.id)})</option>`).join('')}
        </select>
        <button id="refresh-worlds-btn">Refresh</button>
      </div>
      <div class="row layers">
        <label><input type="checkbox" id="lyr-geo" ${state.layers.geometry ? 'checked' : ''}/> Geometry</label>
        <label><input type="checkbox" id="lyr-pts" ${state.layers.pointCloud ? 'checked' : ''}/> Points</label>
        <label><input type="checkbox" id="lyr-cam" ${state.layers.cameras ? 'checked' : ''}/> Cameras</label>
        <label><input type="checkbox" id="lyr-unc" ${state.layers.uncertaintyColoring ? 'checked' : ''}/> Confidence</label>
      </div>
    </section>

    <section class="card">
      <h3>Evidence &amp; Transparency</h3>
      ${
        entity
          ? `
        <div><strong>${esc(entity.name || entity.id)}</strong> (<code>${esc(entity.type)}</code>)</div>
        <div>Provenance: <code>${esc(entity.provenance ?? 'unknown')}</code></div>
        <div>Confidence: <strong>${esc(formatConfidence(state.confidence?.confidence ?? entity.confidence))}</strong></div>
        <div>Trace level: <code>${esc(state.provenance?.trace_level ?? 'loading...')}</code></div>
        <div>Supporting observations: ${state.provenance?.observations?.length ?? 0}</div>
        <div>Source sessions: ${state.provenance?.source_sessions?.length ?? 0}</div>
        <div>Relationships: ${entity.relationships?.length ?? 0}</div>
        `
          : '<div class="muted">Click any 3D entity in the viewport to inspect supporting observations, engines, and uncertainty.</div>'
      }
    </section>

    <section class="card">
      <h3>Corrections (Auditable Version)</h3>
      <div class="row">
        <input id="reclass-input" placeholder="New ontology class (e.g. architecture/door)" />
        <button id="reclass-btn" ${entity ? '' : 'disabled'}>Reclassify</button>
      </div>
    </section>

    <section class="card">
      <h3>Version Diff</h3>
      <button id="diff-btn" ${state.worldId ? '' : 'disabled'}>Compare Latest vs Parent</button>
      ${
        state.diff
          ? `<div class="diff-box">
              <div>Added: ${state.diff.added_entities.length}</div>
              <div>Removed: ${state.diff.removed_entities.length}</div>
              <div>Changed: ${state.diff.changed_entities.length}</div>
            </div>`
          : '<div class="muted">Run diff to inspect added, removed, and changed entities.</div>'
      }
    </section>

    <section class="card">
      <h3>Export</h3>
      <div class="row">
        <select id="export-fmt">
          <option value="gltf">glTF (.gltf)</option>
          <option value="obj">OBJ (.obj)</option>
          <option value="ifc">IFC (.ifc)</option>
        </select>
        <button id="export-btn" ${state.worldId ? '' : 'disabled'}>Export</button>
      </div>
    </section>
  `;

  const worldSel = document.getElementById('world-select') as HTMLSelectElement | null;
  if (worldSel) {
    worldSel.onchange = () => {
      if (worldSel.value) void loadWorld(worldSel.value);
    };
  }
  const refreshBtn = document.getElementById('refresh-worlds-btn');
  if (refreshBtn) refreshBtn.onclick = () => void refreshWorlds();

  (['geo', 'pts', 'cam', 'unc'] as const).forEach((k) => {
    const el = document.getElementById(`lyr-${k}`) as HTMLInputElement | null;
    if (!el) return;
    el.onchange = () => {
      if (k === 'geo') state.layers.geometry = el.checked;
      if (k === 'pts') state.layers.pointCloud = el.checked;
      if (k === 'cam') state.layers.cameras = el.checked;
      if (k === 'unc') {
        state.layers.uncertaintyColoring = el.checked;
        if (state.world) scene3d?.setWorld(state.world, el.checked);
      }
      scene3d?.setLayerVisibility(state.layers);
    };
  });

  const reclassBtn = document.getElementById('reclass-btn');
  if (reclassBtn && entity) {
    reclassBtn.onclick = async () => {
      const inp = document.getElementById('reclass-input') as HTMLInputElement | null;
      const newClass = inp?.value.trim();
      if (!newClass) return;
      try {
        await client.reclassifyEntity(state.worldId, entity.id, newClass, 'Studio correction');
        state.statusMessage = `Reclassified ${entity.id} -> ${newClass}`;
        await loadWorld(state.worldId);
      } catch (err) {
        state.statusMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
        renderSidebar();
      }
    };
  }

  const diffBtn = document.getElementById('diff-btn');
  if (diffBtn && state.worldId) {
    diffBtn.onclick = async () => {
      try {
        state.diff = await client.getDiff(state.worldId);
        state.statusMessage = 'Loaded version diff.';
      } catch (err) {
        state.statusMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
      }
      renderSidebar();
    };
  }

  const exportBtn = document.getElementById('export-btn');
  if (exportBtn && state.worldId) {
    exportBtn.onclick = async () => {
      const fmt = (document.getElementById('export-fmt') as HTMLSelectElement | null)?.value ?? 'gltf';
      try {
        const res = await client.exportWorld(state.worldId, fmt);
        state.statusMessage = `Exported ${fmt}: ${res.content_hash}`;
      } catch (err) {
        state.statusMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
      }
      renderSidebar();
    };
  }
}

async function refreshWorlds(): Promise<void> {
  try {
    state.worlds = await client.listWorlds();
    state.statusMessage = `Found ${state.worlds.length} worlds.`;
  } catch (err) {
    state.statusMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
  }
  renderSidebar();
}

function mount(): void {
  const root = document.getElementById('app');
  if (!root) return;
  root.innerHTML = `
    <header class="topbar">
      <div class="brand">Reality Studio</div>
      <nav id="workflow-bar"></nav>
    </header>
    <main class="workspace">
      <div id="viewport" class="viewport"></div>
      <aside id="side-panel" class="side-panel"></aside>
    </main>
    <footer id="status-bar" class="status-bar"></footer>
  `;

  const vp = document.getElementById('viewport');
  if (vp) {
    scene3d = new StudioScene3D(vp, (eid) => {
      void selectEntity(eid);
    });
  }
  renderWorkflowBar();
  renderSidebar();
  void refreshWorlds();
}

mount();
