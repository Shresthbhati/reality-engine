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

const TOKEN_KEY = 'reality_studio_token';

const client = new RealityStudioClient(
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE_URL ?? '/api',
  () => localStorage.getItem(TOKEN_KEY),
);

const state: {
  stage: WorkflowStage;
  userEmail: string | null;
  worlds: WorldSummary[];
  worldId: string;
  sessionId: string;
  world: WorldIRPayload | null;
  versions: WorldVersionInfo[];
  selectedEntityId: string | null;
  provenance: EntityProvenanceResponse | null;
  confidence: EntityConfidenceResponse | null;
  diff: WorldDiffResponse | null;
  loading: boolean;
  errorMessage: string | null;
  statusMessage: string;
  layers: LayerVisibility;
} = {
  stage: 'project',
  userEmail: null,
  worlds: [],
  worldId: '',
  sessionId: '',
  world: null,
  versions: [],
  selectedEntityId: null,
  provenance: null,
  confidence: null,
  diff: null,
  loading: false,
  errorMessage: null,
  statusMessage: 'Sign in or create a project to begin.',
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

function setError(err: unknown): void {
  state.errorMessage = err instanceof StudioApiError ? `API ${err.status}: ${err.message}` : String(err);
}

async function selectEntity(entityId: string | null): Promise<void> {
  state.selectedEntityId = entityId;
  state.provenance = null;
  state.confidence = null;
  state.errorMessage = null;
  if (entityId && state.worldId) {
    state.loading = true;
    renderSidebar();
    try {
      const [prov, conf] = await Promise.all([
        client.getEntityProvenance(state.worldId, entityId),
        client.getEntityConfidence(state.worldId, entityId),
      ]);
      state.provenance = prov;
      state.confidence = conf;
    } catch (err) {
      setError(err);
    } finally {
      state.loading = false;
    }
  }
  renderSidebar();
}

async function loadWorld(worldId: string): Promise<void> {
  state.worldId = worldId;
  state.selectedEntityId = null;
  state.errorMessage = null;
  state.loading = true;
  renderSidebar();
  try {
    const versions = await client.getVersions(worldId);
    state.versions = versions;
    if (versions.length === 0) {
      state.world = null;
      state.statusMessage = `World ${worldId} has 0 versions yet — upload images and reconstruct.`;
    } else {
      const world = await client.getWorldIR(worldId);
      state.world = world;
      scene3d?.setWorld(world, state.layers.uncertaintyColoring);
      scene3d?.setLayerVisibility(state.layers);

      client.getPoints(worldId).then((pts) => scene3d?.setPointCloud(pts)).catch(() => {});
      client.getCameras(worldId).then((cams) => scene3d?.setCameras(cams.cameras ?? [])).catch(() => {});

      state.statusMessage = `Loaded ${Object.keys(world.entities ?? {}).length} entities (${versions.length} versions).`;
    }
  } catch (err) {
    setError(err);
  } finally {
    state.loading = false;
  }
  renderSidebar();
}

async function refreshWorlds(): Promise<void> {
  state.loading = true;
  state.errorMessage = null;
  renderSidebar();
  try {
    state.worlds = await client.listWorlds();
    state.statusMessage = `Found ${state.worlds.length} worlds.`;
  } catch (err) {
    setError(err);
  } finally {
    state.loading = false;
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
  if (statusEl) {
    statusEl.textContent = state.loading ? `Loading... ${state.statusMessage}` : state.statusMessage;
    statusEl.setAttribute('data-loading', String(state.loading));
  }
  if (!panel) return;

  const entity = state.selectedEntityId && state.world ? state.world.entities[state.selectedEntityId] : null;
  const entityIds = Object.keys(state.world?.entities ?? {});

  panel.innerHTML = `
    ${state.errorMessage ? `<div class="card error-banner" data-error="true">${esc(state.errorMessage)}</div>` : ''}

    <section class="card">
      <h3>1. Account &amp; Project</h3>
      <div class="muted">${state.userEmail ? `Signed in: ${esc(state.userEmail)}` : 'Not signed in'}</div>
      <div class="row">
        <input id="auth-email" placeholder="Email" />
        <input id="auth-pass" type="password" placeholder="Password" />
      </div>
      <div class="row">
        <button id="login-btn">Log In</button>
        <button id="signup-btn">Sign Up</button>
      </div>
      <div class="row">
        <input id="new-world-name" placeholder="New project/world name" />
        <button id="create-world-btn">Create World</button>
      </div>
      <div class="row">
        <select id="world-select">
          <option value="">-- Select World --</option>
          ${state.worlds.map((w) => `<option value="${esc(w.id)}" ${w.id === state.worldId ? 'selected' : ''}>${esc(w.name)} (${esc(w.id)})</option>`).join('')}
        </select>
        <button id="refresh-worlds-btn">Refresh</button>
      </div>
    </section>

    <section class="card">
      <h3>2 &amp; 3. Upload &amp; Reconstruct</h3>
      <div class="muted">Session: <code>${esc(state.sessionId || 'none')}</code></div>
      <div class="row">
        <input id="upload-files" type="file" accept="image/*" multiple />
        <button id="upload-btn" ${state.worldId ? '' : 'disabled'}>Upload</button>
      </div>
      <div class="row">
        <button id="reconstruct-btn" ${state.worldId && state.sessionId ? '' : 'disabled'}>Reconstruct</button>
        <button id="status-btn" ${state.worldId ? '' : 'disabled'}>Check Status</button>
      </div>
    </section>

    <section class="card">
      <h3>4. 3D Layers &amp; Evidence</h3>
      <div class="row layers">
        <label><input type="checkbox" id="lyr-geo" ${state.layers.geometry ? 'checked' : ''}/> Geometry</label>
        <label><input type="checkbox" id="lyr-pts" ${state.layers.pointCloud ? 'checked' : ''}/> Points</label>
        <label><input type="checkbox" id="lyr-cam" ${state.layers.cameras ? 'checked' : ''}/> Cameras</label>
        <label><input type="checkbox" id="lyr-unc" ${state.layers.uncertaintyColoring ? 'checked' : ''}/> Confidence</label>
      </div>
      <div class="row">
        <select id="entity-select">
          <option value="">-- Inspect Entity (${entityIds.length}) --</option>
          ${entityIds.map((eid) => `<option value="${esc(eid)}" ${eid === state.selectedEntityId ? 'selected' : ''}>${esc(state.world?.entities[eid]?.name || eid)} (${esc(eid)})</option>`).join('')}
        </select>
      </div>
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
          : '<div class="muted">Select or click an entity to inspect provenance, observations, and uncertainty.</div>'
      }
    </section>

    <section class="card">
      <h3>5. Corrections</h3>
      <div class="row">
        <input id="reclass-input" placeholder="New ontology class (e.g. architecture/door)" />
        <button id="reclass-btn" ${entity ? '' : 'disabled'}>Reclassify</button>
      </div>
      <div class="row">
        <button id="split-btn" ${entity ? '' : 'disabled'}>Split X (0.5)</button>
      </div>
    </section>

    <section class="card">
      <h3>6 &amp; 7. Compare &amp; Export</h3>
      <div class="row">
        <button id="diff-btn" ${state.worldId ? '' : 'disabled'}>Compare Versions</button>
      </div>
      ${
        state.diff
          ? `<div class="diff-box" data-diff="true">
              <div>Added: ${state.diff.added_entities.length}</div>
              <div>Removed: ${state.diff.removed_entities.length}</div>
              <div>Changed: ${state.diff.changed_entities.length}</div>
            </div>`
          : ''
      }
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

  const loginBtn = document.getElementById('login-btn');
  if (loginBtn) {
    loginBtn.onclick = async () => {
      const email = (document.getElementById('auth-email') as HTMLInputElement)?.value.trim();
      const pass = (document.getElementById('auth-pass') as HTMLInputElement)?.value;
      if (!email || !pass) return;
      try {
        const res = await client.login(email, pass);
        localStorage.setItem(TOKEN_KEY, res.token);
        state.userEmail = res.user.email;
        state.statusMessage = `Signed in as ${res.user.email}`;
        await refreshWorlds();
      } catch (err) {
        setError(err);
        renderSidebar();
      }
    };
  }

  const signupBtn = document.getElementById('signup-btn');
  if (signupBtn) {
    signupBtn.onclick = async () => {
      const email = (document.getElementById('auth-email') as HTMLInputElement)?.value.trim();
      const pass = (document.getElementById('auth-pass') as HTMLInputElement)?.value;
      if (!email || !pass) return;
      try {
        const res = await client.signup(email, pass, 'Reality Studio Org');
        localStorage.setItem(TOKEN_KEY, res.token);
        state.userEmail = res.user.email;
        state.statusMessage = `Registered & signed in as ${res.user.email}`;
        await refreshWorlds();
      } catch (err) {
        setError(err);
        renderSidebar();
      }
    };
  }

  const createWorldBtn = document.getElementById('create-world-btn');
  if (createWorldBtn) {
    createWorldBtn.onclick = async () => {
      const name = (document.getElementById('new-world-name') as HTMLInputElement)?.value.trim();
      if (!name) return;
      try {
        const w = await client.createWorld(name);
        state.worlds = [w, ...state.worlds];
        state.statusMessage = `Created world ${w.name} (${w.id})`;
        await loadWorld(w.id);
      } catch (err) {
        setError(err);
        renderSidebar();
      }
    };
  }

  const worldSel = document.getElementById('world-select') as HTMLSelectElement | null;
  if (worldSel) {
    worldSel.onchange = () => {
      if (worldSel.value) void loadWorld(worldSel.value);
    };
  }

  const entitySel = document.getElementById('entity-select') as HTMLSelectElement | null;
  if (entitySel) {
    entitySel.onchange = () => {
      void selectEntity(entitySel.value || null);
    };
  }

  const refreshBtn = document.getElementById('refresh-worlds-btn');
  if (refreshBtn) refreshBtn.onclick = () => void refreshWorlds();

  const uploadBtn = document.getElementById('upload-btn');
  if (uploadBtn && state.worldId) {
    uploadBtn.onclick = async () => {
      const fileInput = document.getElementById('upload-files') as HTMLInputElement | null;
      const files = Array.from(fileInput?.files ?? []);
      if (!files.length) return;
      state.loading = true;
      state.errorMessage = null;
      renderSidebar();
      try {
        if (!state.sessionId) {
          const sess = await client.createSession(`studio-${Date.now()}`);
          state.sessionId = sess.id;
          await client.attachSession(state.worldId, sess.id);
        }
        for (const f of files) {
          await client.uploadImage(state.sessionId, f);
        }
        state.statusMessage = `Uploaded ${files.length} image(s) to session ${state.sessionId}.`;
      } catch (err) {
        setError(err);
      } finally {
        state.loading = false;
      }
      renderSidebar();
    };
  }

  const reconBtn = document.getElementById('reconstruct-btn');
  if (reconBtn && state.worldId && state.sessionId) {
    reconBtn.onclick = async () => {
      try {
        const res = await client.triggerReconstruction(state.sessionId);
        state.statusMessage = `Reconstruction ${res.status} (job ${res.job_id})`;
      } catch (err) {
        setError(err);
      }
      renderSidebar();
    };
  }

  const statusBtn = document.getElementById('status-btn');
  if (statusBtn && state.worldId) {
    statusBtn.onclick = async () => {
      try {
        const res = await client.getReconstructionStatus(state.worldId);
        state.statusMessage = `Reconstruction status: ${res.status}`;
      } catch (err) {
        setError(err);
      }
      renderSidebar();
    };
  }

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
      const newClass = (document.getElementById('reclass-input') as HTMLInputElement | null)?.value.trim();
      if (!newClass) return;
      try {
        await client.reclassifyEntity(state.worldId, entity.id, newClass, 'Studio correction');
        state.statusMessage = `Reclassified ${entity.id} -> ${newClass}`;
        await loadWorld(state.worldId);
      } catch (err) {
        setError(err);
        renderSidebar();
      }
    };
  }

  const splitBtn = document.getElementById('split-btn');
  if (splitBtn && entity) {
    splitBtn.onclick = async () => {
      try {
        await client.splitEntity(state.worldId, entity.id, 'x', 0.5, 'Studio split');
        state.statusMessage = `Split ${entity.id} along X`;
        await loadWorld(state.worldId);
      } catch (err) {
        setError(err);
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
        setError(err);
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
        setError(err);
      }
      renderSidebar();
    };
  }
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
}

mount();
