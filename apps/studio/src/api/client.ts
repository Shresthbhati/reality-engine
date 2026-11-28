import type {
  CameraStation,
  EntityConfidenceResponse,
  EntityProvenanceResponse,
  ExportFormatInfo,
  PointCloudResponse,
  WorldDiffResponse,
  WorldIRPayload,
  WorldQualityResponse,
  WorldSummary,
  WorldVersionInfo,
} from '../types/studio';

export class StudioApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** HTTP client bound to real Reality Engine FastAPI WorldIR / WorldStore endpoints. */
export class RealityStudioClient {
  private readonly baseUrl: string;
  private readonly getToken: () => string | null;

  constructor(baseUrl = '/api', getToken: () => string | null = () => null) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.getToken = getToken;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    if (!headers.has('Content-Type') && init.body && !(init.body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
    }
    const token = this.getToken();
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    const resp = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!resp.ok) {
      const detail = await resp.text().catch(() => resp.statusText);
      throw new StudioApiError(resp.status, detail || `HTTP ${resp.status}`);
    }
    return (await resp.json()) as T;
  }

  signup(
    email: string,
    password: string,
    tenantName: string,
    fullName = 'Studio Operator',
  ): Promise<{ token: string; user: { email: string } }> {
    return this.request('/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ email, password, tenant_name: tenantName, full_name: fullName }),
    });
  }

  login(email: string, password: string): Promise<{ token: string; user: { email: string } }> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  createSession(name: string): Promise<{ id: string; name?: string }> {
    return this.request('/sessions', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  }

  uploadImage(sessionId: string, file: File): Promise<{ id: string; filename?: string }> {
    const form = new FormData();
    form.append('file', file);
    return this.request(`/uploads?session_id=${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      body: form,
    });
  }

  attachSession(worldId: string, sessionId: string): Promise<Record<string, unknown>> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/attach/${encodeURIComponent(sessionId)}`, {
      method: 'POST',
    });
  }

  async listWorlds(): Promise<WorldSummary[]> {
    const res = await this.request<WorldSummary[] | { items: WorldSummary[] }>('/worlds');
    return Array.isArray(res) ? res : (res.items ?? []);
  }

  createWorld(name: string): Promise<WorldSummary> {
    return this.request<WorldSummary>('/worlds', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  }

  triggerReconstruction(sessionId: string): Promise<{ job_id: string; status: string }> {
    return this.request(`/sessions/${encodeURIComponent(sessionId)}/reconstruct`, {
      method: 'POST',
    });
  }

  getReconstructionStatus(worldId: string): Promise<{ status: string; progress?: number; detail?: string }> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/status`);
  }

  getWorldIR(worldId: string, versionId?: string): Promise<WorldIRPayload> {
    const q = versionId ? `?version=${encodeURIComponent(versionId)}` : '';
    return this.request<WorldIRPayload>(`/worlds/${encodeURIComponent(worldId)}/worldir${q}`);
  }

  async getVersions(worldId: string): Promise<WorldVersionInfo[]> {
    const res = await this.request<WorldVersionInfo[] | { items?: WorldVersionInfo[]; versions?: WorldVersionInfo[] }>(
      `/worlds/${encodeURIComponent(worldId)}/versions`,
    );
    return Array.isArray(res) ? res : (res.items ?? res.versions ?? []);
  }

  getPoints(worldId: string): Promise<PointCloudResponse> {
    return this.request<PointCloudResponse>(`/worlds/${encodeURIComponent(worldId)}/points`);
  }

  getCameras(worldId: string): Promise<{ cameras: CameraStation[] }> {
    return this.request<{ cameras: CameraStation[] }>(`/worlds/${encodeURIComponent(worldId)}/cameras`);
  }

  getQuality(worldId: string): Promise<WorldQualityResponse> {
    return this.request<WorldQualityResponse>(`/worlds/${encodeURIComponent(worldId)}/quality`);
  }

  getCoverage(worldId: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(`/worlds/${encodeURIComponent(worldId)}/coverage`);
  }

  getEntityProvenance(worldId: string, entityId: string, versionId?: string): Promise<EntityProvenanceResponse> {
    const q = versionId ? `?version=${encodeURIComponent(versionId)}` : '';
    return this.request<EntityProvenanceResponse>(
      `/worlds/${encodeURIComponent(worldId)}/entities/${encodeURIComponent(entityId)}/provenance${q}`,
    );
  }

  getEntityConfidence(worldId: string, entityId: string): Promise<EntityConfidenceResponse> {
    return this.request<EntityConfidenceResponse>(
      `/worlds/${encodeURIComponent(worldId)}/entities/${encodeURIComponent(entityId)}/confidence`,
    );
  }

  reclassifyEntity(
    worldId: string,
    entityId: string,
    newClass: string,
    note = '',
    parentVersionId?: string,
  ): Promise<Record<string, unknown>> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/corrections/reclassify`, {
      method: 'POST',
      body: JSON.stringify({
        entity_id: entityId,
        new_class: newClass,
        note,
        parent_version_id: parentVersionId,
      }),
    });
  }

  splitEntity(
    worldId: string,
    entityId: string,
    splitAxis: 'x' | 'y' | 'z',
    splitRatio = 0.5,
    note = '',
  ): Promise<Record<string, unknown>> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/corrections/split`, {
      method: 'POST',
      body: JSON.stringify({
        entity_id: entityId,
        split_axis: splitAxis,
        split_ratio: splitRatio,
        note,
      }),
    });
  }

  mergeEntities(
    worldId: string,
    primaryEntityId: string,
    secondaryEntityIds: string[],
    newName?: string,
    note = '',
  ): Promise<Record<string, unknown>> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/corrections/merge`, {
      method: 'POST',
      body: JSON.stringify({
        primary_entity_id: primaryEntityId,
        secondary_entity_ids: secondaryEntityIds,
        new_name: newName,
        note,
      }),
    });
  }

  rollbackVersion(worldId: string, targetVersionId: string): Promise<Record<string, unknown>> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/rollback`, {
      method: 'POST',
      body: JSON.stringify({ target_version_id: targetVersionId }),
    });
  }

  getDiff(worldId: string, baseVersionId?: string, headVersionId?: string): Promise<WorldDiffResponse> {
    const params = new URLSearchParams();
    if (baseVersionId) params.set('base', baseVersionId);
    if (headVersionId) params.set('head', headVersionId);
    const qs = params.toString() ? `?${params.toString()}` : '';
    return this.request<WorldDiffResponse>(`/worlds/${encodeURIComponent(worldId)}/diff${qs}`);
  }

  getExportFormats(worldId: string): Promise<ExportFormatInfo[]> {
    return this.request<ExportFormatInfo[]>(`/worlds/${encodeURIComponent(worldId)}/export/formats`);
  }

  exportWorld(worldId: string, format: string): Promise<{ content_hash: string; download_url?: string }> {
    return this.request(`/worlds/${encodeURIComponent(worldId)}/export`, {
      method: 'POST',
      body: JSON.stringify({ format }),
    });
  }
}
