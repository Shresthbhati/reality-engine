/**
 * Server DTO → screen row adapters.
 *
 * Every value a screen shows is derived from a server field or is explicitly
 * absent. Nothing is estimated: a session with no GPS reports
 * `location: null`, a session with no recorded duration reports `null`, and
 * coverage stays `null` until the backend actually computes it.
 */
import type {
  EvidenceProcessingState,
  EvidenceRow,
  EvidenceType,
  ProcessingState,
  SessionRow,
  SessionStage,
  WorldRow,
} from "@/lib/types";
import type { ActivityDto, EvidenceDto, LocationDto, SessionDto, WorldDto } from "./types";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Server timestamps are the truth; this only formats them for display. */
export function formatTimestamp(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  const day = String(d.getDate()).padStart(2, "0");
  const month = MONTHS[d.getMonth()];
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${day} ${month} ${d.getFullYear()} · ${hh}:${mm}`;
}

export function formatDate(iso: string | null | undefined): string | null {
  const full = formatTimestamp(iso);
  return full ? full.split(" · ")[0] : null;
}

/** Coordinates, because no geocoder is wired: never a guessed place name. */
export function formatCoordinates(loc: LocationDto | null | undefined): string | null {
  if (!loc) return null;
  const lat = `${Math.abs(loc.latitude).toFixed(4)}°${loc.latitude >= 0 ? "N" : "S"}`;
  const lon = `${Math.abs(loc.longitude).toFixed(4)}°${loc.longitude >= 0 ? "E" : "W"}`;
  return `${lat} ${lon}`;
}

export function formatAccuracy(loc: LocationDto | null | undefined): string | null {
  if (!loc || loc.accuracy === null || loc.accuracy === undefined) return null;
  return `±${Math.round(loc.accuracy)} m (${loc.source})`;
}

/** Server session status vocabulary → the pipeline states the UI renders. */
export function sessionStateFrom(status: string): ProcessingState {
  switch (status) {
    case "processing":
      return "PROCESSING";
    case "complete":
    case "completed":
      return "COMPLETE";
    case "failed":
      return "FAILED";
    case "created":
    case "uploaded":
    case "queued":
    default:
      return "QUEUED";
  }
}

const EVIDENCE_TYPE: Record<string, EvidenceType> = {
  photo: "IMAGE",
  image: "IMAGE",
  video: "VIDEO",
  point_cloud: "POINT_CLOUD",
  dataset: "POINT_CLOUD",
  gnss: "SENSOR_DATA",
  sensor_log: "SENSOR_DATA",
  document: "DOCUMENT",
};

export function evidenceTypeFrom(type: string): EvidenceType {
  return EVIDENCE_TYPE[type] ?? "DOCUMENT";
}

export function evidenceProcessingFrom(state: string): EvidenceProcessingState {
  switch (state) {
    case "processing":
      return "PROCESSING";
    case "processed":
      return "PROCESSED";
    case "failed":
      return "FAILED";
    default:
      return "UPLOADING";
  }
}

/** Real stages only: derived from the timestamps the session actually has. */
export function sessionStages(dto: SessionDto): SessionStage[] {
  const stages: SessionStage[] = [];
  const hhmm = (iso: string | null) => {
    if (!iso) return null;
    const d = new Date(iso);
    return Number.isNaN(d.getTime())
      ? null
      : `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  };

  const created = hhmm(dto.created_at);
  if (created) stages.push({ label: "Session created", timestamp: created, status: "COMPLETE" });

  const uploaded = hhmm(dto.uploaded_at);
  if (uploaded) stages.push({ label: "Evidence uploaded", timestamp: uploaded, status: "COMPLETE" });

  const started = hhmm(dto.processing_started_at);
  if (started) {
    const done = hhmm(dto.processing_completed_at);
    stages.push({
      label: "Processing",
      timestamp: started,
      status: dto.status === "failed" ? "FAILED" : done ? "COMPLETE" : "ACTIVE",
    });
  }

  const completed = hhmm(dto.processing_completed_at);
  if (completed) {
    stages.push({ label: "Processing complete", timestamp: completed, status: "COMPLETE" });
  }
  return stages;
}

export function toSessionRow(dto: SessionDto, worldName: string | null = null): SessionRow {
  const durationSec =
    dto.captured_at && dto.processing_completed_at
      ? Math.max(
          0,
          Math.round(
            (new Date(dto.processing_completed_at).getTime() - new Date(dto.captured_at).getTime()) /
              1000,
          ),
        )
      : null;

  return {
    id: dto.id,
    name: dto.name,
    state: sessionStateFrom(dto.status),
    statusRaw: dto.status,
    worldId: dto.world_id,
    worldName: dto.world_id ? worldName : null,
    location: formatCoordinates(dto.location),
    lat: dto.location?.latitude ?? null,
    lng: dto.location?.longitude ?? null,
    locationAccuracyM: dto.location?.accuracy ?? null,
    locationSource: dto.location?.source ?? null,
    coverageKm2: null,
    capturedAt: formatTimestamp(dto.captured_at ?? dto.created_at),
    durationSec,
    evidenceCount: dto.evidence_count,
    stages: sessionStages(dto),
  };
}

export function toWorldRow(dto: WorldDto, evidenceCount = 0): WorldRow {
  const loc: LocationDto | null =
    dto.latitude !== null && dto.longitude !== null
      ? {
          latitude: dto.latitude,
          longitude: dto.longitude,
          altitude: null,
          accuracy: null,
          heading: null,
          speed: null,
          source: "world_origin",
          captured_at: null,
        }
      : null;

  return {
    id: dto.id,
    name: dto.name,
    description: dto.description,
    status: dto.status,
    location: formatCoordinates(loc),
    lat: dto.latitude,
    lng: dto.longitude,
    coverageKm2: null,
    sessionCount: dto.session_count,
    evidenceCount,
    timeRangeStart: null,
    timeRangeEnd: null,
    currentVersionId: dto.current_version_id,
    updatedAt: formatTimestamp(dto.created_at),
  };
}

export function toEvidenceRow(
  dto: EvidenceDto,
  sessionName: string | null = null,
  worldId: string | null = null,
  worldName: string | null = null,
): EvidenceRow {
  const captured = dto.metadata?.captured_at;
  return {
    id: dto.id,
    name: dto.name,
    type: evidenceTypeFrom(dto.type),
    typeRaw: dto.type,
    location: formatCoordinates(dto.location ?? null),
    sessionId: dto.session_id,
    sessionName,
    worldId,
    worldName,
    capturedAt: typeof captured === "string" ? formatTimestamp(captured) : null,
    uploadedAt: formatTimestamp(dto.created_at),
    processedAt: formatTimestamp(dto.processed_at),
    processingState: evidenceProcessingFrom(dto.processing_state),
    mimeType: dto.mime_type,
    sizeBytes: dto.size,
    checksum: dto.checksum,
    processingStateRaw: dto.processing_state,
  };
}

export interface ActivityRow {
  id: string;
  type: string;
  summary: string;
  entityType: string | null;
  entityId: string | null;
  at: string | null;
}

export function toActivityRow(dto: ActivityDto): ActivityRow {
  return {
    id: dto.id,
    type: dto.type,
    summary: dto.summary,
    entityType: dto.entity_type,
    entityId: dto.entity_id,
    at: formatTimestamp(dto.created_at),
  };
}

/**
 * Deep link for an activity/notification entity, only for entity types the
 * product actually has a route for. Unknown types return null — no dead links.
 */
export function entityHref(entityType: string | null, entityId: string | null): string | null {
  if (!entityType || !entityId) return null;
  switch (entityType) {
    case "session":
      return `/sessions/${entityId}`;
    case "world":
      return `/worlds/${entityId}`;
    case "evidence":
      return `/evidence/${entityId}`;
    default:
      return null;
  }
}

