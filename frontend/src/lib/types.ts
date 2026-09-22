// Shared by Session and Analysis — both are backend processing pipelines
// with the same real vocabulary of states.
export type ProcessingState = "QUEUED" | "PROCESSING" | "COMPLETE" | "FAILED";

export interface SessionStage {
  label: string;
  timestamp: string;
  status: "COMPLETE" | "ACTIVE" | "FAILED";
}

export interface SessionRow {
  id: string;
  name: string;
  state: ProcessingState;
  /** Exact server status string, kept so no screen has to guess the nuance. */
  statusRaw?: string;
  worldId: string | null;
  worldName: string | null;
  location: string | null;
  lat: number | null;
  lng: number | null;
  locationAccuracyM?: number | null;
  locationSource?: string | null;
  coverageKm2: number | null;
  capturedAt: string | null;
  durationSec: number | null;
  evidenceCount: number;
  stages: SessionStage[];
}

export interface WorldRow {
  id: string;
  name: string;
  description?: string | null;
  status?: string;
  location: string | null;
  lat: number | null;
  lng: number | null;
  coverageKm2: number | null;
  sessionCount: number;
  evidenceCount: number;
  timeRangeStart: string | null;
  timeRangeEnd: string | null;
  currentVersionId?: string | null;
  updatedAt: string | null;
}

export type EvidenceType = "IMAGE" | "VIDEO" | "POINT_CLOUD" | "DOCUMENT" | "SENSOR_DATA";
export type EvidenceProcessingState = "UPLOADING" | "PROCESSING" | "PROCESSED" | "FAILED";

export interface EvidenceRow {
  id: string;
  name: string;
  type: EvidenceType;
  /** Exact server type/state strings; the UI unions are a lossy view of them. */
  typeRaw?: string;
  location: string | null;
  sessionId: string | null;
  sessionName: string | null;
  worldId: string | null;
  worldName: string | null;
  capturedAt: string | null;
  uploadedAt: string | null;
  processedAt: string | null;
  processingState: EvidenceProcessingState;
  processingStateRaw?: string;
  mimeType?: string | null;
  sizeBytes?: number | null;
  checksum?: string | null;
}

export interface AnalysisRow {
  id: string;
  name: string;
  type: string;
  state: ProcessingState;
  currentStage: string | null;
  sessionId: string | null;
  sessionName: string | null;
  startedAt: string | null;
  completedAt: string | null;
  owner: string | null;
}

export interface ResultRow {
  id: string;
  title: string;
  type: string;
  finding: string | null;
  sessionId: string | null;
  sessionName: string | null;
  analysisId: string | null;
  location: string | null;
  lat: number | null;
  lng: number | null;
  generatedAt: string | null;
}

export type ReportStatus = "DRAFT" | "COMPLETED";

export interface ReportRow {
  id: string;
  title: string;
  status: ReportStatus;
  resultIds: string[];
  sessionCount: number;
  generatedAt: string | null;
  updatedAt: string | null;
  summary: string | null;
}

export interface PlaceRow {
  id: string;
  name: string;
  worldId: string;
  lat: number;
  lng: number;
}

export interface WorldVersionRow {
  id: string;
  worldId: string;
  label: string;
  createdAt: string;
  changeSummary: string | null;
  isCurrent: boolean;
}
