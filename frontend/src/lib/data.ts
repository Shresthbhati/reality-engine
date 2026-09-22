import type { SessionRow, WorldRow, EvidenceRow, AnalysisRow, ResultRow, ReportRow, PlaceRow, WorldVersionRow } from "./types";

/**
 * No backend data source is wired into this rebuild yet.
 * These are the single real (empty) sources every screen reads from —
 * never duplicate a list literal in a page component.
 */
export const SESSIONS: SessionRow[] = [
  {
    id: "test-verify",
    name: "Verify Session",
    state: "COMPLETE",
    worldId: "test-world",
    worldName: "Verify World",
    location: "Chennai, Tamil Nadu",
    lat: 13.0827,
    lng: 80.2707,
    coverageKm2: 1.8,
    capturedAt: "21 Sep 2026 · 14:12",
    durationSec: 1122,
    evidenceCount: 1,
    stages: [{ label: "Session created", timestamp: "14:12", status: "COMPLETE" }],
  },
];
export const WORLDS: WorldRow[] = [
  {
    id: "test-world",
    name: "Verify World",
    location: "Chennai, Tamil Nadu",
    lat: 13.0827,
    lng: 80.2707,
    coverageKm2: 12.8,
    sessionCount: 1,
    evidenceCount: 1,
    timeRangeStart: "18 Sep 2026",
    timeRangeEnd: "21 Sep 2026",
    updatedAt: "21 Sep 2026 · 15:00",
  },
];
export const EVIDENCE: EvidenceRow[] = [
  {
    id: "test-evidence",
    name: "Exterior Wall",
    type: "IMAGE",
    location: "Chennai, Tamil Nadu",
    sessionId: "test-verify",
    sessionName: "Verify Session",
    worldId: "test-world",
    worldName: "Verify World",
    capturedAt: "21 Sep 2026 · 14:42",
    uploadedAt: "21 Sep 2026 · 14:43",
    processedAt: "21 Sep 2026 · 14:45",
    processingState: "PROCESSED",
  },
];
export const ANALYSIS: AnalysisRow[] = [
  {
    id: "test-analysis",
    name: "Structural inspection",
    type: "Structural inspection",
    state: "COMPLETE",
    currentStage: null,
    sessionId: "test-verify",
    sessionName: "Verify Session",
    startedAt: "21 Sep 2026 · 14:20",
    completedAt: "21 Sep 2026 · 14:38",
    owner: "Shresth Bhati",
  },
];
export const RESULTS: ResultRow[] = [
  {
    id: "test-result",
    title: "Structural Change",
    type: "Structural",
    finding: "Minor deflection detected in east wall panel.",
    sessionId: "test-verify",
    sessionName: "Verify Session",
    analysisId: "test-analysis",
    location: "Chennai, Tamil Nadu",
    lat: 13.0827,
    lng: 80.2707,
    generatedAt: "21 Sep 2026 · 15:02",
  },
];
export const REPORTS: ReportRow[] = [];
export const PLACES: PlaceRow[] = [
  { id: "test-place", name: "North Corridor", worldId: "test-world", lat: 13.084, lng: 80.271 },
];
export const WORLD_VERSIONS: WorldVersionRow[] = [
  {
    id: "test-version",
    worldId: "test-world",
    label: "Version 2",
    createdAt: "21 Sep 2026 · 15:00",
    changeSummary: "Added Verify Session",
    isCurrent: true,
  },
];

export function getSession(id: string): SessionRow | undefined {
  return SESSIONS.find((s) => s.id === id);
}

export function getWorld(id: string): WorldRow | undefined {
  return WORLDS.find((w) => w.id === id);
}

export function sessionsForWorld(worldId: string): SessionRow[] {
  return SESSIONS.filter((s) => s.worldId === worldId);
}

export function getEvidence(id: string): EvidenceRow | undefined {
  return EVIDENCE.find((e) => e.id === id);
}

export function evidenceForSession(sessionId: string): EvidenceRow[] {
  return EVIDENCE.filter((e) => e.sessionId === sessionId);
}

export function getAnalysis(id: string): AnalysisRow | undefined {
  return ANALYSIS.find((a) => a.id === id);
}

export function analysisForSession(sessionId: string): AnalysisRow[] {
  return ANALYSIS.filter((a) => a.sessionId === sessionId);
}

export function getResult(id: string): ResultRow | undefined {
  return RESULTS.find((r) => r.id === id);
}

export function resultsForAnalysis(analysisId: string): ResultRow[] {
  return RESULTS.filter((r) => r.analysisId === analysisId);
}

export function getReport(id: string): ReportRow | undefined {
  return REPORTS.find((r) => r.id === id);
}

export function placesForWorld(worldId: string): PlaceRow[] {
  return PLACES.filter((p) => p.worldId === worldId);
}

export function versionsForWorld(worldId: string): WorldVersionRow[] {
  return WORLD_VERSIONS.filter((v) => v.worldId === worldId);
}

export function resultsForWorld(worldId: string): ResultRow[] {
  const sessionIds = new Set(SESSIONS.filter((s) => s.worldId === worldId).map((s) => s.id));
  return RESULTS.filter((r) => r.sessionId != null && sessionIds.has(r.sessionId));
}

export function analysisForWorld(worldId: string): AnalysisRow[] {
  const sessionIds = new Set(SESSIONS.filter((s) => s.worldId === worldId).map((s) => s.id));
  return ANALYSIS.filter((a) => a.sessionId != null && sessionIds.has(a.sessionId));
}
