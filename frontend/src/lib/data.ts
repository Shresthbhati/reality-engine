import type { SessionRow, WorldRow, EvidenceRow, AnalysisRow, ResultRow, ReportRow, PlaceRow, WorldVersionRow } from "./types";

/**
 * No backend data source is wired into this rebuild yet.
 * These are the single real (empty) sources every screen reads from —
 * never duplicate a list literal in a page component.
 */
export const SESSIONS: SessionRow[] = [];
export const WORLDS: WorldRow[] = [];
export const EVIDENCE: EvidenceRow[] = [];
export const ANALYSIS: AnalysisRow[] = [];
export const RESULTS: ResultRow[] = [];
export const REPORTS: ReportRow[] = [];
export const PLACES: PlaceRow[] = [];
export const WORLD_VERSIONS: WorldVersionRow[] = [];

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
