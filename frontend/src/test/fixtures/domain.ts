/**
 * Test/dev fixtures only.
 *
 * These rows were previously `frontend/src/lib/data.ts`, imported by every
 * production screen. They are kept here so component tests and local
 * Storybook-style demos keep working; NO production route may import this
 * module. Production reads real entities through `@/lib/api`.
 */
import type {
  AnalysisRow,
  EvidenceRow,
  PlaceRow,
  ReportRow,
  ResultRow,
  SessionRow,
  WorldRow,
  WorldVersionRow,
} from "@/lib/types";

export const FIXTURE_SESSIONS: SessionRow[] = [
  {
    id: "test-verify",
    name: "Verify Session",
    state: "COMPLETE",
    statusRaw: "complete",
    worldId: "test-world",
    worldName: "Verify World",
    location: "13.0827°N 80.2707°E",
    lat: 13.0827,
    lng: 80.2707,
    coverageKm2: 1.8,
    capturedAt: "21 Sep 2026 · 14:12",
    durationSec: 1122,
    evidenceCount: 1,
    stages: [{ label: "Session created", timestamp: "14:12", status: "COMPLETE" }],
  },
];

export const FIXTURE_WORLDS: WorldRow[] = [
  {
    id: "test-world",
    name: "Verify World",
    location: "13.0827°N 80.2707°E",
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

export const FIXTURE_EVIDENCE: EvidenceRow[] = [
  {
    id: "test-evidence",
    name: "Exterior Wall",
    type: "IMAGE",
    location: "13.0827°N 80.2707°E",
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

export const FIXTURE_ANALYSIS: AnalysisRow[] = [
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

export const FIXTURE_RESULTS: ResultRow[] = [
  {
    id: "test-result",
    title: "Structural Change",
    type: "Structural",
    finding: "Minor deflection detected in east wall panel.",
    sessionId: "test-verify",
    sessionName: "Verify Session",
    analysisId: "test-analysis",
    location: "13.0827°N 80.2707°E",
    lat: 13.0827,
    lng: 80.2707,
    generatedAt: "21 Sep 2026 · 15:02",
  },
];

export const FIXTURE_REPORTS: ReportRow[] = [];
export const FIXTURE_PLACES: PlaceRow[] = [
  { id: "test-place", name: "North Corridor", worldId: "test-world", lat: 13.084, lng: 80.271 },
];
export const FIXTURE_WORLD_VERSIONS: WorldVersionRow[] = [
  {
    id: "test-version",
    worldId: "test-world",
    label: "Version 2",
    createdAt: "21 Sep 2026 · 15:00",
    changeSummary: "Added Verify Session",
    isCurrent: true,
  },
];
