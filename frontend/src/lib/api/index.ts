/**
 * Canonical Reality Engine API client.
 *
 * Every production screen imports from here — never `fetch()` directly, never
 * a static row array. `client.ts` owns the transport and the error contract;
 * the resource modules own the endpoints; `adapters.ts` converts server DTOs
 * into the row types the UI renders.
 */
export * from "./client";
export * from "./types";
export * from "./adapters";
export * from "./sessions";
export * from "./worlds";
export * from "./evidence";
export * from "./analysis";
export * from "./platform";
export * from "./unsupported";
export * from "./worldir";
export * from "./hooks";

/**
 * The row shapes the API layer emits are the UI's view model; re-exported here
 * so screens import one module instead of two.
 */
export type {
  AnalysisRow,
  EvidenceProcessingState,
  EvidenceRow,
  EvidenceType,
  PlaceRow,
  ProcessingState,
  ReportRow,
  ResultRow,
  SessionRow,
  SessionStage,
  WorldRow,
  WorldVersionRow,
} from "@/lib/types";