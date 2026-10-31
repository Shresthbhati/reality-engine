import { apiDelete, apiGet, apiPost } from "./client";
import type { DimensionState, MeasureKind, ScaleBasis, Vec3 } from "@/lib/viewport/measure-model";

export interface SavedMeasurement {
  id: string;
  world_id: string;
  kind: MeasureKind;
  points: Vec3[];
  value: number;
  unit: string;
  uncertainty: number | null;
  method: string;
  scale_state: ScaleBasis;
  created_at: string | null;
  /** D1: state of the dimension; absent only from a pre-D1 server. */
  state?: DimensionState;
  basis?: string;
  uncertainty_m?: number | null;
  verified_value?: number | null;
  dimension?: { value: number; unit: string; state: DimensionState; basis: string; uncertainty_m: number | null };
}

export interface MeasurementIn {
  kind: MeasureKind;
  points: Vec3[];
  scale_state: ScaleBasis;
  method: string;
  value?: number;
  uncertainty?: number | null;
  /** Tape measurement of the same length; marks the dimension VERIFIED (metric/authored worlds only). */
  verified_value?: number;
}

const base = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}/measurements`;

export function listMeasurements(worldId: string): Promise<SavedMeasurement[]> {
  return apiGet<SavedMeasurement[]>(base(worldId));
}

export function createMeasurement(worldId: string, body: MeasurementIn): Promise<SavedMeasurement> {
  return apiPost<SavedMeasurement>(base(worldId), body);
}

export function deleteMeasurement(worldId: string, id: string): Promise<{ id: string; deleted: boolean }> {
  return apiDelete(`${base(worldId)}/${encodeURIComponent(id)}`);
}
