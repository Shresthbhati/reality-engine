import { apiDelete, apiGet, apiPost } from "./client";
import type { MeasureKind, ScaleBasis, Vec3 } from "@/lib/viewport/measure-model";

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
}

export interface MeasurementIn {
  kind: MeasureKind;
  points: Vec3[];
  scale_state: ScaleBasis;
  method: string;
  value?: number;
  uncertainty?: number | null;
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
