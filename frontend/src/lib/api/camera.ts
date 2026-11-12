import { apiGet, apiPost } from "./client";

export type CameraKind = "focal_length_35mm" | "hfov_deg" | "focal_px";

/** Valid ranges, mirroring apps/api/worldstore_service.set_camera_declaration (the server re-validates). */
export const CAMERA_FIELDS: Record<CameraKind, { label: string; unit: string; min: number; max: number }> = {
  focal_length_35mm: { label: "Focal length (35mm equivalent)", unit: "mm", min: 8, max: 600 },
  hfov_deg: { label: "Horizontal field of view", unit: "°", min: 10, max: 150 },
  focal_px: { label: "Focal length in pixels", unit: "px", min: 50, max: 100000 },
};

const url = (worldId: string) => `/api/worlds/${encodeURIComponent(worldId)}/camera`;

export function getCameraDeclaration(worldId: string) {
  return apiGet<{ world_id: string; camera: Partial<Record<CameraKind, number>> | null }>(url(worldId));
}

/** Declare the camera (exactly one value) and re-evaluate the world with it. */
export function declareCamera(worldId: string, kind: CameraKind, value: number) {
  return apiPost<{ job_id: string; camera: Partial<Record<CameraKind, number>> }>(url(worldId), { [kind]: value });
}

/** null when valid, else a user-facing reason. */
export function validateCamera(kind: CameraKind, raw: string): string | null {
  const f = CAMERA_FIELDS[kind];
  const v = Number(raw);
  if (raw.trim() === "" || !Number.isFinite(v)) return "Enter a number.";
  if (v < f.min || v > f.max) return `Must be between ${f.min} and ${f.max} ${f.unit}.`;
  return null;
}
