/**
 * Sites resource: user-directed composition of reconstructed worlds on a
 * top-down metric grid. Not geo-registration - the server stores exactly what
 * the user placed and reports overlaps it computed.
 */
import { apiDelete, apiGet, apiPatch, apiPost } from "./client";
import { useApi } from "./hooks";
import type { Placement } from "@/lib/sites/site-grid-model";

export type { Placement } from "@/lib/sites/site-grid-model";

export interface SiteSummary {
  id: string;
  name: string;
  grid_cell_m: number;
  placement_count: number;
}

export interface Site {
  id: string;
  name: string;
  grid_cell_m: number;
  created_at: string | null;
  placements: Placement[];
  overlaps: [string, string][];
}

export interface CreatePlacementInput {
  world_id: string;
  label?: string;
  cell_x: number;
  cell_z: number;
  offset_x_m?: number;
  offset_z_m?: number;
  rotation_deg?: number;
  scale?: number;
}

export type UpdatePlacementInput = Partial<
  Pick<Placement, "label" | "cell_x" | "cell_z" | "offset_x_m" | "offset_z_m" | "rotation_deg" | "scale">
>;

const enc = encodeURIComponent;

export function listSites(): Promise<SiteSummary[]> {
  return apiGet<SiteSummary[]>("/api/sites");
}

export function createSite(input: { name: string; grid_cell_m?: number }): Promise<Site> {
  return apiPost<Site>("/api/sites", input);
}

/** Throws `ApiError{code:"not_found"}` for an unknown id. */
export function getSite(id: string): Promise<Site> {
  return apiGet<Site>(`/api/sites/${enc(id)}`);
}

export function deleteSite(id: string): Promise<void> {
  return apiDelete<void>(`/api/sites/${enc(id)}`);
}

export function addPlacement(siteId: string, input: CreatePlacementInput): Promise<Placement> {
  return apiPost<Placement>(`/api/sites/${enc(siteId)}/placements`, input);
}

export function updatePlacement(siteId: string, placementId: string, patch: UpdatePlacementInput): Promise<Placement> {
  return apiPatch<Placement>(`/api/sites/${enc(siteId)}/placements/${enc(placementId)}`, patch);
}

export function removePlacement(siteId: string, placementId: string): Promise<void> {
  return apiDelete<void>(`/api/sites/${enc(siteId)}/placements/${enc(placementId)}`);
}

export function useSites() {
  return useApi(() => listSites());
}

export function useSite(id: string) {
  return useApi(() => getSite(id), [id]);
}
