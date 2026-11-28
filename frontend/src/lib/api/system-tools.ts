/** What this machine can run for photo/video reconstruction. Mirrors apps/api/routes_system.py (`/api/system/tools`). */
import { apiGet, apiPut } from "./client";

export interface ColmapToolStatus {
  found: boolean;
  path: string | null;
  /** explicit | env:<NAME> | settings | PATH | common_dir | not_found */
  source: string;
  /** every place that was looked at, in order -- shown when nothing was found */
  tried: string[];
  warnings: string[];
  saved_path: string | null;
  env_vars: string[];
  settings_file: string;
  message: string;
}

export interface SystemTools {
  colmap: ColmapToolStatus;
  video: { available: boolean; backend: string | null; detail: string };
}

export function getSystemTools(): Promise<SystemTools> {
  return apiGet<SystemTools>("/api/system/tools");
}

/** Save (or, with an empty path, clear) the COLMAP location. The API refuses a path that does not exist. */
export function setColmapPath(path: string | null): Promise<SystemTools> {
  return apiPut<SystemTools>("/api/system/tools/colmap", { path });
}
