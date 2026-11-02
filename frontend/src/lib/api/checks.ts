/**
 * Pre-flight check of a photo set (POST /api/uploads/check) and job progress (GET /api/jobs/{id}).
 * Both mirror apps/api exactly; nothing is stored by a check.
 */
import { apiGet, apiPostFiles } from "./client";

export interface CheckedFile {
  name: string;
  size: number;
  kind: string;
  sha256: string;
  duplicate_of: string | null;
  decodable: boolean | null;
  width: number | null;
  height: number | null;
  sharpness: number | null;
  relatively_soft: boolean;
  has_gps: boolean;
  looks_like_photo: boolean | null;
  too_small: boolean;
  problem: string | null;
}

export type CheckVerdict = "good" | "has_problems" | "add_angles" | "no_usable_photos";

export interface CheckResult {
  files: CheckedFile[];
  summary: { total: number; usable: number; flagged: number; soft_threshold: number | null; verdict: CheckVerdict };
}

export function checkUploads(files: File[]): Promise<CheckResult> {
  return apiPostFiles<CheckResult>("/api/uploads/check", "files", files, {}, { timeoutMs: 120_000 });
}

export interface JobView {
  id: string;
  status: string;
  stage: string | null;
  progress: number | null;
  photos_total: number | null;
  photos_placed: number | null;
  terminal: boolean;
  error: string | null;
}

export function getJobProgress(jobId: string): Promise<JobView> {
  return apiGet<JobView>(`/api/jobs/${encodeURIComponent(jobId)}`);
}
