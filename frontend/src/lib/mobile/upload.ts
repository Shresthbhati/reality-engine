"use client";

/**
 * One photo -> POST /api/reconstructions, with REAL upload progress.
 *
 * `fetch` cannot report upload progress, so this uses XMLHttpRequest against the
 * same endpoint and multipart contract as `createReconstruction`
 * (lib/api/reconstructions.ts). Failures are `ApiError`s from the shared client,
 * so the screens reuse its `retryable` / `describe()` vocabulary.
 */
import { ApiError, apiBaseUrl, type CreateReconstructionResult } from "@/lib/api";

export interface UploadOptions {
  worldId: string | null;
  latitude?: number;
  longitude?: number;
  onProgress: (loaded: number, total: number) => void;
  signal?: AbortSignal;
}

const UPLOAD_TIMEOUT_MS = 600_000;

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const b = body as { message?: unknown; detail?: unknown };
    if (typeof b.message === "string" && b.message.trim()) return b.message;
    if (typeof b.detail === "string" && b.detail.trim()) return b.detail;
    if (Array.isArray(b.detail) && b.detail[0] && typeof (b.detail[0] as { msg?: unknown }).msg === "string") {
      return (b.detail[0] as { msg: string }).msg;
    }
  }
  return fallback;
}

export function uploadPhoto(file: File, opts: UploadOptions): Promise<CreateReconstructionResult> {
  return new Promise((resolve, reject) => {
    const url = `${apiBaseUrl()}/api/reconstructions`;
    const form = new FormData();
    form.append("files", file, file.name);
    if (opts.worldId) form.append("world_id", opts.worldId);
    if (opts.latitude !== undefined && opts.longitude !== undefined) {
      form.append("latitude", String(opts.latitude));
      form.append("longitude", String(opts.longitude));
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.timeout = UPLOAD_TIMEOUT_MS;

    const fail = (init: { message: string; status: number; retryable: boolean; code: ConstructorParameters<typeof ApiError>[0]["code"]; details?: unknown }) =>
      reject(new ApiError(init));

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) opts.onProgress(e.loaded, e.total);
    };
    xhr.onerror = () =>
      fail({
        message: `Could not reach the Reality Engine API at ${apiBaseUrl()}. Check the connection and try again.`,
        status: 0,
        code: "network_unreachable",
        retryable: true,
        details: { url },
      });
    xhr.ontimeout = () =>
      fail({ message: "The upload timed out. Check the connection and try again.", status: 0, code: "timeout", retryable: true });
    xhr.onabort = () =>
      fail({ message: "The upload was cancelled.", status: 0, code: "unknown_error", retryable: true });
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300 && body) {
        opts.onProgress(1, 1);
        resolve(body as CreateReconstructionResult);
        return;
      }
      const retryable = xhr.status === 0 || xhr.status === 408 || xhr.status === 429 || xhr.status >= 500;
      fail({
        message: errorMessage(body, `The API answered HTTP ${xhr.status}.`),
        status: xhr.status,
        code: xhr.status === 422 ? "validation_failed" : xhr.status >= 500 ? "server_error" : "invalid_request",
        retryable,
        details: body,
      });
    };

    opts.signal?.addEventListener("abort", () => xhr.abort(), { once: true });
    xhr.send(form);
  });
}
