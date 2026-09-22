/**
 * Reality Engine — canonical API client (transport layer).
 *
 * One place decides how the browser reaches the application API
 * (`apps/api`, FastAPI) and how failures are represented:
 *
 *     ROUTE → API FUNCTION (lib/api/*) → this client → apps/api → engine
 *
 * Rules honoured here:
 * - the base URL comes from `NEXT_PUBLIC_API_BASE_URL` (default
 *   http://localhost:8100); no component hard-codes a host;
 * - every failure is an `ApiError` carrying the standard contract
 *   (code, message, status, retryable, request_id, details) so pages can show
 *   loading / empty / error / retry states without inventing reasons;
 * - a failed request NEVER returns fabricated data — callers get a throw.
 */

export const DEFAULT_API_BASE_URL = "http://localhost:8100";

export function apiBaseUrl(): string {
  const raw = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const base = raw && raw.length > 0 ? raw : DEFAULT_API_BASE_URL;
  return base.replace(/\/+$/, "");
}

export type ApiErrorCode =
  | "network_unreachable"
  | "timeout"
  | "not_found"
  | "conflict"
  | "invalid_request"
  | "validation_failed"
  | "permission_denied"
  | "unavailable"
  | "server_error"
  | "unknown_error";

export class ApiError extends Error {
  readonly status: number;
  readonly code: ApiErrorCode;
  readonly retryable: boolean;
  readonly requestId: string | null;
  readonly details: unknown;

  constructor(init: {
    message: string;
    status: number;
    code: ApiErrorCode;
    retryable: boolean;
    requestId?: string | null;
    details?: unknown;
  }) {
    super(init.message);
    this.name = "ApiError";
    this.status = init.status;
    this.code = init.code;
    this.retryable = init.retryable;
    this.requestId = init.requestId ?? null;
    this.details = init.details;
  }

  /** Operator-facing one-liner: what happened, and whether it is worth retrying. */
  describe(): string {
    const suffix = this.retryable ? " This is retryable." : "";
    return `${this.message}${suffix}`;
  }
}

export function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}

function codeForStatus(status: number): ApiErrorCode {
  switch (status) {
    case 400:
      return "invalid_request";
    case 401:
    case 403:
      return "permission_denied";
    case 404:
      return "not_found";
    case 409:
      return "conflict";
    case 422:
      return "validation_failed";
    case 502:
    case 503:
    case 504:
      return "unavailable";
    default:
      return status >= 500 ? "server_error" : "unknown_error";
  }
}

function retryableForStatus(status: number): boolean {
  return status === 0 || status === 408 || status === 429 || status >= 500;
}

interface ErrorBody {
  detail?: unknown;
  message?: string;
  code?: string;
  entity?: string;
  request_id?: string;
  retryable?: boolean;
  details?: unknown;
}

/** FastAPI reports string/list/object `detail`; normalise without losing it. */
function messageFromBody(body: ErrorBody | null, fallback: string): string {
  if (!body) return fallback;
  if (typeof body.message === "string" && body.message.trim()) return body.message;
  const detail = body.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { msg?: string; loc?: unknown[] } | undefined;
    if (first?.msg) {
      const where = Array.isArray(first.loc) ? ` (${first.loc.join(".")})` : "";
      return `${first.msg}${where}`;
    }
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

export interface RequestOptions {
  /** Milliseconds before the request is abandoned. */
  timeoutMs?: number;
  signal?: AbortSignal;
}

const DEFAULT_TIMEOUT_MS = 15_000;

async function perform<T>(path: string, init: RequestInit, options: RequestOptions): Promise<T> {
  const url = `${apiBaseUrl()}${path.startsWith("/") ? path : `/${path}`}`;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const onExternalAbort = () => controller.abort();
  options.signal?.addEventListener("abort", onExternalAbort);

  let res: Response;
  try {
    res = await fetch(url, { ...init, signal: controller.signal, cache: "no-store" });
  } catch (err) {
    const aborted = (err as Error).name === "AbortError";
    throw new ApiError({
      message: aborted
        ? `The Reality Engine API did not respond within ${Math.round(timeoutMs / 1000)}s (${url}).`
        : `Reality Engine API unreachable at ${apiBaseUrl()}. Start it with: uvicorn apps.api.main:app --port 8100`,
      status: 0,
      code: aborted ? "timeout" : "network_unreachable",
      retryable: true,
      details: { url },
    });
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", onExternalAbort);
  }

  const requestId = res.headers.get("x-request-id");

  if (!res.ok) {
    let body: ErrorBody | null = null;
    try {
      body = (await res.json()) as ErrorBody;
    } catch {
      body = null;
    }
    throw new ApiError({
      message: messageFromBody(body, `Request failed with HTTP ${res.status} (${path})`),
      status: res.status,
      code: (body?.code as ApiErrorCode | undefined) ?? codeForStatus(res.status),
      retryable: body?.retryable ?? retryableForStatus(res.status),
      requestId: requestId ?? body?.request_id ?? null,
      details: body?.details ?? body?.detail ?? null,
    });
  }

  if (res.status === 204) return undefined as T;
  const text = await res.text();
  if (!text) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError({
      message: `API returned a non-JSON body for ${path}`,
      status: res.status,
      code: "unknown_error",
      retryable: false,
      requestId,
    });
  }
}

export function apiGet<T>(path: string, options: RequestOptions = {}): Promise<T> {
  return perform<T>(path, { method: "GET", headers: { Accept: "application/json" } }, options);
}

export function apiPost<T>(path: string, body?: unknown, options: RequestOptions = {}): Promise<T> {
  return perform<T>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body ?? {}),
    },
    options,
  );
}

export function apiPatch<T>(path: string, body?: unknown, options: RequestOptions = {}): Promise<T> {
  return perform<T>(
    path,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body ?? {}),
    },
    options,
  );
}

export function apiDelete<T>(path: string, options: RequestOptions = {}): Promise<T> {
  return perform<T>(path, { method: "DELETE", headers: { Accept: "application/json" } }, options);
}

/** Multipart upload (evidence). Accepts a Blob/File plus extra form fields. */
export function apiPostForm<T>(
  path: string,
  fields: Record<string, string | Blob>,
  options: RequestOptions = {},
): Promise<T> {
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  return perform<T>(
    path,
    { method: "POST", body: form, headers: { Accept: "application/json" } },
    { timeoutMs: 120_000, ...options },
  );
}

/** Query-string helper that drops null/undefined/empty values. */
export function withQuery(
  path: string,
  params: Record<string, string | number | null | undefined>,
): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const qs = search.toString();
  return qs ? `${path}?${qs}` : path;
}

