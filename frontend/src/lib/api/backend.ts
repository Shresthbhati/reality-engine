/**
 * Server-side helper for the Next.js proxy routes (`src/app/api/*`).
 * Adds `Authorization: Bearer <token>` when the API was started with
 * REALITY_API_TOKEN. Reads the non-public `REALITY_API_TOKEN` (never shipped to the
 * browser) and falls back to `NEXT_PUBLIC_API_TOKEN` so one local setting still works.
 */
export const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export function backendFetch(input: string | URL, init: RequestInit = {}): Promise<Response> {
  const token = (process.env.REALITY_API_TOKEN || process.env.NEXT_PUBLIC_API_TOKEN || "").trim();
  const headers = new Headers(init.headers);
  if (token && !headers.has("Authorization")) headers.set("Authorization", `Bearer ${token}`);
  return fetch(input, { ...init, headers });
}
