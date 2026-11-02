import { ApiError, apiBaseUrl } from "./client";

/** Fetch `url` (absolute, or a path on the API host) and hand the bytes to the
 * browser as a file download. The one download path for every export: a failed
 * request throws an `ApiError` carrying the HTTP status and the server's own
 * `detail`, so callers can show a visible reason instead of a silent no-op. */
export async function downloadArtifact(
  url: string,
  filename: string,
  { timeoutMs = 60_000 }: { timeoutMs?: number } = {},
): Promise<void> {
  const target = /^https?:\/\//.test(url) ? url : `${apiBaseUrl()}${url.startsWith("/") ? url : `/${url}`}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res: Response;
  try {
    res = await fetch(target, { signal: controller.signal, cache: "no-store" });
  } catch (err) {
    const aborted = (err as Error).name === "AbortError";
    throw new ApiError({
      message: aborted
        ? `Download did not finish within ${Math.round(timeoutMs / 1000)}s.`
        : `Could not reach the API at ${apiBaseUrl()} to download ${filename}.`,
      status: 0,
      code: aborted ? "timeout" : "network_unreachable",
      retryable: true,
    });
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as { detail?: unknown; message?: string } | null;
    const reason =
      typeof body?.detail === "string" ? body.detail : typeof body?.message === "string" ? body.message : "";
    throw new ApiError({
      message: `Export download failed (HTTP ${res.status})${reason ? `: ${reason}` : ""}`,
      status: res.status,
      code: res.status === 404 ? "not_found" : res.status === 409 ? "conflict" : "server_error",
      retryable: res.status >= 500,
    });
  }

  const blob = await res.blob();
  if (blob.size === 0) {
    throw new ApiError({
      message: `The server returned an empty file for ${filename}.`,
      status: res.status,
      code: "unknown_error",
      retryable: false,
    });
  }
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}
