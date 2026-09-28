import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract
 * `GET /api/worlds/{world_id}/diff` (apps/api/routes_worlds.py), which
 * computes a real structural diff between two WorldStore versions and 404s
 * with a reason when either version is absent.
 *
 * This route used to treat *every* non-ok backend answer and every transport
 * failure as the same thing: a 404 reading "the backend bridge has not
 * registered diff artifacts between these versions". That is a fabricated
 * reason. A 500, a 409, a validation 422 and a dead backend were all reported
 * as "no artifacts registered", so an outage was indistinguishable from a
 * world that legitimately has nothing to compare -- and the real backend
 * `detail` explaining the actual failure was discarded.
 *
 * The backend's own status and body are now forwarded verbatim on non-ok, and
 * a transport failure is a 503 that says so. A genuine "no diff" 404 from the
 * API still reads as a 404, because that is what the API actually decided.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const { searchParams } = new URL(request.url);
  const baseVersion = searchParams.get("base") || searchParams.get("base_version");
  const headVersion = searchParams.get("head") || searchParams.get("head_version") || "latest";

  try {
    const q = new URLSearchParams();
    if (baseVersion) q.set("base_version", baseVersion);
    if (headVersion) q.set("head_version", headVersion);

    const res = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/diff?${q.toString()}`,
      { headers: { Accept: "application/json" }, signal: AbortSignal.timeout(10000) }
    );

    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text) as unknown;
      } catch {
        data = null;
      }
    }

    if (!res.ok) {
      return NextResponse.json(
        data ?? { detail: `Version comparison failed with HTTP ${res.status}`, world_id: id },
        { status: res.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        { detail: "API returned a non-JSON diff body", world_id: id, available: false },
        { status: 502 }
      );
    }

    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API to compare versions: ${message}`,
        world_id: id,
        base_version_id: baseVersion ?? null,
        head_version_id: headVersion,
        available: false,
      },
      { status: 503 }
    );
  }
}
