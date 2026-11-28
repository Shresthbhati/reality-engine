import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{world_id}/diff` (apps/api/routes_worlds.py).
 *
 * Three distinct outcomes, kept distinct:
 *  - backend 2xx              -> the real WorldDiff, verbatim
 *  - backend 404              -> no diff exists for these versions. The
 *                                backend's own 404 body/reason is forwarded
 *                                rather than replaced with an invented
 *                                explanation.
 *  - transport failure        -> 503 "backend unreachable"
 *
 * The previous revision caught every failure — including a 500 and a total
 * outage — and answered 404 "The backend bridge has not registered diff
 * artifacts between these versions." That is a fabricated claim about backend
 * state: it asserts the bridge knows there is no diff when in fact the bridge
 * could not be asked. A user would conclude their lineage is broken rather
 * than that the API is down, and no retry was offered because 404 reads as
 * final.
 */
function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

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
      { signal: AbortSignal.timeout(10000) }
    );

    const data = parseJson(await res.text());

    if (res.ok) {
      // A 200 that is not valid JSON means the diff is unknown. Returning 200
      // with a fabricated `{error: ...}` body would let a caller treat a
      // broken response as a real (empty) diff result.
      if (data === null) {
        return NextResponse.json(
          {
            detail: "Version comparison returned a non-JSON body; the result is unknown.",
            world_id: id,
            available: false,
          },
          { status: 502 },
        );
      }
      return NextResponse.json(data);
    }

    // Forward the backend's own status and body. A real 404 keeps its real
    // reason; a 500 stays a 500 instead of being laundered into "no diff".
    return NextResponse.json(
      data ?? { detail: `Diff request failed with HTTP ${res.status}`, world_id: id },
      { status: res.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail:
          `Failed to reach the Reality Engine API for this version comparison: ${message}. ` +
          "This is a connectivity failure, not a missing diff — the lineage may well exist.",
        world_id: id,
        base_version_id: baseVersion || null,
        head_version_id: headVersion,
        available: false,
      },
      { status: 503 }
    );
  }
}
