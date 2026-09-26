import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy for an Evidence artifact image, so the Inspector can render the
 * exact stored bytes behind a trace.
 *
 * Two defects were fixed here:
 *  - the backend contract is `GET /api/evidence/{id}/artifact`
 *    (apps/api/routes_misc.py); this route asked for `/file`, a path the API
 *    does not serve, so every request fell through to the second defect:
 *  - `datasets/room_capture/images/<evidenceId>.<ext>` was served for any
 *    evidence id that happened to match one of the bundled files, so an
 *    unrelated capture's photo could be presented as this evidence item.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string; evidenceId: string }> }
) {
  const { id, evidenceId } = await context.params;
  void request;

  try {
    const res = await fetch(
      `${BACKEND_URL}/api/evidence/${encodeURIComponent(evidenceId)}/artifact`,
      { signal: AbortSignal.timeout(30000) }
    );

    if (res.ok) {
      const blob = await res.arrayBuffer();
      return new NextResponse(blob, {
        headers: {
          "Content-Type": res.headers.get("content-type") || "application/octet-stream",
          "Cache-Control": "public, max-age=3600",
        },
      });
    }

    const data = (await res.json().catch(() => null)) as unknown;
    return NextResponse.json(
      data ?? {
        error: "Evidence image not available",
        evidence_id: evidenceId,
        world_id: id,
        available: false,
      },
      { status: res.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for this evidence image: ${message}`,
        evidence_id: evidenceId,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
