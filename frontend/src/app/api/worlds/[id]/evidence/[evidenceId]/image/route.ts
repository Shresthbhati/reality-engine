import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the API's own evidence artifact surface
 * `GET /api/evidence/{evidence_id}/artifact` (apps/api/routes_misc.py).
 *
 * The local `datasets/room_capture/images` fallback is gone: an image on the
 * Next server's disk is not an Evidence row's stored artifact, so serving it
 * for a given evidence id would attribute an unrelated photo to that
 * record. Evidence preview only ever shows what the content store holds.
 *
 * The API distinguishes 404 (no such evidence / no artifact recorded) from
 * 410 (recorded, but the blob is gone from the content store); both are
 * passed through so the UI can say which one it is.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string; evidenceId: string }> }
) {
  const { id, evidenceId } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/evidence/${encodeURIComponent(evidenceId)}/artifact`,
      { signal: AbortSignal.timeout(10000) }
    );

    if (response.ok) {
      const blob = await response.arrayBuffer();
      const contentType = response.headers.get("content-type") || "application/octet-stream";
      return new NextResponse(blob, {
        headers: { "Content-Type": contentType, "Cache-Control": "private, max-age=300" },
      });
    }

    return NextResponse.json(
      {
        error: `Evidence file request failed with HTTP ${response.status}`,
        evidence_id: evidenceId,
        world_id: id,
        available: false,
      },
      { status: response.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: `Failed to reach the Reality Engine API for evidence file: ${message}`,
        evidence_id: evidenceId,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}