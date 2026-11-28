import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{id}/points` (the version's real point
 * cloud, as a binary PLY stream).
 *
 * The `world-compiled-seed42` / `datasets/room_capture` local-file fallback
 * is gone: a PLY sitting on the Next server's disk was never committed as a
 * WorldStore version, so serving it as this world's reconstruction is
 * fabricated state.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/points`,
      { signal: AbortSignal.timeout(15000) }
    );

    if (response.ok) {
      const buffer = await response.arrayBuffer();
      return new NextResponse(buffer, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${id}_points.ply"`,
        },
      });
    }

    const detail = await response
      .json()
      .then((b: { detail?: string }) => b?.detail)
      .catch(() => undefined);
    return NextResponse.json(
      {
        error: detail ?? `Point cloud request failed (HTTP ${response.status})`,
        world_id: id,
        available: false,
      },
      { status: response.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: `Failed to reach the Reality Engine API for point cloud: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
