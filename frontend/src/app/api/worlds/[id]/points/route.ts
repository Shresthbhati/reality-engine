import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{world_id}/points`
 * (apps/api/routes_worlds.py), which renders the version's reconstruction
 * points from the geometries' own artifact bytes as one PLY stream.
 *
 * This route previously served
 * `datasets/room_capture/pipeline_out/points.ply` for ids in a hard-coded
 * allow-list, so a world with no point geometry could display a bundled
 * point cloud that was never reconstructed for it. Points now come from the
 * world's own version, or the request fails honestly.
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
      { signal: AbortSignal.timeout(60000) }
    );

    if (response.ok) {
      const buffer = await response.arrayBuffer();
      return new NextResponse(buffer, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition":
            response.headers.get("content-disposition") ??
            `attachment; filename="${id}_points.ply"`,
        },
      });
    }

    const data = (await response.json().catch(() => null)) as unknown;
    return NextResponse.json(
      data ?? {
        error: "Point cloud request failed",
        world_id: id,
        available: false,
      },
      { status: response.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for this world's points: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
