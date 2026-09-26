import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{world_id}/cameras`
 * (apps/api/routes_worlds.py), which returns the camera poses actually
 * registered on the world's current version.
 *
 * Two fabrications were removed here:
 *  - `datasets/room_capture/pipeline_out/cameras.json` was served for ids in
 *    a hard-coded allow-list, so an uncalibrated world showed someone
 *    else's calibrated cameras;
 *  - the response invented `frame` / `rotation_convention` /
 *    `image_size: [1280, 960]` defaults instead of reporting the version's
 *    real values (the API returns `image_size: null` when unrecorded).
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/cameras`,
      {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(30000),
      }
    );

    const data = (await response.json().catch(() => null)) as unknown;

    if (!response.ok) {
      return NextResponse.json(
        data ?? {
          error: "Camera request failed",
          world_id: id,
          available: false,
        },
        { status: response.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        { error: "API returned a non-JSON cameras body", world_id: id, available: false },
        { status: 502 }
      );
    }

    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for this world's cameras: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
