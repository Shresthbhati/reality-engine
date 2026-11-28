import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{id}/cameras` (the version's real registered
 * camera poses).
 *
 * The local `datasets/room_capture/cameras.json` fallback is gone: those poses
 * were never committed as a WorldStore version, so serving them as this
 * world's calibration is fabricated state. `image_size` / `frame` defaults
 * belong to the API, not to invented values here.
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
      { signal: AbortSignal.timeout(10000) }
    );

    const data = (await response.json()) as Record<string, unknown>;
    if (!response.ok) {
      return NextResponse.json(
        {
          error:
            (data?.detail as string | undefined) ??
            `Camera request failed (HTTP ${response.status})`,
          world_id: id,
          available: false,
        },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: `Failed to reach the Reality Engine API for camera poses: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}