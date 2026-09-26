import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{world_id}/worldir`
 * (apps/api/routes_worlds.py), which parses the world's current WorldStore
 * version and 404s when no reconstruction version exists yet.
 *
 * This route previously fell back to reading
 * `datasets/room_capture/pipeline_out/worldir.json` for any world id in a
 * hard-coded allow-list ("world-compiled-seed42", "room-capture",
 * "dataset-room-capture", `world-room*`). That let an uncompiled world
 * present a bundled, unrelated compiled WorldIR as its own. WorldIR now
 * comes from the world's own version, or the request fails honestly.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/worldir`,
      {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(30000),
      }
    );

    const data = (await response.json().catch(() => null)) as unknown;

    if (!response.ok) {
      return NextResponse.json(
        data ?? {
          error: "WorldIR request failed",
          world_id: id,
          available: false,
        },
        { status: response.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        { error: "API returned a non-JSON WorldIR body", world_id: id, available: false },
        { status: 502 }
      );
    }

    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for this WorldIR: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
