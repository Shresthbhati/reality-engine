import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to
 * `GET /api/worlds/{world_id}/entities/{entity_id}/provenance`
 * (apps/api/routes_worlds.py), which traces the entity against the world's
 * own WorldStore version and the real Evidence rows.
 *
 * The previous local-dataset fallback read
 * `datasets/room_capture/pipeline_out/world_ir.json` (a file that does not
 * exist in that directory, so the branch was dead) and, when it did resolve,
 * stamped the fabricated version id `v1-canonical-baseline` and the invented
 * session id `session-room-capture` onto the answer. Provenance that names a
 * version the store never held is worse than no provenance: it is now a
 * proxy or an honest failure.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string; entityId: string }> }
) {
  const { id, entityId } = await context.params;
  void request;

  try {
    const res = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/entities/${encodeURIComponent(entityId)}/provenance`,
      {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(30000),
      }
    );

    const data = (await res.json().catch(() => null)) as unknown;

    if (!res.ok) {
      return NextResponse.json(
        data ?? {
          error: `Provenance unavailable for entity '${entityId}'`,
          world_id: id,
          entity_id: entityId,
          available: false,
        },
        { status: res.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        {
          error: "API returned a non-JSON provenance body",
          world_id: id,
          entity_id: entityId,
          available: false,
        },
        { status: 502 }
      );
    }

    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for entity provenance: ${message}`,
        world_id: id,
        entity_id: entityId,
        available: false,
      },
      { status: 503 }
    );
  }
}
