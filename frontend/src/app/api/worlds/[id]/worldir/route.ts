import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract `GET /api/worlds/{id}/worldir`
 * (apps/api/routes_worlds.py) -- the version's compiled WorldIR.
 *
 * The `world-compiled-seed42` / `datasets/room_capture/pipeline_out` local
 * fallback is gone. A worldir.json on the Next server's disk is not a
 * WorldStore version: it is unversioned, unattributed to any evidence, and
 * invisible to the API's own version/lineage/provenance endpoints, so serving
 * it as a reconstruction is fabricated state.
 *
 * The API distinguishes "no version yet" (404) from "corrupt version" (409);
 * that distinction is preserved so the UI can say which one it is.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const version = new URL(request.url).searchParams.get("version");

  try {
    const query = version ? `?version=${encodeURIComponent(version)}` : "";
    const response = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/worldir${query}`,
      { signal: AbortSignal.timeout(15000) }
    );

    if (response.ok) {
      return NextResponse.json(await response.json());
    }

    const detail = await response
      .json()
      .then((b: { detail?: string }) => b?.detail)
      .catch(() => undefined);

    return NextResponse.json(
      {
        detail:
          detail ??
          `WorldIR request failed with HTTP ${response.status}`,
        world_id: id,
        version: version ?? null,
        available: false,
        // 404 = no reconstruction version yet; 409 = version exists but is
        // unreadable/corrupt. The UI distinguishes these two honestly.
        reason: response.status === 404 ? "no_version" : "unreadable_version",
      },
      { status: response.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for WorldIR: ${message}`,
        world_id: id,
        available: false,
        reason: "unreachable",
      },
      { status: 503 }
    );
  }
}