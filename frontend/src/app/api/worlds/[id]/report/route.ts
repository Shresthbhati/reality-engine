import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `GET /api/worlds/{world_id}/report`
 * (apps/api/routes_worlds.py), which returns the current version's real
 * compile-pipeline report (the CLI's report.json stages, carried through
 * WorldStore).
 *
 * This route previously fell back to
 * `datasets/room_capture/pipeline_out/report.json` for ids in a hard-coded
 * allow-list, so an uncompiled world could present a bundled pipeline report
 * as its own reconstruction evidence. The report now comes from the world's
 * own version, or the request fails honestly.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const { searchParams } = new URL(request.url);
  const version = searchParams.get("version");

  try {
    const url = new URL(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/report`);
    if (version) url.searchParams.set("version", version);

    const response = await fetch(url.toString(), {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(30000),
    });

    const data = (await response.json().catch(() => null)) as unknown;

    if (!response.ok) {
      return NextResponse.json(
        data ?? {
          error: "Pipeline report request failed",
          world_id: id,
          available: false,
        },
        { status: response.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        { error: "API returned a non-JSON report body", world_id: id, available: false },
        { status: 502 }
      );
    }

    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for this world's report: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
