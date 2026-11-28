import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract `GET /api/worlds/{id}/report` -- the
 * version's real compile-pipeline stage facts.
 *
 * The local `datasets/room_capture/pipeline_out/report.json` fallback is gone:
 * that report describes a CLI run that no WorldStore version records, so
 * showing it against this world would attribute one pipeline's stages to a
 * different reconstruction.
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
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/report${query}`,
      { signal: AbortSignal.timeout(10000) }
    );

    const data = (await response.json()) as Record<string, unknown>;
    return NextResponse.json(
      response.ok ? data : { ...data, world_id: id, available: false },
      { status: response.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: `Failed to reach the Reality Engine API for pipeline report: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}