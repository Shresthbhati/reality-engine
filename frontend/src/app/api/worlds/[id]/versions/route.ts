import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract
 * `GET /api/worlds/{world_id}/versions` (apps/api/routes_worlds.py), which
 * mirrors real WorldStore lineage.
 *
 * The previous process-local `VERSION_STORE` — seeded with a fabricated
 * "v1-canonical-baseline" version, and appended to by the commit route — is
 * gone. Lineage that lives only in a Node array is not a version, and serving
 * it with `is_current: true` is a fabricated result. Empty lineage is reported
 * honestly; an unreachable API is a 503 rather than a silently empty list.
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/versions`,
      {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(10000),
      }
    );

    const text = await response.text();
    const data = text ? (JSON.parse(text) as { items?: unknown; detail?: string }) : null;

    if (!response.ok) {
      return NextResponse.json(
        {
          detail: data?.detail ?? `Version lineage request failed with HTTP ${response.status}`,
          world_id: id,
          available: false,
        },
        { status: response.status }
      );
    }

    // A world with no computed versions is an honest empty lineage, not an error.
    return NextResponse.json({ items: data?.items ?? [] });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for version lineage: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
