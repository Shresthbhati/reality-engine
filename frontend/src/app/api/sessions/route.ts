import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract `GET /api/sessions`
 * (apps/api/routes_sessions.py). The application API is the only source of
 * sessions.
 *
 * Shape contract: the backend returns `{"items": [...]}` on success and
 * FastAPI's `{"detail": "..."}` on error. This proxy therefore forwards the
 * backend body verbatim and never invents its own envelope.
 *
 * Previous behaviour synthesized `{error, sessions: [], count: 0}` on failure.
 * That envelope is a fabrication on two counts: it uses keys the backend
 * never emits (a consumer reading `data.sessions` gets `undefined` on the
 * success path), and it presents an empty list plus `count: 0` -- the exact
 * shape of "this user has no sessions" -- for what is actually a transport
 * failure. A UI that renders `sessions` would show an honest-looking empty
 * state instead of the outage. An unreachable backend is a 503 with a
 * reason, not a plausible empty result.
 */
function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  const worldId = new URL(request.url).searchParams.get("world_id");

  try {
    const query = worldId ? `?world_id=${encodeURIComponent(worldId)}` : "";
    const response = await fetch(`${BACKEND_URL}/api/sessions${query}`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await response.text());

    if (!response.ok) {
      return NextResponse.json(
        data ?? { detail: `Session listing failed with HTTP ${response.status}` },
        { status: response.status }
      );
    }

    return NextResponse.json(data ?? { items: [] });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for sessions: ${message}`,
        available: false,
      },
      { status: 503 }
    );
  }
}
