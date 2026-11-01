import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "@/lib/api/backend";

/**
 * Thin proxy to the FastAPI contract `GET/POST /api/worlds`
 * (apps/api/routes_worlds.py). The application API is the only source of
 * worlds.
 *
 * The previous revision of this route also `unshift`ed a fabricated
 * "world-compiled-seed42" row whenever `datasets/room_capture/pipeline_out/
 * worldir.json` happened to exist on the build machine: hard-coded San
 * Francisco coordinates, an invented `0.05 km²` coverage, "25 Evidence",
 * "Verified Pipeline Out". That made the product claim a reconstructed
 * world that no session, no evidence and no WorldStore version ever
 * produced. An unreachable API is a 503 now, not an invented world list.
 */
function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export async function GET() {
  try {
    const backendRes = await backendFetch(`${BACKEND_URL}/api/worlds`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await backendRes.text());

    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `World listing failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }

    if (data === null) {
      // 200 with an empty or unparsable body is a broken backend, not "no worlds"
      return NextResponse.json(
        { detail: "The Reality Engine API returned an empty or invalid body for the world listing." },
        { status: 502 }
      );
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for worlds: ${message}`,
        available: false,
      },
      { status: 503 }
    );
  }
}

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "World body must be valid JSON" }, { status: 400 });
  }

  try {
    const res = await backendFetch(`${BACKEND_URL}/api/worlds`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await res.text());

    if (!res.ok) {
      return NextResponse.json(
        data ?? { detail: `World creation failed with HTTP ${res.status}` },
        { status: res.status }
      );
    }

    return NextResponse.json(data ?? {}, { status: 201 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Backend unavailable; world was not created: ${message}`,
        available: false,
      },
      { status: 503 }
    );
  }
}
