import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

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
    const backendRes = await fetch(`${BACKEND_URL}/api/worlds`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    const text = await backendRes.text();
    const data = parseJson(text);

    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `World listing failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }

    // A 200 with an unparseable body is a broken response, not an empty world
    // list. Defaulting to `{items: []}` would show "No Worlds yet" for a total
    // backend malfunction.
    if (data === null) {
      return NextResponse.json(
        {
          detail: "World listing returned a non-JSON body; the result is unknown.",
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
    const res = await fetch(`${BACKEND_URL}/api/worlds`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10000),
    });

    const text = await res.text();
    const data = parseJson(text);

    if (!res.ok) {
      return NextResponse.json(
        data ?? { detail: `World creation failed with HTTP ${res.status}` },
        { status: res.status }
      );
    }

    // A 2xx that does not name a world is not a created world. Returning 201
    // with an invented/empty body would navigate the UI to a world that does
    // not exist, or show a blank created card.
    const createdId =
      data && typeof data === "object" && typeof (data as { id?: unknown }).id === "string"
        ? (data as { id: string }).id
        : null;
    if (!createdId) {
      return NextResponse.json(
        {
          detail: "World creation returned no world id; the world was not confirmed as created.",
          available: false,
        },
        { status: 502 }
      );
    }

    return NextResponse.json(data, { status: 201 });
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
