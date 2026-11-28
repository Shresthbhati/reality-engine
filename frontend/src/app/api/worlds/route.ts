import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract `GET /api/worlds`.
 *
 * There is no local-dataset discovery fallback: a world that exists only as
 * a file on the Next server's disk is not a world the API reconstructed, and
 * listing it as a capture (with invented lat/lng and evidence counts) would
 * be fabricated state. If the API is unreachable the list is empty and the
 * UI shows its offline state.
 */
export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend error: HTTP ${response.status}`, items: [] },
        { status: response.status }
      );
    }

    const data = (await response.json()) as { items?: unknown[] };
    // Pass the API's own DTOs through untouched. The UI adapts them in
    // lib/api/adapters; inventing `lat`/`evidenceCount`/`coverageKm2` here
    // would fabricate fields the API never reported.
    return NextResponse.json({ items: data.items ?? [] });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to reach the Reality Engine API: ${message}`, items: [] },
      { status: 503 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const res = await fetch(`${BACKEND_URL}/api/worlds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data, { status: 201 });
    }

    return NextResponse.json(
      { error: "Backend failed to create world" },
      { status: res.status }
    );
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable; world was not created" },
      { status: 503 }
    );
  }
}
