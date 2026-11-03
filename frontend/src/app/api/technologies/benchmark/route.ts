import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../../lib/api/backend.ts";

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const techId = searchParams.get("tech_id");
    const backendUrl = techId
      ? `${BACKEND_URL}/api/technologies/benchmark?tech_id=${encodeURIComponent(techId)}`
      : `${BACKEND_URL}/api/technologies/benchmark`;

    const backendRes = await backendFetch(backendUrl, {
      method: "POST",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(30000),
    });

    const data = parseJson(await backendRes.text());

    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `Technology benchmark failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        { detail: "The Reality Engine API returned an empty or invalid body for the benchmark." },
        { status: 502 }
      );
    }
    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for technology benchmark: ${message}`,
        available: false,
      },
      { status: 503 }
    );
  }
}
