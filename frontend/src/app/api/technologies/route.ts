import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../lib/api/backend.ts";

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const subsystem = searchParams.get("subsystem");
    const backendUrl = subsystem
      ? `${BACKEND_URL}/api/technologies?subsystem=${encodeURIComponent(subsystem)}`
      : `${BACKEND_URL}/api/technologies`;

    const backendRes = await backendFetch(backendUrl, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await backendRes.text());

    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `Technology listing failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }

    if (data === null) {
      return NextResponse.json(
        { detail: "The Reality Engine API returned an empty or invalid body for the technology listing." },
        { status: 502 }
      );
    }
    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API for technologies: ${message}`,
        available: false,
      },
      { status: 503 }
    );
  }
}
