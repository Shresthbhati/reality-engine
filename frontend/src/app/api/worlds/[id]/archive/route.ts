import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../../../lib/api/backend.ts";

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export async function POST(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const backendRes = await backendFetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/archive`, {
      method: "POST",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await backendRes.text());
    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `Archive world failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to archive world: ${message}`, available: false },
      { status: 503 }
    );
  }
}
