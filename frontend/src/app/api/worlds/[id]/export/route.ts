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
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const body = await request.json().catch(() => ({}));
    const backendRes = await backendFetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/export`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60000),
      }
    );

    const data = parseJson(await backendRes.text());
    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `Export failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to export world: ${message}`, available: false },
      { status: 503 }
    );
  }
}
