import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../../../../lib/api/backend.ts";

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const searchParams = request.nextUrl.searchParams.toString();
    const query = searchParams ? `?${searchParams}` : "";
    const backendRes = await backendFetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/export/formats${query}`,
      {
        method: "GET",
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(10000),
      }
    );

    const data = parseJson(await backendRes.text());
    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `Fetch export formats failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to fetch export formats: ${message}`, available: false },
      { status: 503 }
    );
  }
}
