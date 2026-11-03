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

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const backendRes = await backendFetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}`, {
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await backendRes.text());
    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `World fetch failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to reach the Reality Engine API for world: ${message}`, available: false },
      { status: 503 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  try {
    const body = await request.text();
    const backendRes = await backendFetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body,
      signal: AbortSignal.timeout(10000),
    });

    const data = parseJson(await backendRes.text());
    if (!backendRes.ok) {
      return NextResponse.json(
        data ?? { detail: `World update failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to update world: ${message}`, available: false },
      { status: 503 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const { searchParams } = new URL(request.url);
  const permanent = searchParams.get("permanent") === "true";
  const url = `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}?permanent=${permanent}`;

  try {
    const backendRes = await backendFetch(url, {
      method: "DELETE",
      headers: { Accept: "application/json" },
      signal: AbortSignal.timeout(10000),
    });

    if (!backendRes.ok) {
      const data = parseJson(await backendRes.text());
      return NextResponse.json(
        data ?? { detail: `World deletion failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return new NextResponse(null, { status: 204 });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to delete world: ${message}`, available: false },
      { status: 503 }
    );
  }
}
