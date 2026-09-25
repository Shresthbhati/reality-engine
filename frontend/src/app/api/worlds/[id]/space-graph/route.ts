import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const { searchParams } = new URL(request.url);
  const versionId = searchParams.get("version_id");

  try {
    const url = new URL(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/space-graph`);
    if (versionId) {
      url.searchParams.set("version_id", versionId);
    }
    const response = await fetch(url.toString(), {
      signal: AbortSignal.timeout(3000),
    });

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    }

    return NextResponse.json(
      { error: "Space graph unavailable for this world", world_id: id },
      { status: response.status }
    );
  } catch {
    return NextResponse.json(
      { error: "Backend bridge offline", world_id: id },
      { status: 503 }
    );
  }
}
