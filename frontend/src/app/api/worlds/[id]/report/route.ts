import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${id}/report`, {
      signal: AbortSignal.timeout(10000),
    });
    const data = await response.json();
    return NextResponse.json(
      response.ok ? data : { ...data, world_id: id },
      { status: response.status }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to connect to backend: ${message}`, world_id: id },
      { status: 503 }
    );
  }
}
