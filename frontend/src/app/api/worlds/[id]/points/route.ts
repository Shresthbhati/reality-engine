import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${id}/points`, {
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) {
      return NextResponse.json(
        { error: "No point cloud available for this world", world_id: id },
        { status: response.status }
      );
    }
    const buffer = await response.arrayBuffer();
    return new NextResponse(buffer, {
      headers: { "Content-Type": "application/octet-stream" },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to connect to backend: ${message}`, world_id: id },
      { status: 503 }
    );
  }
}
