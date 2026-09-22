import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { folder, output, seed = "pkg" } = body;
    
    if (!folder || !output) {
      return NextResponse.json(
        { error: "Missing required fields: folder, output" },
        { status: 400 }
      );
    }
    
    const response = await fetch(`${BACKEND_URL}/api/jobs/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder, output, seed }),
      signal: AbortSignal.timeout(60000),
    });
    
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to connect to backend: ${message}` },
      { status: 503 }
    );
  }
}