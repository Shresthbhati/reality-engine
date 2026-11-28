import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(request: NextRequest) {
  // Parse the request body OUTSIDE the transport try/catch. A malformed body
  // is a client error (400), not a backend outage: reporting it as
  // "Failed to connect to backend" (503) tells the operator their API is down
  // when the request never left the browser.
  let body: { folder?: string; output?: string; seed?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON" },
      { status: 400 }
    );
  }

  try {
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