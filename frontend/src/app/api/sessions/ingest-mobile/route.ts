import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(request: NextRequest) {
  // A malformed body is a client error (400), not a backend outage.
  let body: { bundle_path?: string; session_dir?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Request body must be valid JSON" },
      { status: 400 }
    );
  }

  try {
    const { bundle_path, session_dir } = body;

    if (!bundle_path || !session_dir) {
      return NextResponse.json(
        { error: "Missing required fields: bundle_path, session_dir" },
        { status: 400 }
      );
    }
    
    const response = await fetch(`${BACKEND_URL}/api/sessions/ingest-mobile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bundle_path, session_dir }),
      signal: AbortSignal.timeout(30000),
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