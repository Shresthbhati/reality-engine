import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "@/lib/api/backend";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { bundle_path, session_dir } = body;
    
    if (!bundle_path || !session_dir) {
      return NextResponse.json(
        { error: "Missing required fields: bundle_path, session_dir" },
        { status: 400 }
      );
    }
    
    const response = await backendFetch(`${BACKEND_URL}/api/sessions/ingest-mobile`, {
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