import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { bundle_path, output_path, max_tasks = 12, overwrite = false } = body;
    
    if (!bundle_path || !output_path) {
      return NextResponse.json(
        { error: "Missing required fields: bundle_path, output_path" },
        { status: 400 }
      );
    }
    
    const response = await fetch(`${BACKEND_URL}/api/sessions/mobile-tasks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bundle_path, output_path, max_tasks, overwrite }),
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