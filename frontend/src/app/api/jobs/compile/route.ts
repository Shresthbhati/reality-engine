import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { dataset_path, output_path, colmap_binary = "colmap", no_depth = false, no_mesh = false } = body;
    
    if (!dataset_path) {
      return NextResponse.json(
        { error: "Missing required field: dataset_path" },
        { status: 400 }
      );
    }
    
    const response = await fetch(`${BACKEND_URL}/api/jobs/compile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_path, output_path, colmap_binary, no_depth, no_mesh }),
      signal: AbortSignal.timeout(300000), // 5 minutes for full compile
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