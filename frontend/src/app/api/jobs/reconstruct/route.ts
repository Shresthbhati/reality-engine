import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { package_path, output_path, colmap_binary = "colmap", gpu = false, no_real_geometry = false } = body;
    
    if (!package_path || !output_path) {
      return NextResponse.json(
        { error: "Missing required fields: package_path, output_path" },
        { status: 400 }
      );
    }
    
    const response = await fetch(`${BACKEND_URL}/api/jobs/reconstruct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ package_path, output_path, colmap_binary, gpu, no_real_geometry }),
      signal: AbortSignal.timeout(300000),
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