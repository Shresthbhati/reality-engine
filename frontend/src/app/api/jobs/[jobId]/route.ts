import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "@/lib/api/backend";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;
  
  try {
    const response = await backendFetch(`${BACKEND_URL}/api/jobs/${jobId}`, {
      signal: AbortSignal.timeout(5000),
    });
    
    // Forward the backend's own body and status: FastAPI errors are `{detail}`, and the API client reads `detail`.
    // Rewriting them to a private `{error}` shape threw away the real reason (e.g. a foreign job's 404 vs a 403).
    const text = await response.text();
    let data: unknown = null;
    try {
      data = text ? (JSON.parse(text) as unknown) : null;
    } catch {
      data = null;
    }
    if (!response.ok) {
      return NextResponse.json(data ?? { detail: `Job lookup failed with HTTP ${response.status}` }, { status: response.status });
    }
    if (data === null) {
      return NextResponse.json({ detail: "The Reality Engine API returned an empty or invalid body for this job." }, { status: 502 });
    }
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to reach the Reality Engine API for this job: ${message}`, available: false },
      { status: 503 }
    );
  }
}