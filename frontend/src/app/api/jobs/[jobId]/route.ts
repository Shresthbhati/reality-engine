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
    
    if (!response.ok) {
      if (response.status === 404) {
        return NextResponse.json(
          { error: "Job not found" },
          { status: 404 }
        );
      }
      return NextResponse.json(
        { error: `Backend error: ${response.statusText}` },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to connect to backend: ${message}` },
      { status: 503 }
    );
  }
}