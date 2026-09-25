import { NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/sessions`, {
      signal: AbortSignal.timeout(5000),
    });
    
    if (!response.ok) {
      return NextResponse.json(
        { error: `Backend error: ${response.statusText}`, sessions: [], count: 0 },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to connect to backend: ${message}`, sessions: [], count: 0 },
      { status: 503 }
    );
  }
}
