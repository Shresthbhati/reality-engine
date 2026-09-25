import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract
 * `POST /api/sessions/{session_id}/reconstruct` (apps/api/routes_sessions.py).
 *
 * Enqueueing is not success: the response carries the real `job_id`, which
 * the caller polls at `/api/jobs/{jobId}` for the terminal status.
 */
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ sessionId: string }> }
) {
  const { sessionId } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/sessions/${encodeURIComponent(sessionId)}/reconstruct`,
      {
        method: "POST",
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(15000),
      }
    );

    const text = await response.text();
    const data = text ? (JSON.parse(text) as unknown) : null;

    return NextResponse.json(data, { status: response.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API to enqueue reconstruction: ${message}`,
        session_id: sessionId,
        available: false,
      },
      { status: 503 }
    );
  }
}