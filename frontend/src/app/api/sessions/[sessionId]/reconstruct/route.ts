import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract
 * `POST /api/sessions/{session_id}/reconstruct` (apps/api/routes_sessions.py).
 *
 * Enqueueing is not success: the response carries the real `job_id`, which
 * the caller polls at `/api/jobs/{jobId}` for the terminal status.
 */
function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

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

    const data = parseJson(await response.text());

    if (!response.ok) {
      // Forward the real status: 404 (unknown session) and 409 (a
      // reconstruction is already queued/running) are distinct, meaningful
      // outcomes and must not be flattened into a generic failure.
      return NextResponse.json(
        data ?? {
          detail: `Reconstruction enqueue failed with HTTP ${response.status}`,
          session_id: sessionId,
        },
        { status: response.status }
      );
    }

    // Enqueueing is only meaningful if a job to track was actually created.
    // A 2xx with no `job_id` would leave the UI polling an id that does not
    // exist, and would show a "reconstruction started" state for a job the
    // API never created.
    const jobId =
      data && typeof data === "object" && typeof (data as { job_id?: unknown }).job_id === "string"
        ? (data as { job_id: string }).job_id
        : null;
    if (!jobId) {
      return NextResponse.json(
        {
          detail:
            "The API accepted the request but returned no job id. No reconstruction " +
            "can be tracked, so none should be reported as started.",
          session_id: sessionId,
          available: false,
        },
        { status: 502 }
      );
    }

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