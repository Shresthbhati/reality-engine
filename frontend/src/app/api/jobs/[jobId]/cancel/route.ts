import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

function parseJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

/**
 * Thin proxy to the FastAPI contract
 * `POST /api/jobs/{job_id}/cancel` (apps/api/routes_jobs.py, PR #126).
 *
 * Backend semantics, all of which are meaningful and must survive the proxy:
 *   - queued job  -> 200 { id, status: "cancelled" }  (stopped immediately)
 *   - running job -> 202 { id, status, cancel_requested: true }  (winding down)
 *   - terminal    -> 409 (already succeeded/partial/failed/cancelled)
 *   - unknown id  -> 404
 *
 * The status is forwarded verbatim. In particular a 202 is NOT a confirmation
 * that the job is cancelled: the worker has only been asked to stop and will
 * grade CANCELLED at its next boundary. A caller that rendered 202 as
 * "cancelled" would claim a state the backend has not committed yet.
 */
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/jobs/${encodeURIComponent(jobId)}/cancel`,
      {
        method: "POST",
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(10000),
      }
    );

    const data = parseJson(await response.text());

    if (!response.ok) {
      // Forward 404 (unknown job) and 409 (already terminal) unchanged: both
      // are real backend answers, not transport failures.
      return NextResponse.json(
        data ?? { detail: `Cancel failed with HTTP ${response.status}`, job_id: jobId },
        { status: response.status }
      );
    }

    // A 2xx cancel must at least name the job it acted on. Without an id
    // there is nothing to poll, so the caller cannot know the outcome.
    const id =
      data && typeof data === "object" && typeof (data as { id?: unknown }).id === "string"
        ? (data as { id: string }).id
        : null;
    if (!id) {
      return NextResponse.json(
        {
          detail: "Cancel response carried no job id; the cancellation cannot be tracked.",
          job_id: jobId,
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
        detail: `Failed to reach the Reality Engine API to cancel this job: ${message}. ` +
          "The job may still be running; its state is unknown.",
        job_id: jobId,
        available: false,
      },
      { status: 503 }
    );
  }
}
