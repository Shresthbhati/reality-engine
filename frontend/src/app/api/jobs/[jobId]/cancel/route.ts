import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to `POST /api/jobs/{job_id}/cancel` (apps/api/routes_jobs.py):
 * 200 for an already-cancelled or queued job, 202 while a running job's
 * subprocess tree is terminated and it winds down, 404 for an unknown job,
 * 409 if the job already reached some OTHER terminal state. Forwards the
 * backend's status code and body verbatim rather than collapsing them, so
 * the frontend can tell "already succeeded, can't cancel" (409) apart from
 * "cancellation accepted" (202).
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
        signal: AbortSignal.timeout(10000),
      }
    );

    const data = await response.json().catch(() => null);

    if (!response.ok) {
      return NextResponse.json(
        data ?? { error: "Cancel request failed", job_id: jobId },
        { status: response.status }
      );
    }

    return NextResponse.json(data, { status: response.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: `Failed to reach the Reality Engine API to cancel this job: ${message}`, job_id: jobId },
      { status: 503 }
    );
  }
}
