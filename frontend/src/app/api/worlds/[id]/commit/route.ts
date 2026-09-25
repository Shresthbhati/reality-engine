import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

/**
 * Thin proxy to the FastAPI contract
 * `POST /api/worlds/{world_id}/commit` (apps/api/routes_worlds.py).
 *
 * The request body is forwarded verbatim so the API's own commit semantics
 * apply unchanged: the stale-parent 409, the duplicate-retry short circuit,
 * and the CAS adoption check that rejects a lost update. There is deliberately
 * no in-memory fallback ledger here — a commit that only landed in a Node
 * process-local array is not a persisted WorldStore version, and returning
 * `success: true` for it when the API is unreachable is a fabricated result.
 */
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { detail: "Commit body must be valid JSON" },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/commit`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(10000),
      }
    );

    const text = await response.text();
    const data = text ? (JSON.parse(text) as unknown) : null;
    return NextResponse.json(data, { status: response.status });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail: `Failed to reach the Reality Engine API to commit this correction: ${message}`,
        world_id: id,
        available: false,
      },
      { status: 503 }
    );
  }
}
