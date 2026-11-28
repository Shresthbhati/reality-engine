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
 * Thin proxy to `DELETE /api/evidence/{evidence_id}`
 * (apps/api/routes_misc.py, `status_code=204`).
 *
 * Contract, read from the backend rather than assumed:
 *   - 204 No Content on success. The backend returns an EMPTY body, so the
 *     proxy must not try to JSON-parse one or invent a payload.
 *   - 404 when no Evidence row exists.
 *   - 5xx / transport failures propagate.
 *
 * There is deliberately no local removal. If the proxy deleted something in
 * memory while the API was down, the UI would drop an Evidence item that is
 * still on the server. Removal happens only after a real 204.
 */
export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ evidenceId: string }> }
) {
  const { evidenceId } = await context.params;
  void request;

  try {
    const response = await fetch(
      `${BACKEND_URL}/api/evidence/${encodeURIComponent(evidenceId)}`,
      {
        method: "DELETE",
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(15000),
      }
    );

    // 204 is the success proof. It must be checked explicitly, because a
    // 2xx with an unexpected body means we cannot confirm the deletion.
    if (response.status === 204) {
      return new NextResponse(null, { status: 204 });
    }

    const data = parseJson(await response.text());

    if (!response.ok) {
      // 404 (no such evidence) and any 4xx/5xx keep their real status so the
      // client can distinguish "already gone" from "backend is broken".
      return NextResponse.json(
        data ?? {
          detail: `Evidence delete failed with HTTP ${response.status}`,
          evidence_id: evidenceId,
        },
        { status: response.status }
      );
    }

    // A 2xx that is not the documented 204 is an unexpected contract. Report a
    // failure rather than letting the caller assume the item was deleted.
    return NextResponse.json(
      {
        detail:
          `The API answered ${response.status} instead of the documented 204. ` +
          "The evidence deletion could not be confirmed.",
        evidence_id: evidenceId,
        available: false,
      },
      { status: 502 }
    );
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        detail:
          `Failed to reach the Reality Engine API to delete this evidence: ${message}. ` +
          "The item was NOT removed.",
        evidence_id: evidenceId,
        available: false,
      },
      { status: 503 }
    );
  }
}
