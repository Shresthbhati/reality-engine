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
 * Thin proxy to `POST /api/uploads` (apps/api/routes_misc.py,
 * `status_code=201`).
 *
 * Contract, read from the backend rather than assumed:
 *   - Content-Type: multipart/form-data
 *   - field `file`  : the uploaded artifact (required; FastAPI 422 if absent)
 *   - field `session_id` : optional query/body association
 *   - the body is read BOUNDED (`UPLOAD_MAX_BYTES`, default 1 GiB) and an
 *     empty body is rejected with 400 "Empty upload"
 *   - success 201: { upload_id, evidence_id, job_id, checksum }
 *   - the server also enqueues PROCESS_EVIDENCE, so `job_id` is how the
 *     caller tracks processing
 *
 * The `file` field is forwarded as the ORIGINAL stream, so this proxy never
 * buffers a multi-gigabyte artifact in the Node process.
 *
 * A 2xx is not proof of ingestion on its own: the response must name the
 * `evidence_id` the server actually created. Without it there is no row to
 * show, so the proxy reports a failure rather than a phantom success.
 */
export async function POST(request: NextRequest) {
  // The browser must set the multipart boundary itself; re-encoding here
  // would corrupt the stream. Forward the body and content-type verbatim.
  const contentType = request.headers.get("content-type") ?? "";

  if (!contentType.toLowerCase().includes("multipart/form-data")) {
    return NextResponse.json(
      {
        detail:
          "Uploads must be multipart/form-data with a 'file' field; the API " +
          "rejects any other content type.",
      },
      { status: 415 }
    );
  }

  let body: ArrayBuffer;
  try {
    body = await request.arrayBuffer();
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Upload body could not be read: ${message}` },
      { status: 400 }
    );
  }

  if (body.byteLength === 0) {
    // The backend rejects an empty upload with 400; catching it here gives a
    // clearer message and never contacts the API with a meaningless request.
    return NextResponse.json(
      { detail: "Empty upload: no file content was received." },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(`${BACKEND_URL}/api/uploads${request.nextUrl.search}`, {
      method: "POST",
      headers: { "Content-Type": contentType },
      body,
      signal: AbortSignal.timeout(300_000),
    });

    const text = await response.text();
    const data = parseJson(text);

    if (!response.ok) {
      // 400 (empty/invalid), 413 (too large), 422 (missing `file`), 5xx all
      // keep their real status and the API's own message.
      return NextResponse.json(
        data ?? {
          detail: `Upload failed with HTTP ${response.status}`,
        },
        { status: response.status }
      );
    }

    const evidenceId =
      data && typeof data === "object" && typeof (data as { evidence_id?: unknown }).evidence_id === "string"
        ? (data as { evidence_id: string }).evidence_id
        : null;
    if (!evidenceId) {
      // A 201 without an evidence_id means no Evidence row was created that we
      // can show. Do not report a successful ingestion.
      return NextResponse.json(
        {
          detail:
            "The API accepted the upload but returned no evidence id; ingestion " +
            "could not be confirmed.",
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
        detail:
          `Failed to reach the Reality Engine API to ingest this upload: ${message}. ` +
          "Nothing was stored.",
        available: false,
      },
      { status: 503 }
    );
  }
}
