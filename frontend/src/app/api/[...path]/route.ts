import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../lib/api/backend.ts";

/**
 * Catch-all same-origin proxy (plan T-1.2, decision "proxy-only browser transport").
 *
 * The browser talks only to this origin; this handler forwards `/api/**` to the FastAPI backend and adds the server-side
 * bearer token (see backend.ts), so no token ever ships to the browser. Explicit handlers under `app/api/**` keep priority
 * over this catch-all (Next ranks specific segments first); this covers everything else.
 *
 * Bodies and responses are streamed, so multipart uploads and long downloads are not buffered here.
 */
export const dynamic = "force-dynamic";

// Request headers that must not be copied to the backend: hop-by-hop, fetch-managed, or credentials the browser must not choose.
const DROP_REQUEST = new Set([
  "host", "connection", "content-length", "transfer-encoding", "keep-alive", "upgrade", "te", "trailer",
  "proxy-authorization", "proxy-connection", "authorization", "cookie", "expect",
]);
// Response headers fetch already decoded or that describe the hop, not the payload.
const DROP_RESPONSE = new Set(["content-encoding", "content-length", "transfer-encoding", "connection", "keep-alive"]);

async function forward(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }): Promise<Response> {
  const { path } = await params;
  // Only plain path segments: no traversal out of /api, no encoded separators.
  if (!path.length || path.some((s) => s === "" || s === "." || s === ".." || /[\\/]/.test(s))) {
    return NextResponse.json({ detail: "Invalid API path" }, { status: 400 });
  }
  const target = `${BACKEND_URL}/api/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;

  const headers = new Headers();
  request.headers.forEach((value, key) => {
    if (!DROP_REQUEST.has(key.toLowerCase())) headers.set(key, value);
  });

  const hasBody = request.method !== "GET" && request.method !== "HEAD";
  try {
    const res = await backendFetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // Node's fetch requires this to stream a request body.
      ...(hasBody ? ({ duplex: "half" } as Record<string, unknown>) : {}),
      signal: request.signal,
      redirect: "manual",
    });
    const out = new Headers();
    res.headers.forEach((value, key) => {
      if (!DROP_RESPONSE.has(key.toLowerCase())) out.set(key, value);
    });
    return new Response(res.status === 204 || res.status === 304 ? null : res.body, { status: res.status, headers: out });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to reach the Reality Engine API: ${message}`, available: false },
      { status: 503 }
    );
  }
}

export const GET = forward;
export const POST = forward;
export const PUT = forward;
export const PATCH = forward;
export const DELETE = forward;
export const HEAD = forward;
