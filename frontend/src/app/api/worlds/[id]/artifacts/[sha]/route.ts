import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../../../../lib/api/backend.ts";

/**
 * Same-origin proxy for a world's content-addressed artifact (texture atlas PNG).
 * The viewer builds `/api/worlds/{id}/artifacts/{sha}` (see `atlasUrl` in room-shell-model.ts); `<img>`/texture
 * requests cannot send the server-side bearer token, so this route adds it, as every other proxy here does.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string; sha: string }> }
) {
  const { id, sha } = await params;
  try {
    const version = request.nextUrl.searchParams.get("version");
    const qs = version ? `?version=${encodeURIComponent(version)}` : "";
    const backendRes = await backendFetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/artifacts/${encodeURIComponent(sha)}${qs}`,
      { method: "GET", signal: AbortSignal.timeout(30000) }
    );
    if (!backendRes.ok) {
      return NextResponse.json(
        { detail: `Artifact fetch failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }
    return new NextResponse(await backendRes.arrayBuffer(), {
      status: 200,
      headers: {
        "Content-Type": backendRes.headers.get("Content-Type") || "application/octet-stream",
        "Cache-Control": backendRes.headers.get("Cache-Control") || "public, max-age=31536000, immutable",
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to fetch artifact: ${message}`, available: false },
      { status: 503 }
    );
  }
}
