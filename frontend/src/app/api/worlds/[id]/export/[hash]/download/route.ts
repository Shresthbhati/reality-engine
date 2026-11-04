import { NextRequest, NextResponse } from "next/server";

import { BACKEND_URL, backendFetch } from "../../../../../../../lib/api/backend.ts";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string; hash: string }> }
) {
  const { id, hash } = await params;
  try {
    const format = request.nextUrl.searchParams.get("format") || "bin";
    const backendRes = await backendFetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/export/${encodeURIComponent(hash)}/download?format=${encodeURIComponent(format)}`,
      {
        method: "GET",
        signal: AbortSignal.timeout(60000),
      }
    );

    if (!backendRes.ok) {
      return NextResponse.json(
        { detail: `Download failed with HTTP ${backendRes.status}` },
        { status: backendRes.status }
      );
    }

    const contentType = backendRes.headers.get("Content-Type") || "application/octet-stream";
    const contentDisposition = backendRes.headers.get("Content-Disposition") || `attachment; filename="${id}.${format}"`;
    const buffer = await backendRes.arrayBuffer();

    return new NextResponse(buffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": contentDisposition,
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { detail: `Failed to download export artifact: ${message}`, available: false },
      { status: 503 }
    );
  }
}
