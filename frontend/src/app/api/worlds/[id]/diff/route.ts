import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;
  const { searchParams } = new URL(request.url);
  const baseVersion = searchParams.get("base") || searchParams.get("base_version");
  const headVersion = searchParams.get("head") || searchParams.get("head_version") || "latest";

  // Try live backend if available
  try {
    const q = new URLSearchParams();
    if (baseVersion) q.set("base_version", baseVersion);
    if (headVersion) q.set("head_version", headVersion);

    const res = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/diff?${q.toString()}`,
      { signal: AbortSignal.timeout(2000) }
    );
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend offline
  }

  // Zero mock guarantee: return honest unavailable status when no diff exists
  return NextResponse.json(
    {
      error: "Version comparison unavailable. The backend bridge has not registered diff artifacts between these versions.",
      world_id: id,
      base_version_id: baseVersion || "unspecified",
      head_version_id: headVersion,
      available: false,
    },
    { status: 404 }
  );
}
