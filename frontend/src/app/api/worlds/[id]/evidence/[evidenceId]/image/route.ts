import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

function getLocalDatasetPath(...segments: string[]) {
  const candidates = [
    path.resolve(process.cwd(), "..", "datasets", ...segments),
    path.resolve(process.cwd(), "datasets", ...segments),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string; evidenceId: string }> }
) {
  const { id, evidenceId } = await context.params;

  // 1. Try Live Backend if reachable
  try {
    const res = await fetch(`${BACKEND_URL}/api/evidence/${encodeURIComponent(evidenceId)}/file`, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const blob = await res.arrayBuffer();
      const contentType = res.headers.get("content-type") || "image/jpeg";
      return new NextResponse(blob, {
        headers: {
          "Content-Type": contentType,
          "Cache-Control": "public, max-age=3600",
        },
      });
    }
  } catch {
    // Backend offline; check authentic local dataset
  }

  // 2. Verified Local Pipeline Out Dataset (room_capture/images)
  const imageExtensions = [".jpg", ".jpeg", ".png", ".webp"];
  for (const ext of imageExtensions) {
    const filename = evidenceId.endsWith(ext) ? evidenceId : `${evidenceId}${ext}`;
    const imagePath = getLocalDatasetPath("room_capture", "images", filename);
    if (imagePath && fs.existsSync(/*turbopackIgnore: true*/ imagePath)) {
      const buffer = fs.readFileSync(/*turbopackIgnore: true*/ imagePath);
      const mime = ext === ".png" ? "image/png" : ext === ".webp" ? "image/webp" : "image/jpeg";
      return new NextResponse(buffer, {
        headers: {
          "Content-Type": mime,
          "Cache-Control": "public, max-age=86400",
        },
      });
    }
  }

  return NextResponse.json(
    { error: "Evidence image not available", evidence_id: evidenceId, world_id: id },
    { status: 404 }
  );
}
