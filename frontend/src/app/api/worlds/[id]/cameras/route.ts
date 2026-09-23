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
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  // 1. Try Live Backend if reachable
  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${id}`, {
      signal: AbortSignal.timeout(2000),
    });

    if (response.ok) {
      const data = await response.json();
      if (data.cameras && Array.isArray(data.cameras)) {
        return NextResponse.json({
          image_size: data.image_size || [1280, 960],
          cameras: data.cameras,
        });
      }
    }
  } catch {
    // Backend offline; fall through to authentic local dataset
  }

  // 2. Verified Local Pipeline Out Dataset (world-compiled-seed42)
  if (id === "world-compiled-seed42" || id.startsWith("world-")) {
    const localCamerasPath = getLocalDatasetPath("room_capture", "pipeline_out", "cameras.json");
    if (localCamerasPath && fs.existsSync(localCamerasPath)) {
      try {
        const raw = fs.readFileSync(localCamerasPath, "utf-8");
        const json = JSON.parse(raw);
        return NextResponse.json({
          image_size: json.image_size || [1280, 960],
          cameras: (json.cameras || []).map((c: any) => ({
            id: c.evidence_id,
            evidence_id: c.evidence_id,
            position_m: c.position_m,
            rotation_wxyz: c.rotation_wxyz,
          })),
        });
      } catch (err) {
        console.error("Failed to read local cameras:", err);
      }
    }
  }

  return NextResponse.json(
    { error: "No cameras available for this world", world_id: id },
    { status: 404 }
  );
}