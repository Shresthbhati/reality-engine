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

  // 1. Try Live Backend computational surface (returns raw PLY stream)
  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/points`, {
      signal: AbortSignal.timeout(2500),
    });

    if (response.ok) {
      const buffer = await response.arrayBuffer();
      return new NextResponse(buffer, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${id}_points.ply"`,
        },
      });
    }
  } catch {
    // Backend offline; fall through to authentic local dataset
  }

  // 2. Verified Local Pipeline Out Dataset (datasets/room_capture/pipeline_out)
  const isLocalDataset =
    id === "world-compiled-seed42" ||
    id === "room-capture" ||
    id === "dataset-room-capture" ||
    id.startsWith("world-room");

  if (isLocalDataset) {
    const localPlyPath = getLocalDatasetPath("room_capture", "pipeline_out", "points.ply");
    if (localPlyPath && fs.existsSync(/*turbopackIgnore: true*/ localPlyPath)) {
      const buffer = fs.readFileSync(/*turbopackIgnore: true*/ localPlyPath);
      return new NextResponse(buffer, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": 'attachment; filename="points.ply"',
        },
      });
    }
  }

  return NextResponse.json(
    {
      error: "No point cloud reconstructed for this world",
      world_id: id,
      available: false,
    },
    { status: 404 }
  );
}
