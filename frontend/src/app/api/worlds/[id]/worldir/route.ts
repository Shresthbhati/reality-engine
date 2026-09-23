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

  // 1. Try Live Backend computational surface (WorldStore is authoritative)
  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/worldir`, {
      signal: AbortSignal.timeout(2500),
    });

    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    } else if (response.status === 404) {
      // World exists in DB but has no compiled reconstruction version yet
      return NextResponse.json(
        {
          error: "World has no reconstruction versions yet — run reconstruction first",
          world_id: id,
          available: false,
        },
        { status: 404 }
      );
    }
  } catch {
    // Backend offline; fall through to authentic local dataset if available
  }

  // 2. Verified Local Pipeline Out Dataset (datasets/room_capture/pipeline_out)
  const isLocalDataset =
    id === "world-compiled-seed42" ||
    id === "room-capture" ||
    id === "dataset-room-capture" ||
    id.startsWith("world-room");

  if (isLocalDataset) {
    const localWorldirPath = getLocalDatasetPath("room_capture", "pipeline_out", "worldir.json");
    if (localWorldirPath && fs.existsSync(localWorldirPath)) {
      try {
        const raw = fs.readFileSync(localWorldirPath, "utf-8");
        const json = JSON.parse(raw);
        return NextResponse.json(json);
      } catch (err) {
        console.error("Failed to read local worldir:", err);
      }
    }
  }

  return NextResponse.json(
    {
      error: "No 3D WorldIR compiled for this world yet",
      world_id: id,
      available: false,
    },
    { status: 404 }
  );
}
