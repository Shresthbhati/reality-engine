import { NextRequest, NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${id}/report`, {
      signal: AbortSignal.timeout(10000),
    });
    const data = await response.json();
    return NextResponse.json(
      response.ok ? data : { ...data, world_id: id },
      { status: response.status }
    );
  } catch {
    // Backend offline; check local verified dataset report
    const isLocalDataset =
      id === "world-compiled-seed42" ||
      id === "room-capture" ||
      id === "dataset-room-capture" ||
      id.startsWith("world-room");

    if (isLocalDataset) {
      const candidates = [
        path.resolve(process.cwd(), "..", "datasets", "room_capture", "pipeline_out", "report.json"),
        path.resolve(process.cwd(), "datasets", "room_capture", "pipeline_out", "report.json"),
      ];
      for (const c of candidates) {
        if (fs.existsSync(/*turbopackIgnore: true*/ c)) {
          try {
            const raw = fs.readFileSync(/*turbopackIgnore: true*/ c, "utf-8");
            return NextResponse.json(JSON.parse(raw));
          } catch (e) {
            console.error("Failed to read report", e);
          }
        }
      }
    }

    return NextResponse.json(
      { error: "No pipeline report generated for this world yet", world_id: id },
      { status: 404 }
    );
  }
}
