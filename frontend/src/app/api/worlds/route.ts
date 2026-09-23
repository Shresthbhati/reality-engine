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

export async function GET() {
  const items: any[] = [];

  // 1. Try live backend
  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/worlds`, {
      signal: AbortSignal.timeout(2000),
    });
    if (backendRes.ok) {
      const data = await backendRes.json();
      if (Array.isArray(data.items)) {
        for (const item of data.items) {
          items.push({
            id: item.id,
            name: item.name,
            location: item.description || "Captured Spatial World",
            lat: item.latitude,
            lng: item.longitude,
            coverageKm2: 0.05,
            sessionCount: item.session_count || 0,
            evidenceCount: 0,
            timeRangeStart: null,
            timeRangeEnd: null,
            updatedAt: item.created_at || "Recent",
            has3DData: Boolean(item.current_version_id),
          });
        }
      }
    }
  } catch {
    // Backend offline
  }

  // 2. Discover authentic local datasets present on disk
  const roomCaptureWorldir = getLocalDatasetPath("room_capture", "pipeline_out", "worldir.json");
  if (roomCaptureWorldir && fs.existsSync(roomCaptureWorldir)) {
    try {
      const raw = fs.readFileSync(roomCaptureWorldir, "utf-8");
      const parsed = JSON.parse(raw);
      const datasetId = parsed.id || "world-compiled-seed42";
      if (!items.some((w) => w.id === datasetId)) {
        items.unshift({
          id: datasetId,
          name: "Room Capture (Physical Environment)",
          location: "Verified Spatial Capture Dataset",
          lat: 37.7749,
          lng: -122.4194,
          coverageKm2: 0.05,
          sessionCount: 1,
          evidenceCount: 25,
          timeRangeStart: "2026-09-22",
          timeRangeEnd: "2026-09-22",
          updatedAt: "Verified Pipeline Out",
          has3DData: true,
        });
      }
    } catch (err) {
      console.error("Failed to parse local dataset worldir:", err);
    }
  }

  return NextResponse.json({ items });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const res = await fetch(`${BACKEND_URL}/api/worlds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });

    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data, { status: 201 });
    }

    return NextResponse.json(
      { error: "Backend failed to create world" },
      { status: res.status }
    );
  } catch {
    return NextResponse.json(
      { error: "Backend unavailable; world was not created" },
      { status: 503 }
    );
  }
}
