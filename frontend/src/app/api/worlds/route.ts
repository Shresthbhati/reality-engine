import { NextResponse } from "next/server";
import { WORLDS } from "@/lib/data";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET() {
  const items = [
    {
      id: "world-compiled-seed42",
      name: "Room Capture (Compiled Vertical Slice)",
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
    },
    ...WORLDS.map((w) => ({
      ...w,
      has3DData: false,
    })),
  ];

  // If live backend is reachable, try to merge its worlds
  try {
    const backendRes = await fetch(`${BACKEND_URL}/api/worlds`, {
      signal: AbortSignal.timeout(600),
    });
    if (backendRes.ok) {
      const data = await backendRes.json();
      if (data.worlds && Array.isArray(data.worlds)) {
        for (const item of data.worlds) {
          if (!items.some((w) => w.id === item.version_id)) {
            items.push({
              id: item.version_id,
              name: item.name || item.world_id,
              location: item.coordinate_frame || "Captured Location",
              lat: item.latitude || 0,
              lng: item.longitude || 0,
              coverageKm2: 0.1,
              sessionCount: item.session_count || 0,
              evidenceCount: item.entity_count || 0,
              timeRangeStart: item.created_at,
              timeRangeEnd: item.modified_at,
              updatedAt: item.created_at || "Unknown",
              has3DData: item.entity_count > 0,
            });
          }
        }
      } else if (Array.isArray(data.items)) {
        // Legacy format support
        for (const item of data.items) {
          if (!items.some((w) => w.id === item.id)) {
            items.push({
              id: item.id,
              name: item.name,
              location: item.description || "Captured Location",
              lat: item.latitude,
              lng: item.longitude,
              coverageKm2: 0.1,
              sessionCount: item.session_count || 0,
              evidenceCount: 0,
              timeRangeStart: null,
              timeRangeEnd: null,
              updatedAt: item.created_at,
              has3DData: false,
            });
          }
        }
      }
    }
  } catch {
    // Backend offline; use authoritative local dataset
  }

  return NextResponse.json({ items });
}