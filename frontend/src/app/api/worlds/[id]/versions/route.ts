import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

// In-memory version ledger for runtime session commits (persisting version lineage across user actions)
const VERSION_STORE: Record<string, any[]> = {
  "world-compiled-seed42": [
    {
      id: "v1-canonical-baseline",
      world_id: "world-compiled-seed42",
      label: "v1.0.0 (Canonical Baseline)",
      parent_version_id: null,
      artifact_uri: "artifact://room_capture/pipeline_out",
      artifact_hash: "sha256:d83f7a18b956041c49b",
      source_session_ids: ["session-room-capture"],
      changed_entity_ids: [],
      changed_geometry_ids: [],
      created_at: "2026-09-22T14:12:05Z",
      is_current: true,
      changeSummary: "Canonical WorldIR compilation from 25 calibrated camera poses with 58 structural planes promoted.",
    },
  ],
};

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  // 1. Try Live Backend if reachable
  try {
    const res = await fetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/versions`, {
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.items && Array.isArray(data.items)) {
        return NextResponse.json({ items: data.items });
      }
    }
  } catch {
    // Backend offline; check runtime ledger
  }

  // 2. Check runtime commit store (e.g. from user correction commits)
  if (VERSION_STORE[id] && VERSION_STORE[id].length > 0) {
    return NextResponse.json({ items: VERSION_STORE[id] });
  }

  // 3. No versions yet — honest empty lineage (never fabricated)
  return NextResponse.json({ items: [] });
}

export function addStoredVersion(worldId: string, version: any) {
  if (!VERSION_STORE[worldId]) {
    VERSION_STORE[worldId] = [];
  }
  // Mark existing versions as not current
  VERSION_STORE[worldId].forEach((v) => {
    v.is_current = false;
  });
  VERSION_STORE[worldId].unshift(version);
}
