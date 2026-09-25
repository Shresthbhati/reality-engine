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
    if (fs.existsSync(/*turbopackIgnore: true*/ c)) return c;
  }
  return null;
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string; entityId: string }> }
) {
  const { id, entityId } = await context.params;

  // 1. Try Live Backend daemon if reachable
  try {
    const res = await fetch(
      `${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/entities/${encodeURIComponent(entityId)}/provenance`,
      { signal: AbortSignal.timeout(2000) }
    );
    if (res.ok) {
      const data = await res.json();
      return NextResponse.json(data);
    }
  } catch {
    // Backend offline; resolve from local dataset / WorldIR observations
  }

  // 2. Check local dataset WorldIR
  const isLocalDataset =
    id === "world-compiled-seed42" ||
    id === "room-capture" ||
    id === "dataset-room-capture" ||
    id.startsWith("world-room");

  if (isLocalDataset) {
    const localWorldPath = getLocalDatasetPath("room_capture", "pipeline_out", "world_ir.json");
    if (localWorldPath && fs.existsSync(/*turbopackIgnore: true*/ localWorldPath)) {
      try {
        const raw = fs.readFileSync(/*turbopackIgnore: true*/ localWorldPath, "utf-8");
        const world = JSON.parse(raw);
        const entity = world.entities?.[entityId];

        if (!entity) {
          return NextResponse.json(
            { error: `Entity '${entityId}' not found in world '${id}'` },
            { status: 404 }
          );
        }

        // Check if entity has direct observations
        const observations = entity.observations || [];
        if (observations.length > 0) {
          return NextResponse.json({
            entity_id: entityId,
            version_id: "v1-canonical-baseline",
            provenance: entity.provenance || "RECONSTRUCTED",
            trace_level: "observation",
            evidence: observations.map((obs: { id?: string; data_hash?: string; data_uri?: string }) => ({
              observation_id: obs.id || "obs-direct",
              evidence_id: obs.id || "IMG_0000",
              evidence_name: `${obs.id || "frame_0000"}.jpg`,
              evidence_type: "IMAGE",
            })),
          });
        }

        // Check session-level evidence
        const depthViews = world.metadata?.depth?.per_view || [];
        if (depthViews.length > 0) {
          return NextResponse.json({
            entity_id: entityId,
            version_id: "v1-canonical-baseline",
            provenance: entity.provenance || "RECONSTRUCTED",
            trace_level: "session",
            source_session_ids: ["session-room-capture"],
            evidence: depthViews.slice(0, 8).map((dv: { evidence_id: string }) => ({
              evidence_id: dv.evidence_id,
              evidence_name: `${dv.evidence_id}.jpg`,
              evidence_type: "IMAGE",
              session_id: "session-room-capture",
            })),
          });
        }

        return NextResponse.json({
          entity_id: entityId,
          version_id: "v1-canonical-baseline",
          provenance: entity.provenance || "INFERRED",
          trace_level: "none",
          evidence: [],
          reason: "Entity has no direct observation rays; inferred from surrounding planar boundaries.",
        });
      } catch (err) {
        console.error("Failed to read local WorldIR for provenance:", err);
      }
    }
  }

  // 3. Entity not found or world uncompiled
  return NextResponse.json(
    {
      entity_id: entityId,
      version_id: "uncompiled",
      provenance: null,
      trace_level: "none",
      evidence: [],
      reason: "No compiled version exists for this world.",
    },
    { status: 404 }
  );
}
