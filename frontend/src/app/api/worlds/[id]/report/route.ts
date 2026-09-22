import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${id}`, {
      signal: AbortSignal.timeout(5000),
    });
    
    if (!response.ok) {
      return NextResponse.json(
        { error: "No pipeline report available for this world", world_id: id },
        { status: 404 }
      );
    }
    
    const data = await response.json();
    
    // Build a report from the world data
    const report = {
      version_id: data.version_id,
      world_id: data.world_id,
      name: data.name,
      schema_version: data.schema_version,
      global_provenance: data.global_provenance,
      global_confidence: data.global_confidence,
      coordinate_frame: data.coordinate_frame,
      entity_count: data.entity_count,
      geometry_count: data.geometry_count,
      entities: data.entities?.map((e: any) => ({
        id: e.id,
        type: e.type,
        provenance: e.provenance,
        confidence: e.confidence,
        geometry_ids: e.geometry_ids,
      })),
      geometries: Object.entries(data.geometries || {}).map(([gid, g]: [string, any]) => ({
        id: gid,
        type: g.type,
        lod_level: g.lod_level,
        vertex_count: g.vertex_count,
        triangle_count: g.triangle_count,
        data_uri: g.data_uri,
        data_hash: g.data_hash,
      })),
    };
    
    return NextResponse.json(report);
  } catch {
    return NextResponse.json(
      { error: "No pipeline report available for this world", world_id: id },
      { status: 404 }
    );
  }
}