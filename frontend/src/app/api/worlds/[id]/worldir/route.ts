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
        {
          error: "No 3D WorldIR compiled for this world yet",
          world_id: id,
          available: false,
        },
        { status: 404 }
      );
    }
    
    const data = await response.json();
    
    // Extract just the worldir.json structure from the full world data
    const worldir = {
      id: data.world_id,
      schema_version: data.schema_version,
      global_provenance: data.global_provenance,
      global_confidence: data.global_confidence,
      coordinate_frame: data.coordinate_frame,
      entities: data.entities,
      geometries: data.geometries,
    };
    
    return new NextResponse(JSON.stringify(worldir), {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return NextResponse.json(
      {
        error: "No 3D WorldIR compiled for this world yet",
        world_id: id,
        available: false,
      },
      { status: 404 }
    );
  }
}