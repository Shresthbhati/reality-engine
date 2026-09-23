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
        { error: "No cameras available for this world", world_id: id },
        { status: 404 }
      );
    }
    
    const data = await response.json();
    
    // Extract camera data from the world - look for camera entities/geometries
    const cameras = data.entities
      ?.filter((e: any) => e.type === "camera" || e.type === "camera_pose")
      .map((e: any) => ({
        id: e.id,
        position: e.custom_properties?.position || [0, 0, 0],
        rotation: e.custom_properties?.rotation || [0, 0, 0, 1],
        intrinsics: e.custom_properties?.intrinsics,
        frame_id: e.custom_properties?.frame_id,
      })) || [];
    
    return NextResponse.json({
      cameras,
      coordinate_frame: data.coordinate_frame,
    });
  } catch {
    return NextResponse.json(
      { error: "No cameras available for this world", world_id: id },
      { status: 404 }
    );
  }
}