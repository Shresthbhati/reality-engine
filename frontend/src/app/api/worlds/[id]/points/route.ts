import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const response = await fetch(`${BACKEND_URL}/api/worlds/${id}/points`, {
      signal: AbortSignal.timeout(10000),
    });
    
    if (!response.ok) {
      return NextResponse.json(
        { error: "No point cloud available for this world", world_id: id },
        { status: 404 }
      );
    }
    
    const data = await response.json();
    
    if (data.found && data.path) {
      // For local files, we need to read and serve them
      // In production, this would be served via a file server or object storage
      const fs = await import("fs");
      const path = await import("path");
      
      if (fs.existsSync(data.path)) {
        const buffer = fs.readFileSync(data.path);
        return new NextResponse(buffer, {
          headers: {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="points.ply"',
          },
        });
      }
    }
    
    return NextResponse.json(
      { error: "Point cloud file not found locally", world_id: id },
      { status: 404 }
    );
  } catch {
    return NextResponse.json(
      { error: "No point cloud available for this world", world_id: id },
      { status: 404 }
    );
  }
}