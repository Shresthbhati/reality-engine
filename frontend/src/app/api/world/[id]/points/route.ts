/**
 * GET /api/world/[id]/points
 * Streams the real point cloud artifact (points.ply) for a specific WorldStore version.
 * Adheres strictly to the REAL-DATA FIREWALL:
 * - If points.ply exists, streams real binary PLY bytes
 * - If unavailable, returns 404 with explicit UNAVAILABLE status and diagnostic info
 */

import { NextResponse } from "next/server";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";
import fs from "fs";

const execFileAsync = promisify(execFile);

const BRIDGE_SCRIPT = path.join(
  process.cwd(),
  "..",
  "apps",
  "cli",
  "api_bridge.py"
);

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  if (!id || typeof id !== "string" || !/^[a-zA-Z0-9_\-\.]+$/.test(id)) {
    return NextResponse.json(
      { error: "Invalid version id format", status: "BLOCKED" },
      { status: 400 }
    );
  }

  try {
    const { stdout } = await execFileAsync(
      "python",
      [BRIDGE_SCRIPT, "points-path", id],
      {
        timeout: 15000,
        env: {
          ...process.env,
          PYTHONPATH: process.cwd() + path.sep + "..",
        },
        cwd: path.join(process.cwd(), ".."),
      }
    );

    const result = JSON.parse(stdout.trim().split("\n")[0]);

    if (!result.found || !result.path) {
      return NextResponse.json(
        {
          error: result.error || `Point cloud artifact unavailable for version ${id}`,
          status: "UNAVAILABLE",
          version_id: id,
        },
        { status: 404 }
      );
    }

    if (!fs.existsSync(result.path)) {
      return NextResponse.json(
        {
          error: `Point cloud file referenced at ${result.path} does not exist on disk`,
          status: "UNAVAILABLE",
          version_id: id,
        },
        { status: 404 }
      );
    }

    const buffer = await fs.promises.readFile(result.path);

    return new Response(buffer, {
      status: 200,
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": `inline; filename="${id}-points.ply"`,
        "Content-Length": buffer.length.toString(),
        "Cache-Control": "public, max-age=3600, immutable",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message, status: "ERROR", version_id: id },
      { status: 500 }
    );
  }
}
