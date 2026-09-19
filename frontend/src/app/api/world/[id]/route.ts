/**
 * GET /api/world/[id]
 * Loads a specific WorldStore version by version_id.
 * Returns real entities with provenance, confidence, geometry_ids, relationships.
 */

import { NextResponse } from "next/server";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";

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

  if (!id || typeof id !== "string") {
    return NextResponse.json({ error: "Missing version id" }, { status: 400 });
  }

  // Sanitize: alphanumeric version IDs per WorldStore (e.g. v-f145b8290bfc, v-kolkata-01, v-1, latest)
  if (!/^[a-zA-Z0-9_\-\.]+$/.test(id)) {
    return NextResponse.json(
      { error: `Invalid version id format: ${id}` },
      { status: 400 }
    );
  }

  try {
    const { stdout } = await execFileAsync(
      "python",
      [BRIDGE_SCRIPT, "load-world", id],
      {
        timeout: 30000,
        env: {
          ...process.env,
          PYTHONPATH: process.cwd() + path.sep + "..",
        },
        cwd: path.join(process.cwd(), ".."),
        maxBuffer: 50 * 1024 * 1024, // 50 MB — WorldIR can be large
      }
    );

    const data = JSON.parse(stdout.trim().split("\n")[0]);
    if (data.error) {
      return NextResponse.json(data, { status: 404 });
    }
    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message, version_id: id, entities: [] },
      { status: 503 }
    );
  }
}
