/**
 * GET /api/world
 * Lists all WorldStore versions available on disk.
 * Returns real entity counts and metadata from the backend.
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

export async function GET() {
  try {
    const { stdout } = await execFileAsync(
      "python",
      [BRIDGE_SCRIPT, "list-worlds"],
      {
        timeout: 30000,
        env: {
          ...process.env,
          PYTHONPATH: process.cwd() + path.sep + "..",
        },
        cwd: path.join(process.cwd(), ".."),
      }
    );

    const data = JSON.parse(stdout.trim().split("\n")[0]);
    return NextResponse.json(data);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message, worlds: [], count: 0 },
      { status: 503 }
    );
  }
}
