/**
 * GET /api/status
 * Checks whether the Reality Engine Python backend is reachable.
 * Calls apps/cli/api_bridge.py status via child_process.
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
      [BRIDGE_SCRIPT, "status"],
      {
        timeout: 8000,
        env: {
          ...process.env,
          PYTHONPATH: process.cwd() + path.sep + "..",
        },
        cwd: path.join(process.cwd(), ".."),
      }
    );

    const data = JSON.parse(stdout.trim().split("\n")[0]);
    return NextResponse.json(data, {
      status: data.backend ? 200 : 503,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      {
        backend: false,
        error: message,
        store_path: process.env.REALITY_STORE_PATH ?? "~/.reality_engine/store",
      },
      { status: 503 }
    );
  }
}
