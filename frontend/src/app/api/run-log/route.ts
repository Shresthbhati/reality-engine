/**
 * GET /api/run-log
 *
 * Serves the newest measured reconstruction run record (datasets/<dataset>/runs/run_*.json)
 * as a log timeline. Every entry is derived from the recorded run — stages,
 * durations, outcomes, refusal reasons — never synthesized.
 */

import { NextResponse } from "next/server";
import { execFile } from "child_process";
import { promisify } from "util";
import path from "path";

const execFileAsync = promisify(execFile);

const BRIDGE_SCRIPT = path.join(process.cwd(), "..", "apps", "cli", "api_bridge.py");

export async function GET() {
  try {
    const { stdout } = await execFileAsync(
      "python",
      [BRIDGE_SCRIPT, "run-log"],
      {
        timeout: 15000,
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
      { available: false, reason: `run-log bridge failed: ${message}` },
      { status: 200 }
    );
  }
}
