/**
 * GET /api/world/diff?base=<base_version_id>&head=<head_version_id>
 * Computes structural diff between two WorldStore versions.
 * Returns honest diagnostics or full WorldDiff payload.
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

export async function GET(req: Request) {
  const url = new URL(req.url);
  const base = url.searchParams.get("base");
  const head = url.searchParams.get("head");

  if (!base || !head) {
    return NextResponse.json(
      { error: "Both 'base' and 'head' version query parameters are required" },
      { status: 400 }
    );
  }

  const idRegex = /^[a-zA-Z0-9_\-\.]+$/;
  if (!idRegex.test(base) || !idRegex.test(head)) {
    return NextResponse.json(
      { error: `Invalid version id format. base='${base}', head='${head}'` },
      { status: 400 }
    );
  }

  try {
    const { stdout } = await execFileAsync(
      "python",
      [BRIDGE_SCRIPT, "diff", base, head],
      {
        timeout: 30000,
        env: {
          ...process.env,
          PYTHONPATH: process.cwd() + path.sep + "..",
        },
        cwd: path.join(process.cwd(), ".."),
        maxBuffer: 50 * 1024 * 1024,
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
      {
        error: `Diff execution failed: ${message}`,
        from_version_id: base,
        to_version_id: head,
      },
      { status: 500 }
    );
  }
}
