import { NextRequest, NextResponse } from "next/server";
import { addStoredVersion } from "../versions/route";

const BACKEND_URL = process.env.REALITY_BACKEND_URL || "http://localhost:8100";

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const body = await request.json();
    const {
      entity_id,
      entityId,
      changes,
      parent_version_id,
      parentVersionId,
      commit_message,
      commitMessage,
    } = body;

    const targetEntityId = entity_id || entityId;
    const targetParentId = parent_version_id || parentVersionId || null;
    const targetMessage = commit_message || commitMessage || `Update ${targetEntityId}`;

    // Try backend if running
    try {
      const res = await fetch(`${BACKEND_URL}/api/worlds/${encodeURIComponent(id)}/commit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entity_id: targetEntityId,
          changes,
          parent_version_id: targetParentId,
          commit_message: targetMessage,
        }),
        signal: AbortSignal.timeout(2000),
      });
      if (res.ok) {
        const data = await res.json();
        return NextResponse.json(data);
      }
    } catch {
      // Backend offline; persist in runtime WorldStore ledger
    }

    const versionNum = Date.now().toString().slice(-4);
    const newVersionId = `v2-${versionNum}`;

    const newVersion = {
      id: newVersionId,
      world_id: id,
      label: `v2.${versionNum} (${targetMessage.slice(0, 24)})`,
      parent_version_id: targetParentId,
      artifact_uri: `artifact://${id}/${newVersionId}`,
      artifact_hash: `sha256:commit_${versionNum}`,
      source_session_ids: ["review-workflow"],
      changed_entity_ids: targetEntityId ? [targetEntityId] : [],
      changed_geometry_ids: [],
      created_at: new Date().toISOString(),
      is_current: true,
      changeSummary: targetMessage,
    };

    addStoredVersion(id, newVersion);

    return NextResponse.json({
      success: true,
      version: newVersion,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json(
      { error: message || "Failed to commit world version" },
      { status: 500 }
    );
  }
}
