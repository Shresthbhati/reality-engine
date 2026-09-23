import { NextRequest, NextResponse } from "next/server";
import { addStoredVersion } from "../versions/route";

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const { id } = await context.params;

  try {
    const body = await request.json();
    const {
      entityId,
      changes,
      parentVersionId,
      commitMessage,
    } = body;

    const versionNum = Date.now().toString().slice(-4);
    const newVersionId = `v2-${versionNum}`;

    const newVersion = {
      id: newVersionId,
      world_id: id,
      label: `v2.${versionNum} (${commitMessage ? commitMessage.slice(0, 24) : "Correction"})`,
      parent_version_id: parentVersionId || "v1-canonical-seed42",
      artifact_uri: `artifact://${id}/${newVersionId}`,
      artifact_hash: `sha256:commit_${versionNum}`,
      source_session_ids: ["session-room-capture", "review-workflow"],
      changed_entity_ids: entityId ? [entityId] : [],
      changed_geometry_ids: [],
      created_at: new Date().toISOString(),
      is_current: true,
      changeSummary: commitMessage || `Corrected entity ${entityId}: ${JSON.stringify(changes)}`,
    };

    addStoredVersion(id, newVersion);

    return NextResponse.json({
      success: true,
      version: newVersion,
    });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "Failed to commit world version" },
      { status: 500 }
    );
  }
}
