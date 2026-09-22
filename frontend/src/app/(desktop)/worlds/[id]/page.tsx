import { notFound } from "next/navigation";
import {
  getWorld,
  isApiError,
  listEvidence,
  listSessions,
  type EvidenceRow,
  type SessionRow,
  type WorldRow,
} from "@/lib/api";
import WorldWorkspaceClient from "./WorldWorkspaceClient";

/**
 * World workspace: real application state for this World.
 *
 * Sessions and Evidence come from the API. Places, Versions, Results and
 * Analysis have no backend service yet, so they are passed as empty
 * collections and the workspace states that honestly rather than filling them
 * with sample rows.
 */
export default async function WorldWorkspacePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let world: WorldRow;
  try {
    world = (await getWorld(id)).row;
  } catch (err) {
    if (isApiError(err) && err.code === "not_found") notFound();
    throw err;
  }

  let sessions: SessionRow[] = [];
  let evidence: EvidenceRow[] = [];
  let partial = false;
  try {
    const s = await listSessions({ worldId: id });
    sessions = s.rows;
    const sessionIds = new Set(sessions.map((row) => row.id));
    // Evidence belongs to this World through its Session; the API filters by
    // session, so this is a real join, not a guess.
    const perSession = await Promise.all(
      sessions.map((row) => listEvidence({ sessionId: row.id }).then((r) => r.rows)),
    );
    evidence = perSession.flat().filter((e) => e.sessionId !== null && sessionIds.has(e.sessionId));
  } catch {
    partial = true;
  }

  return (
    <WorldWorkspaceClient
      world={world}
      sessions={sessions}
      evidence={evidence}
      places={[]}
      versions={[]}
      results={[]}
      analysis={[]}
      partial={partial}
    />
  );
}


