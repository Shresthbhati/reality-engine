/**
 * Search — runs against real backend entities.
 *
 * The API has no search endpoint yet, so this fetches the real collections
 * (Sessions, Worlds, Evidence) and filters them locally. Results are therefore
 * always real records; the only limitation is that filtering happens client
 * side over the most recent page of each collection. Analysis/Result/Report
 * contribute nothing because no backend service serves them yet.
 */
import { listEvidence, listSessions, listWorlds } from "./api";

export type SearchResultType = "SESSION" | "WORLD" | "EVIDENCE" | "ANALYSIS" | "RESULT" | "REPORT";

export interface SearchResult {
  type: SearchResultType;
  id: string;
  title: string;
  subtitle: string | null;
  href: string;
}

export async function searchAll(query: string): Promise<SearchResult[]> {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  const results: SearchResult[] = [];

  const [sessions, worlds, evidence] = await Promise.all([
    listSessions().catch(() => ({ rows: [], dtos: [] })),
    listWorlds().catch(() => ({ rows: [], dtos: [] })),
    listEvidence().catch(() => ({ rows: [], dtos: [] })),
  ]);

  for (const s of sessions.rows) {
    if (!s.name.toLowerCase().includes(q)) continue;
    results.push({
      type: "SESSION",
      id: s.id,
      title: s.name,
      subtitle: [s.location, s.capturedAt].filter(Boolean).join(" · ") || null,
      href: `/sessions/${s.id}`,
    });
  }

  for (const w of worlds.rows) {
    if (!w.name.toLowerCase().includes(q)) continue;
    results.push({
      type: "WORLD",
      id: w.id,
      title: w.name,
      subtitle: [w.location, `${w.sessionCount} Sessions`].filter(Boolean).join(" · "),
      href: `/worlds/${w.id}`,
    });
  }

  for (const e of evidence.rows) {
    if (!e.name.toLowerCase().includes(q)) continue;
    results.push({
      type: "EVIDENCE",
      id: e.id,
      title: e.name,
      subtitle: [e.location, e.sessionName].filter(Boolean).join(" · ") || null,
      href: `/evidence/${e.id}`,
    });
  }

  return results;
}
