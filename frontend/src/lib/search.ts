import { SESSIONS, WORLDS, EVIDENCE, ANALYSIS, RESULTS, REPORTS } from "./data";

export type SearchResultType = "SESSION" | "WORLD" | "EVIDENCE" | "ANALYSIS" | "RESULT" | "REPORT";

export interface SearchResult {
  type: SearchResultType;
  id: string;
  title: string;
  subtitle: string | null;
  href: string;
}

/**
 * Searches across every real entity source. Reads directly from lib/data.ts —
 * no separate index to keep in sync, no fabricated results.
 */
export function searchAll(query: string): SearchResult[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  const results: SearchResult[] = [];

  for (const s of SESSIONS) {
    if (s.name.toLowerCase().includes(q)) {
      results.push({
        type: "SESSION",
        id: s.id,
        title: s.name,
        subtitle: [s.location, s.capturedAt].filter(Boolean).join(" · ") || null,
        href: `/sessions/${s.id}`,
      });
    }
  }

  for (const w of WORLDS) {
    if (w.name.toLowerCase().includes(q)) {
      results.push({
        type: "WORLD",
        id: w.id,
        title: w.name,
        subtitle: [w.location, w.coverageKm2 != null ? `${w.coverageKm2} km²` : null].filter(Boolean).join(" · ") || null,
        href: `/worlds/${w.id}`,
      });
    }
  }

  for (const e of EVIDENCE) {
    if (e.name.toLowerCase().includes(q)) {
      results.push({
        type: "EVIDENCE",
        id: e.id,
        title: e.name,
        subtitle: [e.location, e.sessionName].filter(Boolean).join(" · ") || null,
        href: `/evidence/${e.id}`,
      });
    }
  }

  for (const a of ANALYSIS) {
    if (a.name.toLowerCase().includes(q)) {
      results.push({
        type: "ANALYSIS",
        id: a.id,
        title: a.name,
        subtitle: a.sessionName,
        href: `/analysis/${a.id}`,
      });
    }
  }

  for (const r of RESULTS) {
    if (r.title.toLowerCase().includes(q)) {
      results.push({
        type: "RESULT",
        id: r.id,
        title: r.title,
        subtitle: [r.location, r.sessionName].filter(Boolean).join(" · ") || null,
        href: `/results/${r.id}`,
      });
    }
  }

  for (const rep of REPORTS) {
    if (rep.title.toLowerCase().includes(q)) {
      results.push({
        type: "REPORT",
        id: rep.id,
        title: rep.title,
        subtitle: rep.generatedAt,
        href: `/reports/${rep.id}`,
      });
    }
  }

  return results;
}
