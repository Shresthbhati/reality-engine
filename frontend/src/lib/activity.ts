import { SESSIONS, WORLDS, EVIDENCE, ANALYSIS, RESULTS, REPORTS } from "./data";

export type ActivityType = "SESSION" | "WORLD" | "EVIDENCE" | "ANALYSIS" | "RESULT" | "REPORT";

export interface ActivityEvent {
  type: ActivityType;
  id: string;
  label: string;
  timestamp: string;
  href: string;
  /** Parsed from `timestamp` for sorting; null when the string can't be parsed. */
  at: Date | null;
}

/**
 * Every timestamp string in this codebase follows one consistent format —
 * "D MMM YYYY · HH:mm" (spec §83) — because we write it ourselves. Parsing
 * that one known format is reliable; parsing arbitrary external date strings
 * would not be, so this parser is intentionally narrow.
 */
const MONTHS: Record<string, number> = {
  jan: 0, feb: 1, mar: 2, apr: 3, may: 4, jun: 5,
  jul: 6, aug: 7, sep: 8, oct: 9, nov: 10, dec: 11,
};

function parseTimestamp(value: string | null): Date | null {
  if (!value) return null;
  const match = value.match(/(\d{1,2})\s+([A-Za-z]{3})\w*\s+(\d{4})(?:\s*·\s*(\d{1,2}):(\d{2}))?/);
  if (!match) return null;
  const [, day, monAbbr, year, hour, minute] = match;
  const month = MONTHS[monAbbr.toLowerCase()];
  if (month === undefined) return null;
  return new Date(Number(year), month, Number(day), hour ? Number(hour) : 0, minute ? Number(minute) : 0);
}

/**
 * Aggregates real timestamped events across every entity. Each entity
 * contributes only the events its actual data supports — never a fabricated
 * "created"/"updated" pair when only one real timestamp exists.
 */
export function getActivity(): ActivityEvent[] {
  const events: ActivityEvent[] = [];

  for (const s of SESSIONS) {
    if (s.capturedAt) {
      events.push({ type: "SESSION", id: s.id, label: `${s.name} captured`, timestamp: s.capturedAt, href: `/sessions/${s.id}`, at: parseTimestamp(s.capturedAt) });
    }
  }

  for (const w of WORLDS) {
    if (w.updatedAt) {
      events.push({ type: "WORLD", id: w.id, label: `${w.name} updated`, timestamp: w.updatedAt, href: `/worlds/${w.id}`, at: parseTimestamp(w.updatedAt) });
    }
  }

  for (const e of EVIDENCE) {
    const ts = e.processedAt ?? e.uploadedAt ?? e.capturedAt;
    if (ts) {
      const verb = e.processedAt ? "processed" : e.uploadedAt ? "uploaded" : "captured";
      events.push({ type: "EVIDENCE", id: e.id, label: `${e.name} ${verb}`, timestamp: ts, href: `/evidence/${e.id}`, at: parseTimestamp(ts) });
    }
  }

  for (const a of ANALYSIS) {
    const ts = a.completedAt ?? a.startedAt;
    if (ts) {
      events.push({ type: "ANALYSIS", id: a.id, label: `${a.name} ${a.completedAt ? "completed" : "started"}`, timestamp: ts, href: `/analysis/${a.id}`, at: parseTimestamp(ts) });
    }
  }

  for (const r of RESULTS) {
    if (r.generatedAt) {
      events.push({ type: "RESULT", id: r.id, label: `${r.title} generated`, timestamp: r.generatedAt, href: `/results/${r.id}`, at: parseTimestamp(r.generatedAt) });
    }
  }

  for (const rep of REPORTS) {
    if (rep.generatedAt) {
      events.push({ type: "REPORT", id: rep.id, label: `${rep.title} generated`, timestamp: rep.generatedAt, href: `/reports/${rep.id}`, at: parseTimestamp(rep.generatedAt) });
    }
  }

  return events.sort((a, b) => {
    if (a.at && b.at) return b.at.getTime() - a.at.getTime();
    if (a.at) return -1;
    if (b.at) return 1;
    return 0;
  });
}
