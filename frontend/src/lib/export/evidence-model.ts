/** Pure helpers for the export evidence report (T-2.7): what was measured vs inferred, per element. */

export interface ExportElement {
  id: string;
  name: string;
  type: string;
  state: string;
  confidence: number | null;
  note: string | null;
}

export interface EvidenceSummary {
  total: number;
  by_state: Record<string, number>;
}

const LABEL: Record<string, string> = { measured: "Measured", inferred: "Inferred", estimated: "Estimated" };
/** weakest evidence first: estimated, then inferred, then unknown, then measured */
const WEAKNESS: Record<string, number> = { estimated: 0, inferred: 1, unknown: 2, measured: 3 };

export function stateLabel(state: string): string {
  return LABEL[state] ?? "Unknown";
}

export function evidenceHeadline(s: EvidenceSummary): string {
  if (s.total === 0) return "No elements were exported";
  const measured = s.by_state.measured ?? 0;
  const parts = ["measured", "estimated", "inferred", "unknown"]
    .filter((k) => (s.by_state[k] ?? 0) > 0)
    .map((k) => `${s.by_state[k]} ${k}`);
  return `${s.total} elements: ${parts.join(", ")}${measured === 0 ? ", none measured" : ""}`;
}

export function orderedElements(rows: ExportElement[]): ExportElement[] {
  return [...rows].sort(
    (a, b) => (WEAKNESS[a.state] ?? 2) - (WEAKNESS[b.state] ?? 2) || a.name.localeCompare(b.name),
  );
}
