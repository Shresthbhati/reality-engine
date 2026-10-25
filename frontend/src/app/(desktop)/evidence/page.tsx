"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Search, ShieldCheck } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import DecisionActions from "@/components/evidence/DecisionActions";
import {
  CheckIcon,
  CheckRows,
  DECISION_NOTE,
  EXPLANATION_NOTE,
  Meta,
  Panel,
  ResultChip,
  ScoreRing,
  TONE_BG,
  TONE_COLOR,
} from "@/components/evidence/QualityParts";
import { isApiError } from "@/lib/api/client";
import { evidenceArtifactUrl } from "@/lib/api/evidence";
import { listEvidenceQuality, type EvidenceQualityList } from "@/lib/api/evidence-quality";
import {
  RESULT_LABEL,
  TABS,
  countResults,
  filterItems,
  formatBytes,
  formatDimensions,
  percentOf,
  resultOf,
  resultTone,
  type EvidenceQualityItem,
  type QualityResult,
  type QualityTab,
} from "@/lib/api/evidence-quality-model";

const COUNTERS: Array<{ key: QualityResult | "total"; label: string }> = [
  { key: "total", label: "Total files" },
  { key: "accepted", label: "Accepted" },
  { key: "accepted_with_warning", label: "Accepted with warning" },
  { key: "needs_review", label: "Needs review" },
  { key: "rejected", label: "Rejected" },
];

function fmtDate(iso: string | null): string {
  if (!iso) return "Unknown";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "Unknown" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Recompute `result` after a local decision change (a decision overrides the automatic result). */
function withDecision(i: EvidenceQualityItem, d: EvidenceQualityItem["decision"]): EvidenceQualityItem {
  return { ...i, decision: d, result: d ? d.decision : i.auto_result };
}

export default function EvidenceQualityFirewallPage() {
  const [data, setData] = useState<EvidenceQualityList | null>(null);
  const [items, setItems] = useState<EvidenceQualityItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<QualityTab>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const load = useCallback(() => {
    listEvidenceQuality()
      .then((d) => {
        setData(d);
        setItems(d.items);
      })
      .catch((e) => setError(isApiError(e) ? e.describe() : String(e)))
      .finally(() => setLoading(false));
  }, []);
  useEffect(load, [load]);

  const counts = useMemo(() => countResults(items), [items]);
  const shown = useMemo(() => filterItems(items, tab, query), [items, tab, query]);
  const selected = shown.find((i) => i.id === selectedId) ?? items.find((i) => i.id === selectedId) ?? shown[0] ?? null;

  return (
    <div className="flex flex-col h-full min-h-0">
      <PageHeader
        title="Evidence Quality Firewall"
        action={
          <Link
            href="/sessions"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md"
            style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            View Sessions
          </Link>
        }
      />

      {loading ? (
        <p className="px-6 py-4 text-sm" style={{ color: "var(--text-tertiary)" }}>Loading...</p>
      ) : error ? (
        <p className="px-6 py-4 text-sm" style={{ color: "var(--error)" }}>{error}</p>
      ) : (
        <div className="flex flex-col flex-1 min-h-0 overflow-y-auto">
          <p className="px-6 pt-4 text-sm" style={{ color: "var(--text-secondary)" }}>
            Photographs are checked from measured facts before they enter reconstruction. Scores summarise the checks; they are not
            calibrated quality measures.
          </p>

          <div className="grid gap-3 px-6 pt-4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))" }}>
            {COUNTERS.map(({ key, label }) => {
              const n = counts[key];
              const tone = key === "total" ? "neutral" : resultTone(key);
              const color = key === "needs_review" ? "var(--purple)" : key === "total" ? "var(--accent)" : TONE_COLOR[tone];
              return (
                <div
                  key={key}
                  className="rounded-md p-3"
                  style={{ background: key === "needs_review" ? "var(--purple-subtle)" : key === "total" ? "var(--accent-subtle)" : TONE_BG[tone], border: `1px solid ${color}` }}
                >
                  <div className="text-2xl font-semibold font-mono-num" style={{ color: "var(--text-primary)" }}>{n}</div>
                  <div className="text-xs" style={{ color: "var(--text-secondary)" }}>{label}</div>
                  {key !== "total" && (
                    <div className="text-xs font-mono-num" style={{ color }}>{percentOf(n, counts.total)}</div>
                  )}
                </div>
              );
            })}
          </div>
          {counts.not_inspected > 0 && (
            <p className="px-6 pt-2 text-xs" style={{ color: "var(--text-tertiary)" }}>
              {counts.not_inspected} file(s) could not be inspected (artifact missing from the store).
            </p>
          )}
          {data && data.not_screened_non_photo > 0 && (
            <p className="px-6 pt-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
              {data.not_screened_non_photo} non-photo evidence item(s) (video, point cloud, sensor data) are not screened here.
            </p>
          )}

          <div className="flex items-center gap-3 px-6 pt-4 flex-wrap">
            <div className="flex items-center gap-1 flex-wrap">
              {TABS.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTab(t)}
                  className="px-2.5 h-7 rounded text-sm font-medium"
                  style={{
                    color: tab === t ? "var(--accent)" : "var(--text-secondary)",
                    background: tab === t ? "var(--accent-subtle)" : "transparent",
                  }}
                >
                  {t === "all" ? `All (${counts.total})` : `${RESULT_LABEL[t]} (${counts[t]})`}
                </button>
              ))}
            </div>
            <div
              className="flex items-center gap-2 h-8 px-2.5 rounded-md ml-auto w-64"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
            >
              <Search className="w-3.5 h-3.5" style={{ color: "var(--text-tertiary)" }} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search files..."
                aria-label="Search files"
                className="bg-transparent outline-none text-sm w-full"
                style={{ color: "var(--text-primary)" }}
              />
            </div>
          </div>

          {items.length === 0 ? (
            <EmptyState icon={ShieldCheck} message="No photographs have been added yet." />
          ) : (
            <div className="flex gap-4 p-6 items-start flex-col xl:flex-row">
              <div className="flex-1 min-w-0 w-full overflow-x-auto rounded-md" style={{ border: "1px solid var(--border)", background: "var(--bg-surface)" }}>
                <table className="w-full text-sm" style={{ minWidth: 820 }}>
                  <thead>
                    <tr className="text-left text-xs" style={{ color: "var(--text-tertiary)", borderBottom: "1px solid var(--border)" }}>
                      {["Preview", "File name", "Source", "Added", "Checks", "Score", "Result", "Key issues"].map((h) => (
                        <th key={h} className="px-3 py-2 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {shown.length === 0 && (
                      <tr><td colSpan={8} className="px-3 py-6 text-center" style={{ color: "var(--text-tertiary)" }}>No files match this filter.</td></tr>
                    )}
                    {shown.map((i) => (
                      <tr
                        key={i.id}
                        onClick={() => setSelectedId(i.id)}
                        className="cursor-pointer"
                        style={{
                          borderBottom: "1px solid var(--border-subtle)",
                          background: selected?.id === i.id ? "var(--accent-subtle)" : undefined,
                        }}
                      >
                        <td className="px-3 py-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={evidenceArtifactUrl(i.id)}
                            alt=""
                            loading="lazy"
                            decoding="async"
                            className="w-14 h-10 object-cover rounded"
                            style={{ background: "var(--bg-elevated)" }}
                          />
                        </td>
                        <td className="px-3 py-2 max-w-[200px]">
                          <button
                            type="button"
                            onClick={() => setSelectedId(i.id)}
                            className="block truncate text-left font-medium max-w-full"
                            style={{ color: "var(--text-primary)" }}
                            title={i.name}
                          >
                            {i.name}
                          </button>
                          <div className="text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                            {formatBytes(i.size)} · {formatDimensions(i)}
                          </div>
                        </td>
                        <td className="px-3 py-2" style={{ color: "var(--text-secondary)" }}>{i.source}</td>
                        <td className="px-3 py-2 text-xs" style={{ color: "var(--text-secondary)" }}>{fmtDate(i.added_at)}</td>
                        <td className="px-3 py-2">
                          <span className="inline-flex gap-1" aria-label="Check results">
                            {i.checks.map((c) => (
                              <span key={c.name} title={`${c.name}: ${c.message}`}>
                                <CheckIcon status={c.status} />
                              </span>
                            ))}
                          </span>
                        </td>
                        <td className="px-3 py-2"><ScoreRing score={i.score} /></td>
                        <td className="px-3 py-2"><ResultChip result={resultOf(i)} /></td>
                        <td className="px-3 py-2 text-xs max-w-[220px]" style={{ color: "var(--text-secondary)" }}>
                          {i.key_issues.length ? i.key_issues.join("; ") : "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {selected && (
                <aside className="w-full xl:w-[380px] shrink-0 flex flex-col gap-3" aria-label="Selected file">
                  <div className="flex items-start justify-between gap-2">
                    <h2 className="text-sm font-semibold break-all" style={{ color: "var(--text-primary)" }}>{selected.name}</h2>
                    <ResultChip result={resultOf(selected)} />
                  </div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={evidenceArtifactUrl(selected.id)}
                    alt={selected.name}
                    className="w-full max-h-56 object-contain rounded"
                    style={{ background: "var(--bg-base)", border: "1px solid var(--border)" }}
                  />
                  <Panel title="Metadata">
                    <Meta label="Source" value={selected.source} />
                    <Meta label="Added" value={fmtDate(selected.added_at)} />
                    <Meta label="Type" value={selected.mime_type ?? "Unknown"} />
                    <Meta label="Resolution" value={formatDimensions(selected)} />
                    <Meta label="File size" value={formatBytes(selected.size)} />
                    <Meta label="Class" value={selected.evidence_class ?? "Not inspected"} />
                    <Meta label="Location" value={selected.gps ? `${selected.gps[0].toFixed(4)}, ${selected.gps[1].toFixed(4)}` : "No GPS in file"} />
                  </Panel>
                  <Panel title="Quality checks">
                    <CheckRows checks={selected.checks} />
                  </Panel>
                  <Panel title="Explanation">
                    <p className="text-sm" style={{ color: "var(--text-primary)" }}>{selected.explanation}</p>
                    <p className="text-xs mt-1" style={{ color: "var(--text-tertiary)" }}>{EXPLANATION_NOTE}</p>
                  </Panel>
                  <Panel title="Decision">
                    {selected.decision?.reason && (
                      <p className="text-sm mb-2" style={{ color: "var(--text-secondary)" }}>Reason: {selected.decision.reason}</p>
                    )}
                    <DecisionActions
                      key={selected.id}
                      evidenceId={selected.id}
                      current={selected.decision?.decision ?? null}
                      initialReason={selected.decision?.reason ?? null}
                      onSaved={(d) => setItems((all) => all.map((x) => (x.id === selected.id ? withDecision(x, d) : x)))}
                    />
                    <p className="text-xs mt-2" style={{ color: "var(--text-tertiary)" }}>{DECISION_NOTE}</p>
                  </Panel>
                  <Link
                    href={`/evidence/${selected.id}/view`}
                    className="inline-flex items-center justify-center h-9 rounded-md text-sm font-medium"
                    style={{ background: "var(--primary)", color: "white" }}
                  >
                    Open asset inspector
                  </Link>
                </aside>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
