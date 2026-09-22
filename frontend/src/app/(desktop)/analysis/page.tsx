"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Plus, Search, RefreshCw, TriangleAlert, Workflow } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import StatusBadge from "@/components/ui/StatusBadge";
import type { ProcessingState } from "@/lib/types";
import { getAnalysis, isApiError, type AnalysisRow } from "@/lib/api";


const FILTERS: Array<{ id: ProcessingState | "ALL"; label: string }> = [
  { id: "ALL", label: "All" },
  { id: "PROCESSING", label: "Running" },
  { id: "COMPLETE", label: "Completed" },
  { id: "FAILED", label: "Failed" },
];

export default function AnalysisPage() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["id"]>("ALL");
  const [query, setQuery] = useState("");

      const [rows, setRows] = useState<AnalysisRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getAnalysis()
      .then((d) => setRows(d.items as AnalysisRow[]))
      .catch((e) => setError(isApiError(e) ? e.describe() : String(e)))
      .finally(() => setLoading(false));
  }, []);

  const filtered = rows.filter(
    (a) => filter === "ALL" || a.state === filter,
  ).filter((a) => !query || a.name.toLowerCase().includes(query.toLowerCase()));

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Analysis"
        action={
          <Link
            href="/analysis/new"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Create Analysis
          </Link>
        }
      />

      <div className="flex items-center gap-4 px-6 h-12 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilter(f.id)}
              className="px-2.5 h-7 rounded text-sm font-medium transition-colors"
              style={{
                color: filter === f.id ? "var(--accent)" : "var(--text-secondary)",
                background: filter === f.id ? "var(--accent-subtle)" : "transparent",
              }}
            >
              {f.label}
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
            placeholder="Search Analysis…"
            className="bg-transparent outline-none text-sm w-full"
            style={{ color: "var(--text-primary)" }}
          />
        </div>
      </div>

      {loading ? (
        <p className="px-6 py-4 text-sm" style={{ color: "var(--text-tertiary)" }}>Loading�</p>
      ) : error ? (
        <p className="px-6 py-4 text-sm" style={{ color: "var(--error)" }}>{error}</p>
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={Workflow}
          message="No Analysis has been run yet. Analysis needs a Session or Evidence to work from."
          actionLabel="View Sessions"
          actionHref="/sessions"
        />
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b" style={{ borderColor: "var(--border)" }}>
              {["Analysis", "State", "Session", "Started", "Stage"].map((h) => (
                <th key={h} className="px-6 py-2 font-medium" style={{ color: "var(--text-tertiary)" }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((a) => (
              <tr key={a.id} className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <td className="px-6 py-2.5">
                  <Link href={`/analysis/${a.id}`} className="font-medium hover:underline" style={{ color: "var(--text-primary)" }}>
                    {a.name}
                  </Link>
                </td>
                <td className="px-6 py-2.5">
                  <StatusBadge state={a.state} />
                </td>
                <td className="px-6 py-2.5" style={{ color: "var(--text-secondary)" }}>
                  {a.sessionName ?? "No session"}
                </td>
                <td className="px-6 py-2.5 font-mono-num" style={{ color: "var(--text-secondary)" }}>
                  {a.startedAt ?? "—"}
                </td>
                <td className="px-6 py-2.5" style={{ color: "var(--text-secondary)" }}>
                  {a.state === "PROCESSING" && a.currentStage ? a.currentStage : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
