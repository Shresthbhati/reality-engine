"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  FileText,
  Plus,
  RefreshCw,
  Workflow,
  BarChart3,
  ClipboardList,
} from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import StatusBadge from "@/components/ui/StatusBadge";
import type { ReportStatus } from "@/lib/types";
import { REPORTS } from "@/lib/data";
import { getResults, type ResultRow } from "@/lib/api/results";
import { getAnalysis, type AnalysisRow } from "@/lib/api/analysis";
import { isApiError } from "@/lib/api/client";

const REPORT_FILTERS: Array<{ id: ReportStatus | "ALL"; label: string }> = [
  { id: "ALL", label: "All" },
  { id: "DRAFT", label: "Draft" },
  { id: "COMPLETED", label: "Completed" },
];

const TABS: Array<{ id: string; label: string; icon: React.ElementType }> = [
  { id: "reports", label: "Reports", icon: ClipboardList },
  { id: "results", label: "Results", icon: BarChart3 },
  { id: "analysis", label: "Analysis", icon: Workflow },
];

export default function ReportsPage() {
  const [tab, setTab] = useState<TABS[number]["id"]>("reports");
  const [reportFilter, setReportFilter] = useState<ReportStatus | "ALL">("ALL");
  const [results, setResults] = useState<ResultRow[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisRow[]>([]);
  const [loading, setLoading] = useState({ reports: false, results: false, analysis: false });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadResults();
    void loadAnalysis();
  }, []);

  const loadResults = async () => {
    setLoading((p) => ({ ...p, results: true }));
    try {
      const data = await getResults();
      setResults(data.items);
    } catch (err) {
      setError(isApiError(err) ? err.describe() : String(err));
    } finally {
      setLoading((p) => ({ ...p, results: false }));
    }
  };

  const loadAnalysis = async () => {
    setLoading((p) => ({ ...p, analysis: true }));
    try {
      const data = await getAnalysis();
      setAnalysis(data.items);
    } catch (err) {
      setError(isApiError(err) ? err.describe() : String(err));
    } finally {
      setLoading((p) => ({ ...p, analysis: false }));
    }
  };

  const reports = REPORTS.filter((r) => reportFilter === "ALL" || r.status === reportFilter);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Reports"
        action={
          <Link
            href="/reports/new"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Create Report
          </Link>
        }
      />

      {/* Unified tabs: Reports / Results / Analysis */}
      <div className="flex border-b border-zinc-800 shrink-0" style={{ borderColor: "var(--border)" }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-2 border-b-2 px-4 py-2.5 text-xs font-semibold transition-colors ${tab === t.id ? "border-sky-500 text-sky-400" : "border-transparent text-zinc-400 hover:text-zinc-200"}`}
          >
            <t.icon className="w-4 h-4" />
            <span>{t.label}</span>
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto p-6">
        {tab === "reports" && <ReportsTab filter={reportFilter} />}
        {tab === "results" && <ResultsTab rows={results} loading={loading.results} error={error} />}
        {tab === "analysis" && <AnalysisTab rows={analysis} loading={loading.analysis} error={error} />}
      </div>
    </div>
  );
}

function ReportsTab({ filter }: { filter: ReportStatus | "ALL" }) {
  const rows = REPORTS.filter((r) => filter === "ALL" || r.status === filter);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-1 px-6 h-12 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        {REPORT_FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            onClick={() => setReportFilter(f.id)}
            className="px-2.5 h-7 rounded text-sm font-medium transition-colors"
            style={{
              color: reportFilter === f.id ? "var(--accent)" : "var(--text-secondary)",
              background: reportFilter === f.id ? "var(--accent-subtle)" : "transparent",
            }}
          >
            {f.label}
          </button>
        ))}
      </div>

      {rows.length === 0 ? (
        <EmptyState
          icon={FileText}
          message="No reports yet."
          actionLabel="Create Report"
          actionHref="/reports/new"
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 p-6 overflow-y-auto">
          {rows.map((r) => (
            <Link
              key={r.id}
              href={`/reports/${r.id}`}
              className="flex flex-col gap-2.5 p-4 rounded-lg transition-colors"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {r.title}
                </span>
                <span
                  className="text-xs font-medium shrink-0"
                  style={{ color: r.status === "COMPLETED" ? "var(--success)" : "var(--text-secondary)" }}
                >
                  {r.status === "COMPLETED" ? "Completed" : "Draft"}
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
                <span className="font-mono-num">{r.generatedAt ?? "Unavailable"}</span>
                <span>{r.resultIds.length} Results</span>
                <span>{r.sessionCount} Sessions</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function ResultsTab({ rows, loading, error }: { rows: ResultRow[]; loading: boolean; error: string | null }) {
  if (loading) {
    return <p className="px-6 py-4 text-sm" style={{ color: "var(--text-tertiary)" }}>Loading results…</p>;
  }
  if (error) {
    return <p className="px-6 py-4 text-sm" style={{ color: "var(--error)" }}>{error}</p>;
  }
  if (rows.length === 0) {
    return (
      <EmptyState
        icon={BarChart3}
        message="No results yet."
        actionLabel="View Sessions"
        actionHref="/sessions"
      />
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 p-6 overflow-y-auto">
      {rows.map((r) => (
        <Link
          key={r.id}
          href={`/results/${r.id}`}
          className="flex flex-col gap-2.5 p-4 rounded-lg transition-colors"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
              {r.title}
            </span>
            <span className="text-xs font-medium shrink-0" style={{ color: "var(--text-tertiary)" }}>
              {r.type}
            </span>
          </div>
          {r.finding && (
            <p className="text-sm line-clamp-2" style={{ color: "var(--text-secondary)" }}>
              {r.finding}
            </p>
          )}
          <div className="flex items-center gap-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
            <span className="font-mono-num">{r.generatedAt ?? "Unavailable"}</span>
            {r.sessionName && <span className="truncate">{r.sessionName}</span>}
          </div>
        </Link>
      ))}
    </div>
  );
}

function AnalysisTab({ rows, loading, error }: { rows: AnalysisRow[]; loading: boolean; error: string | null }) {
  if (loading) {
    return <p className="px-6 py-4 text-sm" style={{ color: "var(--text-tertiary)" }}>Loading analysis…</p>;
  }
  if (error) {
    return <p className="px-6 py-4 text-sm" style={{ color: "var(--error)" }}>{error}</p>;
  }
  if (rows.length === 0) {
    return (
      <EmptyState
        icon={Workflow}
        message="No analysis has been run yet."
        actionLabel="View Sessions"
        actionHref="/sessions"
      />
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 p-6 overflow-y-auto">
      {rows.map((r) => (
        <Link
          key={r.id}
          href={`/analysis/${r.id}`}
          className="flex flex-col gap-2.5 p-4 rounded-lg transition-colors"
          style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
              {r.name}
            </span>
            <StatusBadge state={r.state} />
          </div>
          <div className="flex items-center gap-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
            <span className="font-mono-num">{r.startedAt ?? "Unavailable"}</span>
            <span className="truncate">{r.sessionName ?? "No session"}</span>
          </div>
        </Link>
      ))}
    </div>
  );
}