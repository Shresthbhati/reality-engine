"use client";

import { useState } from "react";
import Link from "next/link";
import { FileText, Plus } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import type { ReportStatus } from "@/lib/types";
import { REPORTS } from "@/lib/data";

const FILTERS: Array<{ id: ReportStatus | "ALL"; label: string }> = [
  { id: "ALL", label: "All" },
  { id: "DRAFT", label: "Draft" },
  { id: "COMPLETED", label: "Completed" },
];

export default function ReportsPage() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["id"]>("ALL");

  const rows = REPORTS.filter((r) => filter === "ALL" || r.status === filter);

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

      <div className="flex items-center gap-1 px-6 h-12 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
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

      {rows.length === 0 ? (
        <EmptyState
          icon={FileText}
          message="Generate a Report from completed Results."
          actionLabel="View Analysis"
          actionHref="/analysis"
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
