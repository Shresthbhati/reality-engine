"use client";

import { useState } from "react";
import Link from "next/link";
import { Save } from "lucide-react";
import type { ReportRow, ResultRow } from "@/lib/types";

const OUTLINE = ["Summary", "Findings", "Maps", "Evidence", "Results", "Appendix"] as const;

export default function EditorClient({ report, results }: { report: ReportRow; results: ResultRow[] }) {
  const [active, setActive] = useState<(typeof OUTLINE)[number]>("Summary");
  const [summary, setSummary] = useState(report.summary ?? "");
  const [saved, setSaved] = useState(false);

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-6 h-14 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <h1 className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>
          {report.title}
        </h1>
        <div className="flex items-center gap-2 shrink-0">
          {saved && (
            <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
              Saved locally — not persisted to a backend.
            </span>
          )}
          <button
            type="button"
            onClick={() => setSaved(true)}
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Save className="w-3.5 h-3.5" />
            Save
          </button>
          <Link
            href={`/reports/${report.id}`}
            className="text-sm font-medium px-3 h-8 flex items-center rounded-md"
            style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Preview
          </Link>
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        <nav className="w-44 shrink-0 border-r flex flex-col gap-0.5 p-2" style={{ borderColor: "var(--border)" }}>
          {OUTLINE.map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => setActive(item)}
              className="text-left px-2.5 h-8 rounded-md text-sm font-medium transition-colors"
              style={{
                color: active === item ? "var(--accent)" : "var(--text-secondary)",
                background: active === item ? "var(--accent-subtle)" : "transparent",
              }}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="flex-1 overflow-y-auto p-6 max-w-2xl">
          {active === "Summary" && (
            <div className="flex flex-col gap-2">
              <label className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
                Executive Summary
              </label>
              <textarea
                value={summary}
                onChange={(e) => setSummary(e.target.value)}
                rows={10}
                placeholder="Write a summary of this Report's findings…"
                className="w-full p-3 rounded-md text-sm resize-y outline-none"
                style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
              />
            </div>
          )}

          {active === "Findings" && (
            <SectionList
              title="Findings"
              empty="No findings recorded — findings come from the Results linked to this Report."
              rows={results.filter((r) => r.finding).map((r) => ({ id: r.id, label: r.title, value: r.finding! }))}
            />
          )}

          {active === "Maps" && (
            <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
              No spatial results are linked to this Report yet.
            </p>
          )}

          {active === "Evidence" && (
            <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
              Evidence referenced by this Report&apos;s Results will appear here.
            </p>
          )}

          {active === "Results" && (
            <SectionList
              title="Results"
              empty="No Results are linked to this Report."
              rows={results.map((r) => ({ id: r.id, label: r.title, value: r.generatedAt ?? "Unavailable" }))}
            />
          )}

          {active === "Appendix" && (
            <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
              No appendix content yet.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function SectionList({ title, empty, rows }: { title: string; empty: string; rows: Array<{ id: string; label: string; value: string }> }) {
  if (rows.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        {empty}
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h2>
      {rows.map((r) => (
        <div key={r.id} className="p-3 rounded-md" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}>
          <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            {r.label}
          </p>
          <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>
            {r.value}
          </p>
        </div>
      ))}
    </div>
  );
}
