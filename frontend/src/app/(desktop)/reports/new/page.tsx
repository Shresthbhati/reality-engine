"use client";

import { useState } from "react";
import Link from "next/link";
import PageHeader from "@/components/ui/PageHeader";
import { RESULTS } from "@/lib/data";

export default function CreateReportPage() {
  const [title, setTitle] = useState("");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  function toggleResult(id: string) {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]));
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Create Report" />
      <form className="flex flex-col gap-5 px-6 py-6 max-w-md" onSubmit={(e) => e.preventDefault()}>
        <Field label="Title">
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="e.g. Adyar Site Survey — Q1 Findings"
            className="input"
          />
        </Field>

        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            Results
          </span>
          {RESULTS.length === 0 ? (
            <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
              No completed Results are available to include yet.{" "}
              <Link href="/analysis" className="font-medium hover:underline" style={{ color: "var(--accent)" }}>
                View Analysis
              </Link>
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              {RESULTS.map((r) => (
                <label key={r.id} className="flex items-center gap-2 text-sm" style={{ color: "var(--text-primary)" }}>
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(r.id)}
                    onChange={() => toggleResult(r.id)}
                  />
                  {r.title}
                </label>
              ))}
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={RESULTS.length === 0}
          className="h-9 px-4 rounded-md text-sm font-medium self-start transition-colors disabled:opacity-40"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          Create Report
        </button>

        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          Report creation isn&apos;t wired to a backend yet — this form is the real UI, not yet connected.
        </p>
      </form>

      <style jsx>{`
        .input {
          height: 36px;
          padding: 0 10px;
          border-radius: 6px;
          background: var(--bg-elevated);
          border: 1px solid var(--border);
          color: var(--text-primary);
          font-size: 14px;
          outline: none;
        }
        .input:focus {
          border-color: var(--accent-border);
        }
      `}</style>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        {label}
      </span>
      {children}
      {hint && (
        <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          {hint}
        </span>
      )}
    </label>
  );
}
