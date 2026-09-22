"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Camera, Plus, Search } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import StatusBadge from "@/components/ui/StatusBadge";
import type { ProcessingState } from "@/lib/types";
import { SESSIONS } from "@/lib/data";

const FILTERS: Array<{ id: ProcessingState | "ALL"; label: string }> = [
  { id: "ALL", label: "All" },
  { id: "PROCESSING", label: "Running" },
  { id: "QUEUED", label: "Queued" },
  { id: "COMPLETE", label: "Completed" },
  { id: "FAILED", label: "Failed" },
];

function isProcessingState(v: string | null): v is ProcessingState {
  return v === "QUEUED" || v === "PROCESSING" || v === "COMPLETE" || v === "FAILED";
}

export default function SessionsPage() {
  return (
    <Suspense>
      <SessionsPageInner />
    </Suspense>
  );
}

function SessionsPageInner() {
  const searchParams = useSearchParams();
  const stateParam = searchParams.get("state");
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["id"]>(
    isProcessingState(stateParam) ? stateParam : "ALL",
  );
  const [query, setQuery] = useState("");

  const rows = SESSIONS.filter((s) => (filter === "ALL" || s.state === filter))
    .filter((s) => !query || s.name.toLowerCase().includes(query.toLowerCase()));

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Sessions"
        action={
          <Link
            href="/sessions/new"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Create Session
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
            placeholder="Search Sessions…"
            className="bg-transparent outline-none text-sm w-full"
            style={{ color: "var(--text-primary)" }}
          />
        </div>
      </div>

      {rows.length === 0 ? (
        <EmptyState
          icon={Camera}
          message={
            SESSIONS.length === 0
              ? "No Sessions yet."
              : "No Sessions match this filter."
          }
          actionLabel={SESSIONS.length === 0 ? "Create Session" : undefined}
          actionHref={SESSIONS.length === 0 ? "/sessions/new" : undefined}
        />
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b" style={{ borderColor: "var(--border)" }}>
              {["Session", "State", "World", "Location", "Captured", "Duration"].map((h) => (
                <th key={h} className="px-6 py-2 font-medium" style={{ color: "var(--text-tertiary)" }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.id} className="border-b" style={{ borderColor: "var(--border-subtle)" }}>
                <td className="px-6 py-2.5">
                  <Link href={`/sessions/${s.id}`} className="font-medium hover:underline" style={{ color: "var(--text-primary)" }}>
                    {s.name}
                  </Link>
                </td>
                <td className="px-6 py-2.5">
                  <StatusBadge state={s.state} />
                </td>
                <td className="px-6 py-2.5" style={{ color: "var(--text-secondary)" }}>
                  {s.worldName ?? "Standalone"}
                </td>
                <td className="px-6 py-2.5" style={{ color: "var(--text-secondary)" }}>
                  {s.location ?? "—"}
                </td>
                <td className="px-6 py-2.5 font-mono-num" style={{ color: "var(--text-secondary)" }}>
                  {s.capturedAt ?? "—"}
                </td>
                <td className="px-6 py-2.5 font-mono-num" style={{ color: "var(--text-secondary)" }}>
                  {s.durationSec != null ? `${Math.round(s.durationSec / 60)}m` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
