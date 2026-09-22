"use client";

import { useState } from "react";
import Link from "next/link";
import { Search, ShieldCheck } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import EvidenceCard from "@/components/ui/EvidenceCard";
import type { EvidenceType } from "@/lib/types";
import { EVIDENCE } from "@/lib/data";

const FILTERS: Array<{ id: EvidenceType | "ALL"; label: string }> = [
  { id: "ALL", label: "All" },
  { id: "IMAGE", label: "Imagery" },
  { id: "VIDEO", label: "Video" },
  { id: "POINT_CLOUD", label: "Point Cloud" },
  { id: "DOCUMENT", label: "Documents" },
  { id: "SENSOR_DATA", label: "Sensor Data" },
];

export default function EvidencePage() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["id"]>("ALL");
  const [query, setQuery] = useState("");

  const rows = EVIDENCE.filter((e) => filter === "ALL" || e.type === filter).filter(
    (e) => !query || e.name.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Evidence"
        action={
          <Link
            href="/sessions"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            View Sessions
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
            placeholder="Search Evidence…"
            className="bg-transparent outline-none text-sm w-full"
            style={{ color: "var(--text-primary)" }}
          />
        </div>
      </div>

      {rows.length === 0 ? (
        <EmptyState icon={ShieldCheck} message="No Evidence has been added yet." />
      ) : (
        <div className="grid gap-4 p-6" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
          {rows.map((e) => (
            <EvidenceCard key={e.id} evidence={e} />
          ))}
        </div>
      )}
    </div>
  );
}
