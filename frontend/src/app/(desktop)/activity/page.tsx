"use client";

import { useState, useEffect } from "react";
import PageHeader from "@/components/ui/PageHeader";
import ActivityFeed from "@/components/ui/ActivityFeed";
import { getActivity, isApiError, type ActivityType, type ActivityEvent } from "@/lib/activity";

const FILTERS: Array<{ id: ActivityType | "ALL"; label: string }> = [
  { id: "ALL", label: "All" },
  { id: "WORLD", label: "Worlds" },
  { id: "SESSION", label: "Sessions" },
  { id: "EVIDENCE", label: "Evidence" },
  { id: "ANALYSIS", label: "Analysis" },
  { id: "RESULT", label: "Results" },
  { id: "REPORT", label: "Reports" },
];

export default function ActivityPage() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]["id"]>("ALL");
  // `events === null` while loading or after a failure; an empty array is a
  // real answer from the API meaning nothing has been recorded.
  const [events, setEvents] = useState<ActivityEvent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    getActivity()
      .then((rows) => {
        if (!active) return;
        setEvents(rows.filter((e) => filter === "ALL" || e.type === filter));
        setError(null);
      })
      .catch((err) => {
        if (!active) return;
        setEvents(null);
        setError(isApiError(err) ? err.describe() : (err as Error).message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [filter]);

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Activity" />
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
      <div className="p-6 max-w-2xl">
        {loading && events === null ? (
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            Loading Activity…
          </p>
        ) : error ? (
          <p className="text-sm" role="alert" style={{ color: "var(--error)" }}>
            Activity could not be loaded. {error}
          </p>
        ) : (
          <ActivityFeed events={events ?? []} />
        )}
      </div>
    </div>
  );
}
