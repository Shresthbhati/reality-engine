"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import PageHeader from "@/components/ui/PageHeader";
import { useSessions } from "@/lib/api";
import { createWorld, attachSessionToWorld } from "@/lib/api/worlds";
import { Loader2, AlertCircle } from "lucide-react";

export default function CreateWorldPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [selectedSessionIds, setSelectedSessionIds] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data: rawSessions, isLoading: sessionsLoading } = useSessions();
  const sessions = rawSessions ?? [];

  function toggleSession(id: string) {
    setSelectedSessionIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const selectedSessions = sessions.filter((s) => selectedSessionIds.has(s.id));
  const sessionsWithCoverage = selectedSessions.filter((s) => s.coverageKm2 != null);
  const totalCoverageKm2 = sessionsWithCoverage.reduce((sum, s) => sum + (s.coverageKm2 ?? 0), 0);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || submitting) return;

    setSubmitting(true);
    setError(null);

    try {
      const result = await createWorld({
        name: name.trim(),
        description: location.trim() || undefined,
      });

      // Attach any selected sessions to this new world
      for (const sid of selectedSessionIds) {
        try {
          await attachSessionToWorld(result.id, sid);
        } catch (attachErr) {
          console.warn(`Could not attach session ${sid} to world:`, attachErr);
        }
      }

      router.push(`/worlds/${result.id}`);
    } catch (err: any) {
      setError(err?.message || "Failed to create world");
      setSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#08090b]">
      <PageHeader title="Create World" />

      {error && (
        <div
          className="mx-6 mt-4 flex items-start gap-2 rounded-md px-3 py-2 text-sm"
          style={{ background: "var(--error-subtle)", color: "var(--error)" }}
          role="alert"
        >
          <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <form className="flex flex-col gap-5 px-6 py-6 max-w-md" onSubmit={handleSubmit}>
        <Field label="Name">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. South Building Floor 2"
            className="input"
            required
            disabled={submitting}
          />
        </Field>

        <Field label="Location" hint="Optional coordinate anchor or site description.">
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="e.g. San Francisco Campus, Metric ENU"
            className="input"
            disabled={submitting}
          />
        </Field>

        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            Capture Sessions
          </span>

          {sessionsLoading ? (
            <div className="h-9 px-2.5 flex items-center gap-2 rounded-md text-sm text-neutral-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#00e5ff]" />
              <span>Loading sessions...</span>
            </div>
          ) : sessions.length === 0 ? (
            <div
              className="h-9 px-2.5 flex items-center rounded-md text-sm"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-tertiary)" }}
            >
              No sessions available
            </div>
          ) : (
            <div className="flex flex-col rounded-md overflow-hidden" style={{ border: "1px solid var(--border)" }}>
              {sessions.map((s, i) => (
                <label
                  key={s.id}
                  className="flex items-start gap-2.5 px-2.5 py-2 cursor-pointer transition-colors hover:bg-neutral-800/40"
                  style={{
                    background: "var(--bg-elevated)",
                    borderTop: i === 0 ? undefined : "1px solid var(--border-subtle)",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedSessionIds.has(s.id)}
                    onChange={() => toggleSession(s.id)}
                    disabled={submitting}
                    className="mt-0.5"
                  />
                  <span className="flex flex-col">
                    <span className="text-sm" style={{ color: "var(--text-primary)" }}>
                      {s.name}
                    </span>
                    <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                      {s.location ?? "Unavailable"}
                      {s.capturedAt ? ` · ${s.capturedAt}` : ""}
                    </span>
                  </span>
                </label>
              ))}
            </div>
          )}

          <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
            {sessions.length === 0 && !sessionsLoading ? (
              <>
                No Sessions exist yet to compose this World from.{" "}
                <Link href="/sessions/new" className="underline" style={{ color: "var(--accent)" }}>
                  Create Session
                </Link>
              </>
            ) : (
              "Optional — select capture Sessions to link to this World."
            )}
          </span>
        </div>

        {selectedSessionIds.size > 0 && (
          <div
            className="text-xs rounded-md px-2.5 py-2 flex flex-col gap-0.5"
            style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
          >
            <span>
              {selectedSessionIds.size} Session{selectedSessionIds.size === 1 ? "" : "s"} selected
            </span>
            {sessionsWithCoverage.length > 0 && (
              <span>{totalCoverageKm2.toFixed(1)} km² combined coverage</span>
            )}
          </div>
        )}

        <button
          type="submit"
          disabled={!name.trim() || submitting}
          className="h-9 px-4 rounded-md text-sm font-medium self-start transition-colors disabled:opacity-40 flex items-center gap-2 cursor-pointer"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          <span>{submitting ? "Creating World..." : "Create World"}</span>
        </button>
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

function Field({ label, hint, children }: { label: string; hint?: React.ReactNode; children: React.ReactNode }) {
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
