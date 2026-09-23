"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import PageHeader from "@/components/ui/PageHeader";
import { listSessions } from "@/lib/api/sessions";
import { createWorld, attachSessionToWorld } from "@/lib/api/worlds";
import type { SessionRow } from "@/lib/types";

export default function CreateWorldPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [selectedSessionIds, setSelectedSessionIds] = useState<Set<string>>(new Set());
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listSessions()
      .then((res) => {
        if (!cancelled) setSessions(res.rows);
      })
      .catch(() => {
        if (!cancelled) setSessions([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function toggleSession(id: string) {
    setSelectedSessionIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const world = await createWorld({ name: name.trim(), description: location.trim() || null });
      await Promise.all(
        Array.from(selectedSessionIds).map((sessionId) =>
          attachSessionToWorld(world.id, sessionId).catch(() => {
            // World is created regardless; a failed attach can be retried
            // from the Session detail screen.
          })
        )
      );
      router.push(`/worlds/${world.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create world");
      setSubmitting(false);
    }
  }

  const selectedSessions = sessions.filter((s) => selectedSessionIds.has(s.id));
  const sessionsWithCoverage = selectedSessions.filter((s) => s.coverageKm2 != null);
  const totalCoverageKm2 = sessionsWithCoverage.reduce((sum, s) => sum + (s.coverageKm2 ?? 0), 0);

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Create World" />
      <form className="flex flex-col gap-5 px-6 py-6 max-w-md" onSubmit={handleSubmit}>
        <Field label="Name">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Adyar Survey"
            className="input"
          />
        </Field>

        <Field label="Location" hint="Optional.">
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="e.g. Chennai, Tamil Nadu"
            className="input"
          />
        </Field>

        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
            Sessions
          </span>
          {sessions.length === 0 ? (
            <div
              className="h-9 px-2.5 flex items-center rounded-md text-sm"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-tertiary)" }}
            >
              None selected
            </div>
          ) : (
            <div className="flex flex-col rounded-md overflow-hidden" style={{ border: "1px solid var(--border)" }}>
              {sessions.map((s, i) => (
                <label
                  key={s.id}
                  className="flex items-start gap-2.5 px-2.5 py-2 cursor-pointer"
                  style={{
                    background: "var(--bg-elevated)",
                    borderTop: i === 0 ? undefined : "1px solid var(--border-subtle)",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedSessionIds.has(s.id)}
                    onChange={() => toggleSession(s.id)}
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
            {sessions.length === 0 ? (
              <>
                No Sessions exist yet to compose this World from.{" "}
                <Link href="/sessions/new" className="underline" style={{ color: "var(--accent)" }}>
                  Create Session
                </Link>
              </>
            ) : (
              "Optional — select Sessions to compose this World from now, or add them later."
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

        {error && (
          <div
            className="text-xs rounded-md px-2.5 py-2"
            style={{ background: "var(--error-subtle, #3a1f1f)", border: "1px solid var(--error-border, #7a3030)", color: "var(--error, #e57373)" }}
          >
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={!name.trim() || submitting}
          className="h-9 px-4 rounded-md text-sm font-medium self-start transition-colors disabled:opacity-40"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          {submitting ? "Creating…" : "Create World"}
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
