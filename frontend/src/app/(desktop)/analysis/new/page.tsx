"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { CheckCircle2, XCircle } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import { listSessions, isApiError } from "@/lib/api";
import type { SessionRow } from "@/lib/types";

const METHODS = ["Structural inspection", "Environmental analysis", "Coverage validation"];

export default function CreateAnalysisPage() {
  const [sessionId, setSessionId] = useState("");
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  useEffect(() => {
    listSessions()
      .then((d) => {
        setSessions(d.rows);
        setSessionId((prev) => prev || (d.rows[0]?.id ?? ""));
      })
      .catch((e) => setSessionsError(isApiError(e) ? e.describe() : String(e)));
  }, []);
  const [method, setMethod] = useState(METHODS[0]);

  const selectedSession = sessions.find((s) => s.id === sessionId) ?? null;
  const hasEvidence = selectedSession ? selectedSession.evidenceCount > 0 : false;
  const hasLocation = selectedSession ? selectedSession.lat !== null && selectedSession.lng !== null : false;

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Create Analysis" />
      <form
        className="flex flex-col gap-5 px-6 py-6 max-w-md"
        onSubmit={(e) => e.preventDefault()}
      >
        <Field
          label="Session"
          hint={
            sessions.length === 0 ? (
              <>
                Create a Session first to run Analysis against it.{" "}
                <Link href="/sessions/new" className="underline" style={{ color: "var(--accent)" }}>
                  Create Session
                </Link>
              </>
            ) : undefined
          }
        >
          <select
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            disabled={sessions.length === 0}
            className="input"
          >
            {sessions.length === 0 ? (
              <option>No Sessions available</option>
            ) : (
              sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))
            )}
          </select>
        </Field>

        <Field label="Method">
          <select value={method} onChange={(e) => setMethod(e.target.value)} className="input">
            {METHODS.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </Field>

        {selectedSession && (
          <div className="flex flex-col gap-1.5">
            <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
              Ready to run
            </span>
            <div
              className="flex flex-col gap-1.5 rounded-md px-2.5 py-2"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
            >
              <ChecklistRow pass label="Session selected" />
              <ChecklistRow
                pass={hasEvidence}
                label="Session has Evidence"
                failText="This Session has no Evidence yet."
              />
              <ChecklistRow
                pass={hasLocation}
                label="Session location available"
                failText="No location recorded for this Session."
              />
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={sessions.length === 0 || !selectedSession || !hasEvidence || !!sessionsError}
          className="h-9 px-4 rounded-md text-sm font-medium self-start transition-colors disabled:opacity-40"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          Create Analysis
        </button>

        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          Analysis creation isn&apos;t wired to a backend yet — this form is the real UI, not yet connected.
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
        .input:disabled {
          opacity: 0.6;
        }
      `}</style>
    </div>
  );
}

function ChecklistRow({ pass, label, failText }: { pass: boolean; label: string; failText?: string }) {
  const Icon = pass ? CheckCircle2 : XCircle;
  return (
    <div className="flex items-start gap-2">
      <Icon size={16} className="mt-0.5 shrink-0" style={{ color: pass ? "var(--success)" : "var(--error)" }} />
      <span className="flex flex-col text-sm" style={{ color: "var(--text-primary)" }}>
        {label}
        {!pass && failText && (
          <span className="text-xs" style={{ color: "var(--error)" }}>
            {failText}
          </span>
        )}
      </span>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
}) {
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
