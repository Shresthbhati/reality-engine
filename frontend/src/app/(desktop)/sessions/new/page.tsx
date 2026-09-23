"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageHeader from "@/components/ui/PageHeader";
import { createSession } from "@/lib/api/sessions";
import { attachSessionToWorld, listWorlds } from "@/lib/api/worlds";
import type { WorldRow } from "@/lib/types";

export default function CreateSessionPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [source, setSource] = useState("Upload");
  const [location, setLocation] = useState("");
  const [worldId, setWorldId] = useState("");
  const [worlds, setWorlds] = useState<WorldRow[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listWorlds()
      .then((res) => {
        if (!cancelled) setWorlds(res.rows);
      })
      .catch(() => {
        if (!cancelled) setWorlds([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await createSession({
        name: name.trim(),
        device_metadata: { source, location_label: location.trim() || null },
      });
      if (worldId) {
        await attachSessionToWorld(worldId, result.id).catch(() => {
          // Session is created regardless; attach can be retried from the
          // Session detail screen if this fails.
        });
      }
      router.push(`/sessions/${result.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create session");
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Create Session" />
      <form className="flex flex-col gap-5 px-6 py-6 max-w-md" onSubmit={handleSubmit}>
        <Field label="Name">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Adyar Exterior Capture"
            className="input"
          />
        </Field>

        <Field label="Source">
          <select value={source} onChange={(e) => setSource(e.target.value)} className="input">
            <option>Camera</option>
            <option>Upload</option>
            <option>Import</option>
          </select>
        </Field>

        <Field label="Location" hint="Optional — filled automatically from capture data when available.">
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            placeholder="e.g. Adyar, Chennai"
            className="input"
          />
        </Field>

        <Field label="World" hint="Optional — this Session can stay standalone.">
          <select value={worldId} onChange={(e) => setWorldId(e.target.value)} className="input">
            <option value="">None</option>
            {worlds.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        </Field>

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
          {submitting ? "Creating…" : "Create Session"}
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
