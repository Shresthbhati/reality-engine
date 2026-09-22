"use client";

import { useState } from "react";
import PageHeader from "@/components/ui/PageHeader";

export default function CreateSessionPage() {
  const [name, setName] = useState("");
  const [source, setSource] = useState("Upload");
  const [location, setLocation] = useState("");

  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Create Session" />
      <form
        className="flex flex-col gap-5 px-6 py-6 max-w-md"
        onSubmit={(e) => e.preventDefault()}
      >
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
          <select className="input">
            <option>None</option>
          </select>
        </Field>

        <button
          type="submit"
          disabled={!name.trim()}
          className="h-9 px-4 rounded-md text-sm font-medium self-start transition-colors disabled:opacity-40"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          Create Session
        </button>

        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          Session creation isn&apos;t wired to a backend yet — this form is the real UI, not yet connected.
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
