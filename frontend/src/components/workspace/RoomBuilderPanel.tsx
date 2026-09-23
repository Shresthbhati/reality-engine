"use client";

import { useState } from "react";
import { X, Loader2 } from "lucide-react";
import { createRoom } from "@/lib/api/procedural";
import { isApiError } from "@/lib/api/client";

/** "Create Room" form -> real POST /api/worlds/{id}/rooms -> WorldStore
 * version commit -> caller refetches WorldIR. No local fabrication: the
 * room only exists once the backend has committed it. */
export default function RoomBuilderPanel({
  worldId,
  onClose,
  onCreated,
}: {
  worldId: string;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [name, setName] = useState("room-1");
  const [width, setWidth] = useState(4);
  const [depth, setDepth] = useState(3);
  const [height, setHeight] = useState(2.4);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      await createRoom(worldId, { name, width, depth, height });
      onCreated();
      onClose();
    } catch (err) {
      setError(isApiError(err) ? err.describe() : "Failed to create room.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="absolute top-4 right-4 z-20 w-72 rounded-lg p-4 flex flex-col gap-3 shadow-xl"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          Create Room
        </h3>
        <button type="button" onClick={onClose} aria-label="Close">
          <X className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
        </button>
      </div>

      <Field label="Name">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
          style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
        />
      </Field>

      <div className="grid grid-cols-3 gap-2">
        <Field label="Width (m)">
          <NumberInput value={width} onChange={setWidth} />
        </Field>
        <Field label="Depth (m)">
          <NumberInput value={depth} onChange={setDepth} />
        </Field>
        <Field label="Height (m)">
          <NumberInput value={height} onChange={setHeight} />
        </Field>
      </div>

      {error && (
        <p className="text-xs" style={{ color: "var(--error, #e57373)" }}>
          {error}
        </p>
      )}

      <button
        type="button"
        onClick={handleSubmit}
        disabled={submitting || !name.trim() || width <= 0 || depth <= 0 || height <= 0}
        className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-medium disabled:opacity-40 transition-colors"
        style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
      >
        {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
        {submitting ? "Creating…" : "Create Room"}
      </button>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs" style={{ color: "var(--text-tertiary)" }}>
      {label}
      {children}
    </label>
  );
}

function NumberInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      step="0.1"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full bg-transparent outline-none text-sm px-2 h-8 rounded"
      style={{ border: "1px solid var(--border-subtle)", color: "var(--text-primary)" }}
    />
  );
}
