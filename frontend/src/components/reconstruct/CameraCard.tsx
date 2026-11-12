"use client";

import React, { useEffect, useState } from "react";
import { Camera } from "lucide-react";
import { CAMERA_FIELDS, declareCamera, getCameraDeclaration, validateCamera, type CameraKind } from "@/lib/api/camera";

export interface CameraCardProps {
  worldId: string;
  /** called after a declaration so the owner can refresh the reconstruction status */
  onChanged?: () => void;
}

/**
 * Optional camera declaration. Photos exported without EXIF (messengers, screenshots) give the engine no trusted
 * focal length, so depth, perception and meshing skip. Entering the camera here supplies it; nothing is guessed.
 */
export default function CameraCard({ worldId, onChanged }: CameraCardProps) {
  const [kind, setKind] = useState<CameraKind>("focal_length_35mm");
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    getCameraDeclaration(worldId).then(
      (r) => {
        const entry = r.camera && (Object.entries(r.camera)[0] as [CameraKind, number] | undefined);
        if (live && entry) setSaved(`${CAMERA_FIELDS[entry[0]].label}: ${entry[1]} ${CAMERA_FIELDS[entry[0]].unit}`);
      },
      () => undefined,
    );
    return () => {
      live = false;
    };
  }, [worldId]);

  const f = CAMERA_FIELDS[kind];
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const bad = validateCamera(kind, value);
    if (bad) return setError(bad);
    setBusy(true);
    setError(null);
    try {
      await declareCamera(worldId, kind, Number(value));
      setSaved(`${f.label}: ${value} ${f.unit}`);
      setMessage("Camera saved. The model is being re-evaluated; depth and mesh run if their other checks pass.");
      onChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save the camera. Nothing was changed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <details className="mt-2 rounded border border-[var(--border)] p-2" data-testid="camera-card">
      <summary className="flex cursor-pointer items-center gap-1.5 text-[11px] font-semibold text-fg">
        <Camera className="h-3.5 w-3.5" aria-hidden />
        Camera {saved ? `· ${saved}` : "· not declared"}
      </summary>
      <form onSubmit={submit} className="mt-1.5 flex flex-wrap items-end gap-2">
        <label className="text-[10px] text-fg-muted">
          Known value
          <select
            value={kind}
            onChange={(e) => {
              setKind(e.target.value as CameraKind);
              setError(null);
            }}
            className="mt-0.5 block rounded border border-[var(--border)] bg-transparent px-1 py-0.5 text-[11px] text-fg"
          >
            {(Object.keys(CAMERA_FIELDS) as CameraKind[]).map((k) => (
              <option key={k} value={k}>
                {CAMERA_FIELDS[k].label}
              </option>
            ))}
          </select>
        </label>
        <label className="text-[10px] text-fg-muted">
          {f.unit}
          <input
            inputMode="decimal"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            aria-invalid={error ? true : undefined}
            placeholder={`${f.min}–${f.max}`}
            className="mt-0.5 block w-24 rounded border border-[var(--border)] bg-transparent px-1 py-0.5 text-[11px] text-fg"
          />
        </label>
        <button
          type="submit"
          disabled={busy}
          className="rounded border border-[var(--border)] px-2 py-0.5 text-[11px] font-medium text-fg hover:bg-hover disabled:opacity-50"
        >
          {busy ? "Saving…" : "Use this camera"}
        </button>
      </form>
      <p className="mt-1 text-[10px] text-fg-muted">
        Only needed when the photos have no EXIF focal length. Result is labelled estimated (centre principal point).
      </p>
      {message && (
        <p className="mt-1 text-[10px] text-fg-muted" role="status">
          {message}
        </p>
      )}
      {error && (
        <p className="mt-1 text-[10px] text-error" role="alert">
          {error}
        </p>
      )}
    </details>
  );
}
