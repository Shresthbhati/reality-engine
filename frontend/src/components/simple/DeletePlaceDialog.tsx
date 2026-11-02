"use client";

import { useEffect, useRef, useState } from "react";
import { deleteWorld, isApiError } from "@/lib/api";
import { HOME } from "@/lib/copy";

/** Typed-name confirmation. The API refuses (409) while a build runs; that reason is shown in plain words. */
export default function DeletePlaceDialog({
  id,
  name,
  onClose,
  onDeleted,
}: {
  id: string;
  name: string;
  onClose: () => void;
  onDeleted: () => void;
}) {
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Focus moves into the dialog, Tab stays inside it, Escape closes, and focus returns to the opener on close.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    inputRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key !== "Tab" || !dialogRef.current) return;
      const items = Array.from(dialogRef.current.querySelectorAll<HTMLElement>("button:not(:disabled), input"));
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);

  const matches = typed.trim() === name.trim();

  const confirm = async () => {
    if (!matches || busy) return;
    setBusy(true);
    setError(null);
    try {
      await deleteWorld(id);
      onDeleted();
    } catch (err) {
      setError(isApiError(err) && err.code === "conflict" ? HOME.deleteBusy : err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "rgba(0,0,0,0.5)" }} onClick={onClose}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-place-title"
        className="w-full max-w-sm rounded-lg p-5"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="delete-place-title" className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
          {HOME.deleteTitle}
        </h2>
        <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
          {HOME.deleteBody}
        </p>
        <label className="mt-4 block text-xs" style={{ color: "var(--text-secondary)" }}>
          {HOME.deleteType}: <strong style={{ color: "var(--text-primary)" }}>{name}</strong>
          <input
            ref={inputRef}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void confirm()}
            className="mt-1 h-9 w-full rounded-md px-2 text-sm"
            style={{ background: "var(--bg-base)", border: "1px solid var(--border-strong)", color: "var(--text-primary)" }}
          />
        </label>
        {error && (
          <p role="alert" className="mt-3 text-xs" style={{ color: "var(--error)" }}>
            {error}
          </p>
        )}
        <div className="mt-5 flex justify-end gap-2">
          <button type="button" onClick={onClose} className="h-9 rounded-md px-3 text-sm" style={{ color: "var(--text-secondary)" }}>
            {HOME.cancel}
          </button>
          <button
            type="button"
            disabled={!matches || busy}
            onClick={() => void confirm()}
            className="btn-press h-9 rounded-md px-3 text-sm font-medium disabled:opacity-40"
            style={{ background: "var(--error)", color: "var(--on-error)" }}
          >
            {HOME.delete}
          </button>
        </div>
      </div>
    </div>
  );
}
