"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { Trash2 } from "lucide-react";
import { deleteEvidence, isApiError } from "@/lib/api";

/** Correction action: removes a mis-captured Evidence item via the real API. */
export default function DeleteEvidenceButton({ evidenceId }: { evidenceId: string }) {
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  if (!confirming) {
    return (
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="flex items-center justify-center gap-2 h-11 rounded-md text-sm font-medium"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--error)" }}
      >
        <Trash2 className="w-4 h-4" />
        Delete Evidence
      </button>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-center" style={{ color: "var(--text-tertiary)" }}>
        This permanently removes the Evidence item and its artifact.
      </p>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setConfirming(false)}
          disabled={isPending}
          className="flex-1 h-11 rounded-md text-sm font-medium"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
        >
          Cancel
        </button>
        <button
          type="button"
          disabled={isPending}
          onClick={() =>
            startTransition(async () => {
              try {
                await deleteEvidence(evidenceId);
                router.push("/m/evidence");
                router.refresh();
              } catch (err) {
                setError(isApiError(err) ? err.describe() : "Failed to delete.");
              }
            })
          }
          className="flex-1 h-11 rounded-md text-sm font-semibold disabled:opacity-50"
          style={{ background: "var(--error-subtle)", color: "var(--error)", border: "1px solid var(--error)" }}
        >
          {isPending ? "Deleting…" : "Confirm Delete"}
        </button>
      </div>
      {error && (
        <p className="text-xs text-center" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
