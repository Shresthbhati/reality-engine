"use client";

import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { RefreshCw } from "lucide-react";
import { reconstructSession, isApiError } from "@/lib/api";

/** Enqueues a real reconstruction job for this Session and refreshes the page to show it. */
export default function ReconstructButton({ sessionId }: { sessionId: string }) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [error, setError] = useState<string | null>(null);

  const onClick = () => {
    setError(null);
    startTransition(async () => {
      try {
        await reconstructSession(sessionId);
        router.refresh();
      } catch (err) {
        setError(isApiError(err) ? err.describe() : "Failed to start reconstruction.");
      }
    });
  };

  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        onClick={onClick}
        disabled={isPending}
        className="flex items-center justify-center gap-2 h-11 rounded-md text-sm font-semibold disabled:opacity-50"
        style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
      >
        <RefreshCw className={`w-4 h-4 ${isPending ? "animate-spin" : ""}`} />
        {isPending ? "Starting…" : "Start Reconstruction"}
      </button>
      {error && (
        <p className="text-xs" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
