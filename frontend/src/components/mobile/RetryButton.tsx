"use client";

import { useRouter } from "next/navigation";
import { useTransition } from "react";
import { Loader2, RotateCcw } from "lucide-react";

/** Retry for server-rendered mobile screens: re-runs the server fetch via router.refresh(). Pass `onRetry` for client-fetched screens. */
export default function RetryButton({ onRetry, label = "Try again" }: { onRetry?: () => void; label?: string }) {
  const router = useRouter();
  const [pending, start] = useTransition();
  return (
    <button
      type="button"
      onClick={() => (onRetry ? onRetry() : start(() => router.refresh()))}
      disabled={pending}
      className="mt-2 inline-flex h-11 min-w-[44px] items-center justify-center gap-2 rounded-lg px-4 text-sm font-semibold disabled:opacity-60"
      style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
    >
      {pending ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <RotateCcw className="h-4 w-4" aria-hidden />}
      {label}
    </button>
  );
}
