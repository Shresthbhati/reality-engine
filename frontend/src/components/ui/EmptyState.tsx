import Link from "next/link";
import type { LucideIcon } from "lucide-react";

export default function EmptyState({
  icon: Icon,
  message,
  actionLabel,
  actionHref,
}: {
  icon: LucideIcon;
  message: string;
  actionLabel?: string;
  actionHref?: string;
}) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 py-24 px-6 text-center">
      <Icon className="w-8 h-8" style={{ color: "var(--text-tertiary)" }} />
      <p className="text-sm max-w-sm" style={{ color: "var(--text-secondary)" }}>
        {message}
      </p>
      {actionLabel && actionHref && (
        <Link
          href={actionHref}
          className="text-sm font-medium px-3 h-9 flex items-center rounded-md mt-1 transition-colors"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          {actionLabel}
        </Link>
      )}
    </div>
  );
}
