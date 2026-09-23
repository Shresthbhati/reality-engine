import { AlertTriangle, type LucideIcon } from "lucide-react";

/** Shared error state for mobile screens whose data fetch failed — never a silent blank screen. */
export function MobileErrorState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center gap-2 py-16 px-4 text-center">
      <AlertTriangle className="w-6 h-6" style={{ color: "var(--error)" }} />
      <p className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        Couldn&apos;t load this
      </p>
      <p className="text-xs max-w-xs" style={{ color: "var(--text-tertiary)" }}>
        {message}
      </p>
    </div>
  );
}

export function MobileEmptyState({ icon: Icon, message }: { icon: LucideIcon; message: string }) {
  return (
    <div className="flex flex-col items-center gap-2 py-16 px-4 text-center">
      <Icon className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        {message}
      </p>
    </div>
  );
}
