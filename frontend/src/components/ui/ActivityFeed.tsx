import Link from "next/link";
import { Clock } from "lucide-react";
import type { ActivityEvent } from "@/lib/activity";

export default function ActivityFeed({ events, limit }: { events: ActivityEvent[]; limit?: number }) {
  const rows = limit ? events.slice(0, limit) : events;

  if (rows.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        No activity recorded yet.
      </p>
    );
  }

  return (
    <ol className="flex flex-col gap-1">
      {rows.map((e) => (
        <li key={`${e.type}-${e.id}-${e.timestamp}`}>
          <Link
            href={e.href}
            className="flex items-center gap-2.5 px-2 py-1.5 rounded-md text-sm transition-colors"
            style={{ color: "var(--text-primary)" }}
          >
            <Clock className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--text-tertiary)" }} />
            <span className="flex-1 truncate">{e.label}</span>
            <span className="text-xs font-mono-num shrink-0" style={{ color: "var(--text-tertiary)" }}>
              {e.timestamp}
            </span>
          </Link>
        </li>
      ))}
    </ol>
  );
}
