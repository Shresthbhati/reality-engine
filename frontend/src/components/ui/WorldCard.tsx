import Link from "next/link";
import { Globe } from "lucide-react";
import type { WorldRow } from "@/lib/types";

export default function WorldCard({ world }: { world: WorldRow }) {
  return (
    <Link
      href={`/worlds/${world.id}`}
      className="flex flex-col rounded-lg overflow-hidden transition-colors"
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div
        className="h-32 flex items-center justify-center shrink-0"
        style={{ background: "var(--bg-elevated)" }}
      >
        <Globe className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
      </div>

      <div className="flex flex-col gap-1 p-4">
        <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          {world.name}
        </span>
        {world.location && (
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            {world.location}
          </span>
        )}

        <div className="flex items-center gap-3 mt-2 text-xs font-mono-num" style={{ color: "var(--text-secondary)" }}>
          {world.coverageKm2 != null && <span>{world.coverageKm2} km²</span>}
          <span>{world.sessionCount} Sessions</span>
          <span>{world.evidenceCount} Evidence</span>
        </div>

        {world.timeRangeStart && world.timeRangeEnd && (
          <span className="text-xs font-mono-num mt-1" style={{ color: "var(--text-tertiary)" }}>
            {world.timeRangeStart} – {world.timeRangeEnd}
          </span>
        )}
      </div>
    </Link>
  );
}
