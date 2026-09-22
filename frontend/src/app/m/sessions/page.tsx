import Link from "next/link";
import { Camera } from "lucide-react";
import StatusBadge from "@/components/ui/StatusBadge";
import { SESSIONS } from "@/lib/data";

export default function MobileSessionsPage() {
  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg font-semibold p-4 pb-0" style={{ color: "var(--text-primary)" }}>
        Sessions
      </h1>

      {SESSIONS.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-16 px-4 text-center">
          <Camera className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No Sessions yet.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2 px-4 pb-4">
          {SESSIONS.map((s) => (
            <Link
              key={s.id}
              href={`/m/sessions/${s.id}`}
              className="flex flex-col justify-center gap-1 min-h-[56px] px-3 py-2 rounded-md"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {s.name}
                </span>
                <StatusBadge state={s.state} />
              </div>
              <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                {[s.location, s.capturedAt].filter(Boolean).join(" · ") || "Unavailable"}
              </span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
