import Link from "next/link";
import { Camera } from "lucide-react";
import StatusBadge from "@/components/ui/StatusBadge";
import { MobileEmptyState, MobileErrorState } from "@/components/mobile/AsyncState";
import { listSessions, isApiError } from "@/lib/api";

export default async function MobileSessionsPage() {
  let sessions;
  try {
    sessions = (await listSessions()).rows;
  } catch (err) {
    return (
      <div className="flex flex-col gap-4">
        <h1 className="text-lg font-semibold p-4 pb-0" style={{ color: "var(--text-primary)" }}>
          Sessions
        </h1>
        <MobileErrorState message={isApiError(err) ? err.describe() : "Unexpected error loading Sessions."} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg font-semibold p-4 pb-0" style={{ color: "var(--text-primary)" }}>
        Sessions
      </h1>

      {sessions.length === 0 ? (
        <MobileEmptyState icon={Camera} message="No Sessions yet." />
      ) : (
        <div className="flex flex-col gap-2 px-4 pb-4">
          {sessions.map((s) => (
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
