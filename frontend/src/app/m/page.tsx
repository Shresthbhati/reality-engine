import Link from "next/link";
import { Camera, Bell } from "lucide-react";
import StatusBadge from "@/components/ui/StatusBadge";
import { MobileErrorState } from "@/components/mobile/AsyncState";
import { listSessions, listEvidence, isApiError } from "@/lib/api";

function greeting(): string {
  // Hour is read at request time; this is presentation only, not a
  // Date.now()-dependent computation the app depends on for correctness.
  const hour = new Date().getHours();
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

export default async function MobileHomePage() {
  let active: Awaited<ReturnType<typeof listSessions>>["rows"] = [];
  let recentEvidence: Awaited<ReturnType<typeof listEvidence>>["rows"] = [];
  let loadError: string | null = null;

  try {
    const [sessions, evidence] = await Promise.all([listSessions(), listEvidence()]);
    active = sessions.rows.filter((s) => s.state === "PROCESSING" || s.state === "QUEUED");
    recentEvidence = evidence.rows.slice(0, 3);
  } catch (err) {
    loadError = isApiError(err) ? err.describe() : "Unexpected error loading your workspace.";
  }

  return (
    <div className="flex flex-col gap-6 p-4">
      <div>
        <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
          {greeting()}
        </p>
        <h1 className="text-lg font-semibold mt-0.5" style={{ color: "var(--text-primary)" }}>
          Reality Engine
        </h1>
      </div>

      <Link
        href="/m/camera"
        className="flex items-center justify-center gap-2 h-12 rounded-lg text-sm font-semibold"
        style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
      >
        <Camera className="w-4 h-4" />
        Open Camera
      </Link>

      {loadError ? (
        <MobileErrorState message={loadError} />
      ) : (
        <>
          <section>
            <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
              Active Sessions
            </h2>
            {active.length === 0 ? (
              <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
                Nothing is running right now.
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                {active.map((s) => (
                  <Link
                    key={s.id}
                    href={`/m/sessions/${s.id}`}
                    className="flex items-center justify-between px-3 h-12 rounded-md"
                    style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
                  >
                    <span className="text-sm" style={{ color: "var(--text-primary)" }}>
                      {s.name}
                    </span>
                    <StatusBadge state={s.state} />
                  </Link>
                ))}
              </div>
            )}
          </section>

          <section>
            <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
              Recent Captures
            </h2>
            {recentEvidence.length === 0 ? (
              <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
                No captures yet.
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                {recentEvidence.map((e) => (
                  <Link
                    key={e.id}
                    href={`/m/evidence/${e.id}`}
                    className="flex items-center justify-between px-3 h-12 rounded-md"
                    style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
                  >
                    <span className="text-sm" style={{ color: "var(--text-primary)" }}>
                      {e.name}
                    </span>
                    <span className="text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                      {e.capturedAt ?? "—"}
                    </span>
                  </Link>
                ))}
              </div>
            )}
          </section>
        </>
      )}

      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
          Notifications
        </h2>
        <div className="flex items-center gap-2 text-sm" style={{ color: "var(--text-tertiary)" }}>
          <Bell className="w-4 h-4" />
          No notifications yet.
        </div>
      </section>
    </div>
  );
}
