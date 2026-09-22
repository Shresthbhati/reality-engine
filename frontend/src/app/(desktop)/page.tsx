import { Camera, Globe, ShieldCheck, FileText } from "lucide-react";
import Link from "next/link";
import EmptyState from "@/components/ui/EmptyState";
import StatusBadge from "@/components/ui/StatusBadge";
import ActivityFeed from "@/components/ui/ActivityFeed";
import { SESSIONS, WORLDS, EVIDENCE, RESULTS } from "@/lib/data";
import { getActivity } from "@/lib/activity";

export default function HomePage() {
  const activeSessions = SESSIONS.filter((s) => s.state === "PROCESSING" || s.state === "QUEUED");
  const recentWorlds = WORLDS.slice(0, 4);
  const recentEvidence = EVIDENCE.slice(0, 4);
  const recentResults = RESULTS.slice(0, 4);
  const activity = getActivity();

  return (
    <div className="flex flex-col">
      <div className="px-6 pt-8 pb-6">
        <h1 className="text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
          Reality Engine
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>
          Place, time, evidence, computation.
        </p>

        <div className="flex items-center gap-2 mt-4">
          <Link
            href="/sessions/new"
            className="text-sm font-medium px-3 h-9 flex items-center rounded-md transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            Create Session
          </Link>
          <Link
            href="/worlds"
            className="text-sm font-medium px-3 h-9 flex items-center rounded-md transition-colors"
            style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Explore Worlds
          </Link>
          <Link
            href="/evidence"
            className="text-sm font-medium px-3 h-9 flex items-center rounded-md transition-colors"
            style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
          >
            Add Evidence
          </Link>
        </div>
      </div>

      <div className="px-6 pb-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
          Active Sessions
        </h2>
      </div>
      {activeSessions.length === 0 ? (
        <EmptyState
          icon={Camera}
          message="No active Sessions. Nothing is running right now — create a Session to start capturing a place."
          actionLabel="Create Session"
          actionHref="/sessions/new"
        />
      ) : (
        <div className="grid gap-3 px-6 pb-6" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))" }}>
          {activeSessions.map((s) => (
            <Link
              key={s.id}
              href={`/sessions/${s.id}`}
              className="flex flex-col gap-2 p-3 rounded-lg"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
            >
              <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {s.name}
              </span>
              <StatusBadge state={s.state} />
            </Link>
          ))}
        </div>
      )}

      <div className="grid gap-6 px-6 pb-8" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <HomeSection title="Recent Worlds" icon={Globe} emptyMessage="No Worlds yet.">
          {recentWorlds.map((w) => (
            <Link key={w.id} href={`/worlds/${w.id}`} className="flex items-center justify-between px-2 py-1.5 rounded-md text-sm" style={{ color: "var(--text-primary)" }}>
              <span className="truncate">{w.name}</span>
              {w.location && <span className="text-xs shrink-0" style={{ color: "var(--text-tertiary)" }}>{w.location}</span>}
            </Link>
          ))}
        </HomeSection>

        <HomeSection title="Recent Evidence" icon={ShieldCheck} emptyMessage="No Evidence has been added yet.">
          {recentEvidence.map((e) => (
            <Link key={e.id} href={`/evidence/${e.id}`} className="flex items-center justify-between px-2 py-1.5 rounded-md text-sm" style={{ color: "var(--text-primary)" }}>
              <span className="truncate">{e.name}</span>
              <span className="text-xs font-mono-num shrink-0" style={{ color: "var(--text-tertiary)" }}>{e.capturedAt ?? "—"}</span>
            </Link>
          ))}
        </HomeSection>

        <HomeSection title="Recent Results" icon={FileText} emptyMessage="Results will appear here after Analysis completes.">
          {recentResults.map((r) => (
            <Link key={r.id} href={`/results/${r.id}`} className="flex items-center justify-between px-2 py-1.5 rounded-md text-sm" style={{ color: "var(--text-primary)" }}>
              <span className="truncate">{r.title}</span>
              <span className="text-xs font-mono-num shrink-0" style={{ color: "var(--text-tertiary)" }}>{r.generatedAt ?? "—"}</span>
            </Link>
          ))}
        </HomeSection>

        <div>
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
              Activity
            </h2>
            <Link href="/activity" className="text-xs font-medium" style={{ color: "var(--accent)" }}>
              View all
            </Link>
          </div>
          <ActivityFeed events={activity} limit={5} />
        </div>
      </div>
    </div>
  );
}

function HomeSection({
  title,
  icon: Icon,
  emptyMessage,
  children,
}: {
  title: string;
  icon: typeof Globe;
  emptyMessage: string;
  children: React.ReactNode;
}) {
  const hasChildren = Array.isArray(children) ? children.length > 0 : Boolean(children);
  return (
    <div>
      <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h2>
      {hasChildren ? (
        <div className="flex flex-col gap-0.5">{children}</div>
      ) : (
        <div className="flex items-center gap-2 text-sm" style={{ color: "var(--text-tertiary)" }}>
          <Icon className="w-4 h-4" />
          {emptyMessage}
        </div>
      )}
    </div>
  );
}
