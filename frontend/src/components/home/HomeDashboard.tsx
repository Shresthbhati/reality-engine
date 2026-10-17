"use client";

import Link from "next/link";
import {
  ArrowRight,
  Box,
  Camera,
  CloudUpload,
  FilePlus2,
  FileText,
  Globe,
  Play,
  ShieldCheck,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { useWorlds, useApi, getStatus } from "@/lib/api";
import { fetchActivity, type ActivityType } from "@/lib/activity";
import { relativeTime } from "@/lib/shell/shell-logic";
import type { WorldRow } from "@/lib/types";

const QUICK_START: Array<{ href: string; label: string; hint: string; icon: LucideIcon }> = [
  { href: "/worlds/new", label: "New World", hint: "Create a world from scratch", icon: FilePlus2 },
  { href: "/reconstruct", label: "Upload Data", hint: "Reconstruct from photographs", icon: CloudUpload },
  { href: "/sessions/new", label: "New Session", hint: "Start a capture session", icon: Camera },
  { href: "/reports/new", label: "Create Report", hint: "Share results and findings", icon: FileText },
];

const ACTIVITY_ICON: Record<ActivityType, LucideIcon> = {
  WORLD: Globe,
  SESSION: Camera,
  EVIDENCE: ShieldCheck,
  ANALYSIS: Box,
  RESULT: Box,
  REPORT: FileText,
};

export default function HomeDashboard() {
  const worlds = useWorlds();
  const status = useApi(getStatus);
  const activity = useApi(() => fetchActivity(6));

  const recent = worlds.data
    ? [...worlds.data].sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? "")).slice(0, 5)
    : null;

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden p-5">
      <div className="grid gap-5 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] max-w-[1600px] mx-auto">
        <div className="flex flex-col gap-5 min-w-0">
          <Hero />
          <RecentWorlds
            worlds={recent}
            loading={worlds.isLoading}
            error={worlds.error?.message ?? null}
            onRetry={worlds.refetch}
          />
          <QuickStart />
        </div>

        <div className="flex flex-col gap-5 min-w-0">
          <Overview
            worlds={worlds.data?.length ?? null}
            worldsError={worlds.error?.message ?? null}
            sessions={status.data?.counts.sessions ?? null}
            evidence={status.data?.counts.evidence ?? null}
            statusLoading={status.isLoading}
            statusError={status.error?.message ?? null}
            onRetry={() => {
              worlds.refetch();
              status.refetch();
            }}
          />
          <RecentActivity
            events={activity.data}
            loading={activity.isLoading}
            error={activity.error?.message ?? null}
            onRetry={activity.refetch}
          />
          <AssistantCard />
        </div>
      </div>
    </div>
  );
}

function Panel({
  title,
  action,
  children,
  className = "",
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section
      aria-label={title}
      className={`rounded-xl p-4 min-w-0 ${className}`}
      style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
    >
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function ViewAll({ href }: { href: string }) {
  return (
    <Link href={href} className="flex items-center gap-1 text-xs font-medium rounded" style={{ color: "var(--accent)" }}>
      View All <ArrowRight className="w-3.5 h-3.5" aria-hidden="true" />
    </Link>
  );
}

/** An API failure: shown as an error (never as an empty list) with a retry. */
function ErrorBlock({ what, message, onRetry }: { what: string; message: string; onRetry: () => void }) {
  return (
    <div role="alert" className="rounded-lg p-3 text-sm" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
      <p>
        {what} could not be loaded. {message}
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-2 px-2.5 h-7 rounded text-xs font-semibold"
        style={{ border: "1px solid var(--error)", color: "var(--error)" }}
      >
        Retry
      </button>
    </div>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
      {children}
    </p>
  );
}

function Hero() {
  return (
    <section
      aria-label="Welcome"
      className="rounded-xl p-8 min-h-[260px] flex flex-col justify-center"
      style={{
        border: "1px solid var(--border)",
        background:
          "radial-gradient(ellipse at 85% 10%, rgba(37,99,235,0.28), transparent 55%), radial-gradient(ellipse at 10% 100%, rgba(34,184,255,0.12), transparent 50%), var(--bg-surface)",
      }}
    >
      <p className="text-[11px] tracking-[0.28em] uppercase mb-3" style={{ color: "var(--text-secondary)" }}>
        Real places. Persistent digital worlds.
      </p>
      <h1 className="text-4xl lg:text-5xl font-bold leading-tight" style={{ color: "var(--text-primary)" }}>
        Turn evidence into
        <span className="block" style={{ color: "var(--accent)" }}>
          a world you can trust
        </span>
      </h1>
      <p className="mt-3 max-w-xl text-sm" style={{ color: "var(--text-secondary)" }}>
        Reality Engine reconstructs what your photographs support, says what remains uncertain, and hands you an
        editable, versioned 3D world.
      </p>
      <div className="flex flex-wrap gap-3 mt-6">
        <Link
          href="/sessions/new"
          className="flex items-center gap-2 h-11 px-5 rounded-lg text-sm font-semibold text-white hover:bg-[var(--primary-hover)] transition-colors"
          style={{ background: "var(--primary)" }}
        >
          <Play className="w-4 h-4" aria-hidden="true" /> Create new session <ArrowRight className="w-4 h-4" aria-hidden="true" />
        </Link>
        <Link
          href="/worlds"
          className="flex items-center gap-2 h-11 px-5 rounded-lg text-sm font-semibold hover:bg-[var(--bg-hover)] transition-colors"
          style={{ border: "1px solid var(--border)", color: "var(--text-primary)", background: "var(--bg-elevated)" }}
        >
          <Globe className="w-4 h-4" aria-hidden="true" /> Explore worlds
        </Link>
      </div>
    </section>
  );
}

function RecentWorlds({
  worlds,
  loading,
  error,
  onRetry,
}: {
  worlds: WorldRow[] | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <Panel title="Recent Worlds" action={<ViewAll href="/worlds" />}>
      {error ? (
        <ErrorBlock what="Worlds" message={error} onRetry={onRetry} />
      ) : loading || worlds === null ? (
        <Muted>Loading worlds…</Muted>
      ) : worlds.length === 0 ? (
        <div className="flex flex-col items-start gap-3">
          <Muted>No worlds yet. Create one to see it here.</Muted>
          <Link
            href="/worlds/new"
            className="px-3 h-8 inline-flex items-center rounded-md text-sm font-semibold text-white"
            style={{ background: "var(--primary)" }}
          >
            New World
          </Link>
        </div>
      ) : (
        <ul className="grid grid-cols-2 xl:grid-cols-5 gap-3">
          {worlds.map((w) => {
            const when = relativeTime(w.updatedAt);
            return (
              <li key={w.id} className="min-w-0">
                <Link
                  href={`/worlds/${encodeURIComponent(w.id)}`}
                  className="block rounded-lg overflow-hidden hover:border-[var(--accent)] transition-colors"
                  style={{ border: "1px solid var(--border)", background: "var(--bg-elevated)" }}
                >
                  {/* No preview is rendered for a world yet; an honest placeholder, never a stock image. */}
                  <div
                    className="h-20 flex flex-col items-center justify-center gap-1"
                    style={{ background: "var(--bg-base)", color: "var(--text-tertiary)" }}
                  >
                    <Box className="w-5 h-5" aria-hidden="true" />
                    <span className="text-[11px]">No preview yet</span>
                  </div>
                  <div className="p-2.5">
                    <p className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>
                      {w.name}
                    </p>
                    <p className="text-xs truncate" style={{ color: "var(--text-secondary)" }}>
                      {when ? `Updated ${when}` : "Update time unknown"}
                    </p>
                    <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                      {w.sessionCount} {w.sessionCount === 1 ? "session" : "sessions"}
                    </p>
                  </div>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

function QuickStart() {
  return (
    <Panel title="Quick Start">
      <ul className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        {QUICK_START.map(({ href, label, hint, icon: Icon }) => (
          <li key={href} className="min-w-0">
            <Link
              href={href}
              className="h-full flex flex-col items-center gap-1.5 text-center rounded-lg p-3 hover:border-[var(--accent)] transition-colors"
              style={{ border: "1px solid var(--border)", background: "var(--bg-elevated)" }}
            >
              <Icon className="w-5 h-5" style={{ color: "var(--accent)" }} aria-hidden="true" />
              <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                {label}
              </span>
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                {hint}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function Overview(props: {
  worlds: number | null;
  worldsError: string | null;
  sessions: number | null;
  evidence: number | null;
  statusLoading: boolean;
  statusError: string | null;
  onRetry: () => void;
}) {
  const error = props.worldsError ?? props.statusError;
  const stat = (label: string, value: number | null) => (
    <div className="min-w-0">
      <p className="text-2xl font-semibold font-mono-num" style={{ color: "var(--text-primary)" }}>
        {value === null ? "—" : value.toLocaleString("en-US")}
      </p>
      <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
        {label}
      </p>
    </div>
  );
  return (
    <Panel title="Overview">
      {error ? (
        <ErrorBlock what="Counts" message={error} onRetry={props.onRetry} />
      ) : props.statusLoading ? (
        <Muted>Loading counts…</Muted>
      ) : (
        <div className="grid grid-cols-3 gap-3">
          {stat("Worlds", props.worlds)}
          {stat("Sessions", props.sessions)}
          {stat("Evidence items", props.evidence)}
        </div>
      )}
    </Panel>
  );
}

function RecentActivity({
  events,
  loading,
  error,
  onRetry,
}: {
  events: Awaited<ReturnType<typeof fetchActivity>> | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <Panel title="Recent Activity" action={<ViewAll href="/activity" />}>
      {error ? (
        <ErrorBlock what="Activity" message={error} onRetry={onRetry} />
      ) : loading || events === null ? (
        <Muted>Loading activity…</Muted>
      ) : events.length === 0 ? (
        <Muted>Nothing has happened yet. Activity appears here as you work.</Muted>
      ) : (
        <ul className="flex flex-col">
          {events.map((e) => {
            const Icon = ACTIVITY_ICON[e.type];
            const when = relativeTime(e.timestamp);
            return (
              <li key={e.id} className="border-b last:border-b-0" style={{ borderColor: "var(--border-subtle)" }}>
                <Link href={e.href} className="flex items-center gap-2.5 py-2 min-w-0 hover:opacity-90">
                  <span
                    className="w-6 h-6 rounded-md flex items-center justify-center shrink-0"
                    style={{ background: "var(--purple-subtle)", color: "var(--purple)" }}
                  >
                    <Icon className="w-3.5 h-3.5" aria-hidden="true" />
                  </span>
                  <span className="flex-1 min-w-0 text-sm truncate" style={{ color: "var(--text-primary)" }}>
                    {e.label}
                  </span>
                  {when && (
                    <span className="text-xs shrink-0" style={{ color: "var(--text-tertiary)" }}>
                      {when}
                    </span>
                  )}
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </Panel>
  );
}

/** The assistant does not exist yet: every control is disabled and says so. */
function AssistantCard() {
  return (
    <Panel
      title="AI Assistant"
      action={
        <span
          className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
          style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}
        >
          Coming soon
        </span>
      }
    >
      <div className="flex items-start gap-2 mb-3">
        <Sparkles className="w-4 h-4 mt-0.5 shrink-0" style={{ color: "var(--accent)" }} aria-hidden="true" />
        <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
          Ask questions, analyze data, or get insights about your world. Not available yet.
        </p>
      </div>
      <input
        disabled
        aria-label="Ask the assistant (not available yet)"
        placeholder="Ask anything…"
        title="Not available yet"
        className="w-full h-9 px-3 rounded-md text-sm cursor-not-allowed opacity-60"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
      />
    </Panel>
  );
}
