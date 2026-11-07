"use client";

import Link from "next/link";
import { AlertTriangle, ArrowRight, Box, CheckCircle2, CircleAlert, Globe, Loader2, Play } from "lucide-react";
import { useWorlds, useApi, listJobs } from "@/lib/api";
import { attentionItems, type AttentionItem, type AttentionKind } from "@/lib/dashboard/attention-model";
import { relativeTime } from "@/lib/shell/shell-logic";
import type { WorldRow } from "@/lib/types";

/**
 * Home (plan U.4): one primary action, one secondary, what needs attention, and the worlds you were last in.
 * Everything shown is real backend state; there is no placeholder content.
 */
export default function HomeDashboard() {
  const worlds = useWorlds();
  const jobs = useApi(listJobs);

  const recent = worlds.data
    ? [...worlds.data].sort((a, b) => (b.updatedAt ?? "").localeCompare(a.updatedAt ?? "")).slice(0, 6)
    : null;
  const attention = jobs.data ? attentionItems(jobs.data) : null;

  return (
    <div className="h-full overflow-y-auto overflow-x-hidden p-4 sm:p-6">
      <div className="mx-auto flex max-w-4xl min-w-0 flex-col gap-6">
        <Header />
        <NeedsAttention
          items={attention}
          loading={jobs.isLoading}
          error={jobs.error?.message ?? null}
          onRetry={jobs.refetch}
        />
        <RecentWorlds
          worlds={recent}
          loading={worlds.isLoading}
          error={worlds.error?.message ?? null}
          onRetry={worlds.refetch}
        />
      </div>
    </div>
  );
}

function Header() {
  return (
    <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <h1 className="text-2xl font-bold text-fg sm:text-3xl">Reality Engine</h1>
        <p className="mt-1 text-sm text-fg-muted">Capture a place, see what it shows, and work with the world.</p>
      </div>
      <div className="flex flex-col gap-2 sm:flex-row sm:shrink-0">
        <Link
          href="/reconstruct"
          className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-primary px-5 text-sm font-semibold text-on-primary transition-colors hover:bg-primary-hover"
        >
          <Play className="h-4 w-4" aria-hidden="true" /> New session
        </Link>
        <Link
          href="/worlds"
          className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-line bg-elevated px-5 text-sm font-semibold text-fg transition-colors hover:bg-hover"
        >
          <Globe className="h-4 w-4" aria-hidden="true" /> Open world
        </Link>
      </div>
    </header>
  );
}

function Panel({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section aria-label={title} className="min-w-0 rounded-xl border border-line bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-base font-semibold text-fg">{title}</h2>
        {action}
      </div>
      {children}
    </section>
  );
}

/** An API failure: shown as an error (never as an empty list) with a retry. */
function ErrorBlock({ what, message, onRetry }: { what: string; message: string; onRetry: () => void }) {
  return (
    <div role="alert" className="rounded-lg bg-error-subtle p-3 text-sm text-error">
      <p>
        {what} could not be loaded. {message}
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-2 h-8 rounded border border-error px-3 text-xs font-semibold text-error"
      >
        Retry
      </button>
    </div>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-fg-muted">{children}</p>;
}

const KIND_STYLE: Record<AttentionKind, { icon: typeof AlertTriangle; color: string; spin?: boolean }> = {
  failed: { icon: AlertTriangle, color: "text-error" },
  partial: { icon: CircleAlert, color: "text-warning" },
  running: { icon: Loader2, color: "text-accent", spin: true },
};

function NeedsAttention({
  items,
  loading,
  error,
  onRetry,
}: {
  items: AttentionItem[] | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <Panel title="Needs attention">
      {error ? (
        <ErrorBlock what="Jobs" message={error} onRetry={onRetry} />
      ) : loading || items === null ? (
        <Muted>Checking…</Muted>
      ) : items.length === 0 ? (
        <p className="flex items-center gap-2 text-sm text-fg-muted">
          <CheckCircle2 className="h-4 w-4 text-success" aria-hidden="true" /> Nothing needs your attention.
        </p>
      ) : (
        <ul className="flex flex-col divide-y divide-line-subtle">
          {items.map((it) => (
            <li key={it.id}>
              <AttentionRow item={it} />
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function AttentionRow({ item }: { item: AttentionItem }) {
  const { icon: Icon, color, spin } = KIND_STYLE[item.kind];
  const when = relativeTime(item.at);
  const body = (
    <div className="flex items-start gap-3 py-2.5">
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${color} ${spin ? "animate-spin" : ""}`} aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-fg">{item.title}</p>
        {item.detail && <p className="line-clamp-2 break-words text-xs text-fg-muted">{item.detail}</p>}
      </div>
      {when && <span className="shrink-0 text-xs text-fg-subtle">{when}</span>}
      {item.href && <ArrowRight className="mt-0.5 h-4 w-4 shrink-0 text-fg-subtle" aria-hidden="true" />}
    </div>
  );
  return item.href ? (
    <Link href={item.href} className="block rounded transition-colors hover:bg-hover">
      {body}
    </Link>
  ) : (
    body
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
    <Panel
      title="Recent Worlds"
      action={
        <Link href="/worlds" className="flex items-center gap-1 rounded text-xs font-medium text-accent">
          All worlds <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
        </Link>
      }
    >
      {error ? (
        <ErrorBlock what="Worlds" message={error} onRetry={onRetry} />
      ) : loading || worlds === null ? (
        <Muted>Loading worlds…</Muted>
      ) : worlds.length === 0 ? (
        <div className="flex flex-col items-start gap-3">
          <Muted>No worlds yet. Start a session to create your first one.</Muted>
          <Link
            href="/reconstruct"
            className="inline-flex h-9 items-center rounded-md bg-primary px-3 text-sm font-semibold text-on-primary hover:bg-primary-hover"
          >
            New session
          </Link>
        </div>
      ) : (
        <ul className="grid grid-cols-1 gap-3 min-[460px]:grid-cols-2 lg:grid-cols-3">
          {worlds.map((w) => {
            const when = relativeTime(w.updatedAt);
            return (
              <li key={w.id} className="min-w-0">
                <Link
                  href={`/worlds/${encodeURIComponent(w.id)}`}
                  className="block overflow-hidden rounded-lg border border-line bg-elevated transition-colors hover:border-accent"
                >
                  <div className="flex h-20 flex-col items-center justify-center gap-1 bg-base text-fg-subtle">
                    <Box className="h-5 w-5" aria-hidden="true" />
                    <span className="text-[11px]">No preview yet</span>
                  </div>
                  <div className="p-2.5">
                    <p className="truncate text-sm font-semibold text-fg">{w.name}</p>
                    <p className="truncate text-xs text-fg-muted">{when ? `Updated ${when}` : "Update time unknown"}</p>
                    <p className="text-xs text-fg-subtle">
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
