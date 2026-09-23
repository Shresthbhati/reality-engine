"use client";

import Link from "next/link";
import {
  Globe,
  Plus,
  Camera,
  ShieldCheck,
  Settings,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { useWorlds, useSessions } from "@/lib/api";
import StatusBadge from "@/components/ui/StatusBadge";
import { cn } from "@/lib/utils";

function formatDate(val: string | null): string {
  if (!val) return "—";
  try {
    const d = new Date(val);
    if (!isNaN(d.getTime())) {
      return d.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
      });
    }
  } catch {
    // fallback to raw string
  }
  return val;
}

export default function HomePage() {
  const {
    data: worlds,
    isLoading: worldsLoading,
    error: worldsError,
    refetch: refetchWorlds,
  } = useWorlds();

  const {
    data: sessions,
    isLoading: sessionsLoading,
    error: sessionsError,
    refetch: refetchSessions,
  } = useSessions();

  const recentSessions = (sessions || []).slice(0, 5);

  return (
    <div
      className="h-full overflow-y-auto"
      style={{ background: "var(--bg-base)" }}
    >
      <div className="max-w-5xl mx-auto p-8 flex flex-col gap-8">
        {/* 1. Header Section */}
        <header
          className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 pb-6 border-b"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2.5">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{
                  background: "var(--accent)",
                  boxShadow: "0 0 10px var(--accent)",
                }}
              />
              <h1
                className="text-2xl font-bold tracking-tight"
                style={{ color: "var(--text-primary)" }}
              >
                Reality Engine
              </h1>
            </div>
            <p
              className="text-sm"
              style={{ color: "var(--text-secondary)" }}
            >
              Persistent Spatial Workstation
            </p>
          </div>

          <div
            className="flex items-center gap-2 font-mono-num text-xs"
            style={{ color: "var(--text-tertiary)" }}
          >
            <span
              className="px-2 py-0.5 rounded border"
              style={{
                borderColor: "var(--border)",
                background: "var(--bg-surface)",
              }}
            >
              CORE V1
            </span>
            <span
              className="px-2 py-0.5 rounded border"
              style={{
                borderColor: "var(--border)",
                background: "var(--bg-surface)",
              }}
            >
              RUNTIME ACTIVE
            </span>
          </div>
        </header>

        {/* 2. Your Worlds Grid */}
        <section className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2
              className="text-xs font-semibold uppercase tracking-wider"
              style={{ color: "var(--text-tertiary)" }}
            >
              Your Worlds
            </h2>
            {worlds && worlds.length > 0 && (
              <Link
                href="/worlds"
                className="text-xs transition-colors hover:underline"
                style={{ color: "var(--accent)" }}
              >
                View all ({worlds.length})
              </Link>
            )}
          </div>

          {worldsLoading ? (
            <div
              className="grid gap-4"
              style={{
                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              }}
            >
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="animate-pulse rounded-lg border p-4 h-[132px] flex flex-col justify-between"
                  style={{
                    background: "var(--bg-surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <div className="h-4 w-32 rounded bg-[var(--bg-elevated)]" />
                      <div className="h-2 w-2 rounded-full bg-[var(--bg-elevated)]" />
                    </div>
                    <div className="h-3 w-20 rounded bg-[var(--bg-elevated)]" />
                  </div>
                  <div
                    className="flex items-center justify-between pt-3 border-t"
                    style={{ borderColor: "var(--border-subtle)" }}
                  >
                    <div className="h-3 w-28 rounded bg-[var(--bg-elevated)]" />
                    <div className="h-3 w-16 rounded bg-[var(--bg-elevated)]" />
                  </div>
                </div>
              ))}
            </div>
          ) : worldsError ? (
            <div
              className="flex flex-col sm:flex-row items-center justify-between gap-4 rounded-lg border p-4"
              style={{
                background: "var(--bg-surface)",
                borderColor: "var(--border)",
              }}
            >
              <div className="flex items-center gap-3">
                <AlertCircle
                  className="w-5 h-5 shrink-0"
                  style={{ color: "var(--error)" }}
                />
                <span
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Unable to load worlds. Check that the backend is running.
                </span>
              </div>
              <button
                type="button"
                onClick={() => refetchWorlds()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer"
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border)",
                }}
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Retry
              </button>
            </div>
          ) : !worlds || worlds.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center"
              style={{
                borderColor: "var(--border)",
                background: "var(--bg-surface)",
              }}
            >
              <Globe
                className="w-8 h-8 mb-3"
                style={{ color: "var(--text-tertiary)" }}
              />
              <p
                className="text-sm font-medium mb-1"
                style={{ color: "var(--text-primary)" }}
              >
                Create your first World to begin
              </p>
              <p
                className="text-xs mb-4"
                style={{ color: "var(--text-tertiary)" }}
              >
                Initialize a spatial coordinate frame to aggregate sessions and 3D evidence.
              </p>
              <Link
                href="/worlds/new"
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md text-xs font-medium transition-colors"
                style={{
                  background: "var(--accent-subtle)",
                  color: "var(--accent)",
                  border: "1px solid var(--accent-border)",
                }}
              >
                <Plus className="w-3.5 h-3.5" />
                Create World
              </Link>
            </div>
          ) : (
            <div
              className="grid gap-4"
              style={{
                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              }}
            >
              {worlds.map((world) => (
                <Link
                  key={world.id}
                  href={`/worlds/${world.id}`}
                  className={cn(
                    "flex flex-col justify-between rounded-lg border p-4 transition-colors duration-150",
                    "border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--accent-border)]",
                  )}
                >
                  <div className="flex flex-col gap-1.5">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <Globe
                          className="w-4 h-4 shrink-0"
                          style={{ color: "var(--text-tertiary)" }}
                        />
                        <span
                          className="text-sm font-semibold truncate"
                          style={{ color: "var(--text-primary)" }}
                        >
                          {world.name}
                        </span>
                      </div>
                      {world.currentVersionId && (
                        <span
                          className="w-2 h-2 rounded-full shrink-0 mt-1"
                          style={{
                            background: "var(--accent)",
                            boxShadow: "0 0 6px var(--accent)",
                          }}
                          title="3D spatial data available"
                        />
                      )}
                    </div>
                    <span
                      className="text-xs truncate"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {world.location || "No location set"}
                    </span>
                  </div>

                  <div
                    className="flex items-center justify-between text-xs mt-4 pt-3 border-t font-mono-num"
                    style={{
                      borderColor: "var(--border-subtle)",
                      color: "var(--text-secondary)",
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <span>
                        {world.sessionCount}{" "}
                        {world.sessionCount === 1 ? "Session" : "Sessions"}
                      </span>
                      <span style={{ color: "var(--text-tertiary)" }}>•</span>
                      <span>{world.evidenceCount} Evidence</span>
                    </div>
                    <span
                      className="text-[11px]"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      {world.updatedAt ? formatDate(world.updatedAt) : "—"}
                    </span>
                  </div>
                </Link>
              ))}

              {/* + Create World Card */}
              <Link
                href="/worlds/new"
                className={cn(
                  "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-4 min-h-[132px] transition-colors duration-150 group",
                  "border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--accent-border)]",
                )}
              >
                <div
                  className="w-8 h-8 rounded-full flex items-center justify-center transition-transform group-hover:scale-105"
                  style={{
                    background: "var(--accent-subtle)",
                    color: "var(--accent)",
                    border: "1px solid var(--accent-border)",
                  }}
                >
                  <Plus className="w-4 h-4" />
                </div>
                <span
                  className="text-xs font-medium"
                  style={{ color: "var(--text-secondary)" }}
                >
                  + Create World
                </span>
              </Link>
            </div>
          )}
        </section>

        {/* 3. Recent Sessions Section */}
        <section className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2
              className="text-xs font-semibold uppercase tracking-wider"
              style={{ color: "var(--text-tertiary)" }}
            >
              Recent Sessions
            </h2>
            {sessions && sessions.length > 0 && (
              <Link
                href="/sessions"
                className="text-xs transition-colors hover:underline"
                style={{ color: "var(--accent)" }}
              >
                View all ({sessions.length})
              </Link>
            )}
          </div>

          {sessionsLoading ? (
            <div className="flex flex-col gap-2">
              {[0, 1, 2].map((i) => (
                <div
                  key={i}
                  className="animate-pulse rounded-lg border px-4 py-3 h-12 flex items-center justify-between"
                  style={{
                    background: "var(--bg-surface)",
                    borderColor: "var(--border)",
                  }}
                >
                  <div className="h-4 w-40 rounded bg-[var(--bg-elevated)]" />
                  <div className="flex items-center gap-4">
                    <div className="h-3 w-20 rounded bg-[var(--bg-elevated)] hidden sm:block" />
                    <div className="h-5 w-20 rounded bg-[var(--bg-elevated)]" />
                    <div className="h-3 w-16 rounded bg-[var(--bg-elevated)]" />
                  </div>
                </div>
              ))}
            </div>
          ) : sessionsError ? (
            <div
              className="flex items-center justify-between gap-4 rounded-lg border p-4"
              style={{
                background: "var(--bg-surface)",
                borderColor: "var(--border)",
              }}
            >
              <div className="flex items-center gap-3">
                <AlertCircle
                  className="w-5 h-5 shrink-0"
                  style={{ color: "var(--error)" }}
                />
                <span
                  className="text-sm"
                  style={{ color: "var(--text-secondary)" }}
                >
                  Unable to load sessions.
                </span>
              </div>
              <button
                type="button"
                onClick={() => refetchSessions()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors cursor-pointer"
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border)",
                }}
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Retry
              </button>
            </div>
          ) : recentSessions.length === 0 ? (
            <div
              className="rounded-lg border p-6 text-center text-sm"
              style={{
                background: "var(--bg-surface)",
                borderColor: "var(--border)",
                color: "var(--text-tertiary)",
              }}
            >
              No sessions yet
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {recentSessions.map((session) => (
                <Link
                  key={session.id}
                  href={`/sessions/${session.id}`}
                  className={cn(
                    "flex items-center justify-between gap-4 px-4 py-3 rounded-lg border transition-colors duration-150",
                    "border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--accent-border)]",
                  )}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <Camera
                      className="w-4 h-4 shrink-0"
                      style={{ color: "var(--text-tertiary)" }}
                    />
                    <span
                      className="text-sm font-medium truncate"
                      style={{ color: "var(--text-primary)" }}
                    >
                      {session.name}
                    </span>
                  </div>

                  <div className="flex items-center gap-4 shrink-0">
                    <span
                      className="text-xs font-mono-num hidden sm:inline"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {session.evidenceCount}{" "}
                      {session.evidenceCount === 1 ? "evidence" : "evidence"}
                    </span>
                    <StatusBadge state={session.state} />
                    <span
                      className="text-xs font-mono-num"
                      style={{ color: "var(--text-tertiary)" }}
                    >
                      {formatDate(session.capturedAt)}
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>

        {/* 4. Quick Actions Row */}
        <section className="flex flex-col gap-3">
          <h2
            className="text-xs font-semibold uppercase tracking-wider"
            style={{ color: "var(--text-tertiary)" }}
          >
            Quick Actions
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Link
              href="/sessions/new"
              className={cn(
                "flex items-center gap-3.5 p-4 rounded-lg border transition-colors duration-150 group",
                "border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--accent-border)]",
              )}
            >
              <div
                className="w-9 h-9 rounded-md flex items-center justify-center shrink-0 transition-colors"
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--accent)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <Camera className="w-4 h-4" />
              </div>
              <div className="flex flex-col min-w-0">
                <span
                  className="text-sm font-medium truncate"
                  style={{ color: "var(--text-primary)" }}
                >
                  Create Session
                </span>
                <span
                  className="text-xs truncate"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Ingest reality telemetry
                </span>
              </div>
            </Link>

            <Link
              href="/evidence"
              className={cn(
                "flex items-center gap-3.5 p-4 rounded-lg border transition-colors duration-150 group",
                "border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--accent-border)]",
              )}
            >
              <div
                className="w-9 h-9 rounded-md flex items-center justify-center shrink-0 transition-colors"
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--accent)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div className="flex flex-col min-w-0">
                <span
                  className="text-sm font-medium truncate"
                  style={{ color: "var(--text-primary)" }}
                >
                  View Evidence
                </span>
                <span
                  className="text-xs truncate"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  Browse spatial assets
                </span>
              </div>
            </Link>

            <Link
              href="/settings"
              className={cn(
                "flex items-center gap-3.5 p-4 rounded-lg border transition-colors duration-150 group",
                "border-[var(--border)] bg-[var(--bg-surface)] hover:border-[var(--accent-border)]",
              )}
            >
              <div
                className="w-9 h-9 rounded-md flex items-center justify-center shrink-0 transition-colors"
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--accent)",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                <Settings className="w-4 h-4" />
              </div>
              <div className="flex flex-col min-w-0">
                <span
                  className="text-sm font-medium truncate"
                  style={{ color: "var(--text-primary)" }}
                >
                  Settings
                </span>
                <span
                  className="text-xs truncate"
                  style={{ color: "var(--text-tertiary)" }}
                >
                  System & runtime configuration
                </span>
              </div>
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
}
