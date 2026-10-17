"use client";

import { AlertTriangle, Clock, Cpu, HardDrive, Loader2, MemoryStick } from "lucide-react";
import type { ComponentType } from "react";
import { toProfileRows, type ProfileRow, type ResourceProfileDto } from "@/lib/intent/model";

const ICONS: Record<ProfileRow["key"], ComponentType<{ className?: string }>> = {
  compute: Cpu,
  memory: MemoryStick,
  storage: HardDrive,
  time: Clock,
};

const KIND_LABEL = { detected: "Detected", derived: "Derived", unknown: "Unknown" } as const;

/** Expected Resource Profile. Unknown values say Unknown and why; an error is an error, never an empty panel. */
export default function ResourceProfile({
  profile,
  loading,
  error,
}: {
  profile: ResourceProfileDto | null;
  loading: boolean;
  error: string | null;
}) {
  return (
    <section aria-label="Expected Resource Profile" aria-busy={loading}>
      <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
        Expected Resource Profile
        {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" aria-label="Updating" />}
      </h3>

      {error ? (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-md px-3 py-2 text-xs"
          style={{ background: "var(--error-subtle)", color: "var(--error)" }}
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>Could not compute the resource profile: {error}</span>
        </div>
      ) : profile === null ? (
        <p className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          {loading ? "Checking this machine…" : "Choose a scale to see what this machine can support."}
        </p>
      ) : (
        <>
          <dl className="flex flex-col">
            {toProfileRows(profile).map((r) => {
              const Icon = ICONS[r.key];
              return (
                <div
                  key={r.key}
                  className="flex items-start gap-3 py-2"
                  style={{ borderTop: "1px solid var(--border-subtle)" }}
                >
                  <Icon className="mt-0.5 h-4 w-4 shrink-0" />
                  <dt className="w-28 shrink-0 text-xs" style={{ color: "var(--text-secondary)" }}>
                    {r.label}
                  </dt>
                  <dd className="min-w-0 flex-1 text-xs">
                    <span
                      className="font-medium"
                      style={{ color: r.known ? "var(--text-primary)" : "var(--text-tertiary)" }}
                    >
                      {r.text}
                    </span>
                    <span className="ml-1.5 text-[10px] uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
                      {KIND_LABEL[r.kind]}
                    </span>
                    <span className="mt-0.5 block" style={{ color: "var(--text-tertiary)" }}>
                      {r.note}
                    </span>
                  </dd>
                </div>
              );
            })}
          </dl>
          {profile.warnings.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1.5" aria-label="Warnings">
              {profile.warnings.map((w) => (
                <li
                  key={w}
                  className="flex items-start gap-2 rounded-md px-2.5 py-2 text-xs"
                  style={{ background: "var(--warning-subtle)", color: "var(--warning)" }}
                >
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}
