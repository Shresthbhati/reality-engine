import Link from "next/link";
import type { WorldRow } from "@/lib/types";

export default function WorldInspector({ world }: { world: WorldRow }) {
  return (
    <aside className="w-80 shrink-0 border-l overflow-y-auto p-5 flex flex-col gap-5" style={{ borderColor: "var(--border)" }}>
      <div>
        <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
          {world.name}
        </h2>
        {world.location && (
          <p className="text-xs mt-0.5" style={{ color: "var(--text-secondary)" }}>
            {world.location}
          </p>
        )}
      </div>

      <Section title="Coverage">
        <Row label="Area" value={world.coverageKm2 != null ? `${world.coverageKm2} km²` : "Unavailable"} />
        <Row label="Sessions" value={String(world.sessionCount)} />
        <Row label="Evidence" value={String(world.evidenceCount)} />
      </Section>

      <Section title="Time Range">
        <Row
          label="Coverage window"
          value={world.timeRangeStart && world.timeRangeEnd ? `${world.timeRangeStart} – ${world.timeRangeEnd}` : "Unavailable"}
        />
        <Row label="Last updated" value={world.updatedAt ?? "Unavailable"} />
      </Section>

      <div className="flex flex-col gap-2 mt-1">
        <Link
          href="/sessions/new"
          className="text-sm font-medium h-8 flex items-center justify-center rounded-md transition-colors"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          Add Session
        </Link>
      </div>
    </aside>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h3>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span style={{ color: "var(--text-tertiary)" }}>{label}</span>
      <span className="font-mono-num" style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}
