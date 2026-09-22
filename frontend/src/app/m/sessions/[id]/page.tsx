import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronRight } from "lucide-react";
import StatusBadge from "@/components/ui/StatusBadge";
import DesktopHandoff from "@/components/mobile/DesktopHandoff";
import { getSession } from "@/lib/data";

export default async function MobileSessionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const session = getSession(id);

  if (!session) {
    notFound();
  }

  return (
    <div className="flex flex-col gap-5 p-4">
      <div className="flex items-center justify-between gap-2">
        <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
          {session.name}
        </h1>
        <StatusBadge state={session.state} />
      </div>

      <Section title="Location">
        <Row label="Place" value={session.location ?? "Unavailable"} />
        {session.lat != null && session.lng != null && (
          <Row label="Coordinates" value={`${session.lat.toFixed(5)}, ${session.lng.toFixed(5)}`} mono />
        )}
      </Section>

      <Section title="Time">
        <Row label="Started" value={session.capturedAt ?? "Unavailable"} mono />
        <Row
          label="Duration"
          value={session.durationSec != null ? `${Math.round(session.durationSec / 60)}m` : "Unavailable"}
          mono
        />
      </Section>

      {session.coverageKm2 != null && (
        <Section title="Coverage">
          <Row label="Area" value={`${session.coverageKm2} km²`} mono />
        </Section>
      )}

      <Section title="Evidence">
        <Row label="Items" value={String(session.evidenceCount)} mono />
      </Section>

      <Section title="World">
        {session.worldId ? (
          <Link
            href={`/m/worlds/${session.worldId}`}
            className="flex items-center justify-between text-sm min-h-[44px]"
            style={{ color: "var(--accent)" }}
          >
            {session.worldName ?? session.worldId}
            <ChevronRight className="w-4 h-4" />
          </Link>
        ) : (
          <Row label="Relationship" value="Standalone" />
        )}
      </Section>

      <DesktopHandoff href={`/sessions/${session.id}`} />
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h2>
      {children}
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between text-sm min-h-[28px]">
      <span style={{ color: "var(--text-tertiary)" }}>{label}</span>
      <span className={mono ? "font-mono-num" : undefined} style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}
