import Link from "next/link";
import { notFound } from "next/navigation";
import { ChevronRight } from "lucide-react";
import StatusBadge from "@/components/ui/StatusBadge";
import DesktopHandoff from "@/components/mobile/DesktopHandoff";
import ReconstructButton from "@/components/mobile/ReconstructButton";
import { MobileErrorState } from "@/components/mobile/AsyncState";
import { getSession, getSessionJobs, isApiError } from "@/lib/api";

export default async function MobileSessionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let session;
  try {
    session = (await getSession(id)).row;
  } catch (err) {
    if (isApiError(err) && err.code === "not_found") notFound();
    return (
      <div className="flex flex-col gap-5 p-4">
        <MobileErrorState message={isApiError(err) ? err.describe() : "Unexpected error loading this Session."} />
      </div>
    );
  }

  const jobs = await getSessionJobs(id).catch(() => []);
  const latestJob = jobs[0] ?? null;

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
        <div className="flex items-center justify-between">
          <Row label="Items" value={String(session.evidenceCount)} mono />
        </div>
        <Link
          href={`/m/evidence?session=${session.id}`}
          className="text-sm font-medium min-h-[36px] flex items-center"
          style={{ color: "var(--accent)" }}
        >
          Review Evidence →
        </Link>
      </Section>

      <Section title="Processing">
        {latestJob ? (
          <>
            <Row label="Job" value={latestJob.type} />
            <Row label="Status" value={latestJob.stage ? `${latestJob.status} · ${latestJob.stage}` : latestJob.status} />
            {latestJob.error && (
              <p className="text-xs" style={{ color: "var(--error)" }}>
                {latestJob.error}
              </p>
            )}
          </>
        ) : (
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No reconstruction job has run yet.
          </p>
        )}
        <ReconstructButton sessionId={session.id} />
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

      <DesktopHandoff
        href={`/sessions/${session.id}`}
        worldName={session.worldName}
        sessionName={session.name}
        pendingWork={latestJob && latestJob.status !== "COMPLETE" ? `Reconstruction ${latestJob.status.toLowerCase()}` : null}
      />
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
