import Link from "next/link";
import { notFound } from "next/navigation";
import { MapPin } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import WorldMap from "@/components/map/WorldMap";
import { getResult, getSession, getAnalysis, evidenceForSession } from "@/lib/data";
import type { ResultRow, SessionRow, AnalysisRow } from "@/lib/types";

export default async function ResultDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const result = getResult(id);

  if (!result) {
    notFound();
  }

  const hasLocation = result.lat != null && result.lng != null;
  const session = result.sessionId ? getSession(result.sessionId) : undefined;
  const analysis = result.analysisId ? getAnalysis(result.analysisId) : undefined;
  const worldId = hasLocation && session?.worldId ? session.worldId : null;

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title={result.title}
        action={
          worldId ? (
            <Link
              href={`/worlds/${worldId}`}
              className="text-sm font-medium px-3 h-8 flex items-center rounded-md transition-colors"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
            >
              View in World
            </Link>
          ) : undefined
        }
      />

      <div className="flex flex-1 min-h-0">
        <div className="flex-1 flex flex-col min-h-0">
          <h3
            className="text-xs font-semibold uppercase tracking-wide px-6 pt-5 pb-3 shrink-0"
            style={{ color: "var(--text-tertiary)" }}
          >
            Result Visualization
          </h3>
          <div className="flex-1 min-h-0">
            {hasLocation ? (
              <WorldMap center={[result.lng as number, result.lat as number]} zoom={13} className="w-full h-full" />
            ) : (
              <div className="w-full h-full flex items-center justify-center" style={{ background: "var(--bg-base)" }}>
                <div className="flex flex-col items-center gap-2 text-sm" style={{ color: "var(--text-tertiary)" }}>
                  <MapPin className="w-6 h-6" />
                  <span>No spatial location recorded for this Result.</span>
                </div>
              </div>
            )}
          </div>
        </div>

        <aside
          className="w-80 shrink-0 border-l overflow-y-auto p-5 flex flex-col gap-5"
          style={{ borderColor: "var(--border)" }}
        >
          <InfoSection title="Finding">
            <p className="text-sm leading-relaxed" style={{ color: "var(--text-primary)" }}>
              {result.finding ?? "Unavailable"}
            </p>
          </InfoSection>

          <InfoSection title="Session">
            <InfoRow
              label="Name"
              value={result.sessionName ?? "Unavailable"}
              href={result.sessionId && result.sessionName ? `/sessions/${result.sessionId}` : undefined}
            />
          </InfoSection>

          <InfoSection title="Analysis">
            <InfoRow
              label="Reference"
              value={result.analysisId ? (analysis?.name ?? "View Analysis") : "Unavailable"}
              href={result.analysisId ? `/analysis/${result.analysisId}` : undefined}
            />
          </InfoSection>

          <InfoSection title="Location">
            <InfoRow label="Place" value={result.location ?? "Unavailable"} />
          </InfoSection>

          <InfoSection title="Generated">
            <InfoRow label="Date" value={result.generatedAt ?? "Unavailable"} />
          </InfoSection>
        </aside>
      </div>

      <div className="shrink-0 border-t px-6 py-4" style={{ borderColor: "var(--border)" }}>
        <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Provenance
        </h3>
        <ProvenanceChain result={result} session={session} analysis={analysis} />
      </div>
    </div>
  );
}

function InfoSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-tertiary)" }}>
        {title}
      </h3>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

function InfoRow({ label, value, href }: { label: string; value: string; href?: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span style={{ color: "var(--text-tertiary)" }}>{label}</span>
      {href ? (
        <Link href={href} className="font-mono-num hover:underline truncate" style={{ color: "var(--accent)" }}>
          {value}
        </Link>
      ) : (
        <span className="font-mono-num truncate" style={{ color: "var(--text-primary)" }}>
          {value}
        </span>
      )}
    </div>
  );
}

// Result -> Analysis -> Session -> Evidence, but only the segments the real
// data chain actually resolves — never a fabricated full chain.
function ProvenanceChain({
  result,
  session,
  analysis,
}: {
  result: ResultRow;
  session: SessionRow | undefined;
  analysis: AnalysisRow | undefined;
}) {
  const segments: { label: string; href?: string }[] = [{ label: result.title }];

  if (result.analysisId) {
    segments.push({ label: analysis?.name ?? result.analysisId, href: `/analysis/${result.analysisId}` });
  }

  if (result.sessionId && result.sessionName) {
    segments.push({ label: result.sessionName, href: `/sessions/${result.sessionId}` });
  }

  if (session) {
    const count = evidenceForSession(session.id).length;
    segments.push({ label: `${count} Evidence item${count === 1 ? "" : "s"}` });
  }

  if (segments.length <= 1) {
    return (
      <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
        No provenance chain recorded for this Result.
      </p>
    );
  }

  return (
    <div className="flex items-center gap-2 flex-wrap text-sm">
      {segments.map((seg, i) => (
        <span key={i} className="flex items-center gap-2">
          {i > 0 && <span style={{ color: "var(--text-tertiary)" }}>→</span>}
          {seg.href ? (
            <Link href={seg.href} className="hover:underline" style={{ color: "var(--accent)" }}>
              {seg.label}
            </Link>
          ) : (
            <span style={{ color: "var(--text-secondary)" }}>{seg.label}</span>
          )}
        </span>
      ))}
    </div>
  );
}
