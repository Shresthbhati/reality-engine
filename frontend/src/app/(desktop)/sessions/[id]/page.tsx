import Link from "next/link";
import { notFound } from "next/navigation";
import { Map } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import StatusBadge from "@/components/ui/StatusBadge";
import SessionTimeline from "@/components/ui/SessionTimeline";
import { getSession, listEvidence, isApiError, unsupportedForAnalysis } from "@/lib/api";

export default async function SessionDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  let session;
  try { session = (await getSession(id)).row; }
  catch (e) { if (isApiError(e) && e.code === "not_found") notFound(); throw e; }

  let evidenceList: { id: string; name: string; type: string; processingState?: string }[] = [];
  let evidenceError: string | null = null;
  try { evidenceList = (await listEvidence({ sessionId: session.id })).rows; }
  catch (e) { evidenceError = isApiError(e) ? e.describe() : String(e); }

  // Analysis API is not yet supported — surface an unsupported resource empty state.
  const analysisResource = unsupportedForAnalysis();
  const analysis: never[] = [];
  const results: never[] = [];

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title={session.name}
        action={
          <div className="flex items-center gap-2">
            <StatusBadge state={session.state} />
            <Link
              href={`/sessions/${session.id}/map`}
              className="text-sm font-medium px-3 h-8 flex items-center rounded-md transition-colors"
              style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
            >
              Open Map
            </Link>
          </div>
        }
      />

      <div className="flex flex-1 min-h-0">
        <div className="flex-1 flex items-center justify-center" style={{ background: "var(--bg-base)" }}>
          {session.location ? (
            <div className="flex flex-col items-center gap-2 text-sm" style={{ color: "var(--text-tertiary)" }}>
              <Map className="w-6 h-6" />
              <span>Map preview unavailable — no coverage geometry recorded.</span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 text-sm" style={{ color: "var(--text-tertiary)" }}>
              <Map className="w-6 h-6" />
              <span>No location recorded for this Session.</span>
            </div>
          )}
        </div>

        <aside
          className="w-80 shrink-0 border-l overflow-y-auto p-5 flex flex-col gap-5"
          style={{ borderColor: "var(--border)" }}
        >
          <InfoSection title="Time">
            <InfoRow label="Captured" value={session.capturedAt ?? "Unavailable"} />
            <InfoRow
              label="Duration"
              value={session.durationSec != null ? `${Math.round(session.durationSec / 60)}m` : "Unavailable"}
            />
          </InfoSection>

          <InfoSection title="Location">
            <InfoRow label="Place" value={session.location ?? "Unavailable"} />
          </InfoSection>

          <InfoSection title="World">
            <InfoRow label="Relationship" value={session.worldName ?? "Standalone"} />
          </InfoSection>

          <InfoSection title="Evidence">
            <InfoRow label="Items" value={String(evidenceList.length)} />
          </InfoSection>
        </aside>
      </div>

      <div
        className="shrink-0 border-t px-6 py-4"
        style={{ borderColor: "var(--border)" }}
      >
        <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Timeline
        </h3>
        <SessionTimeline stages={session.stages} />
      </div>

      <div className="shrink-0 border-t px-6 py-4" style={{ borderColor: "var(--border)" }}>
        <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Evidence
        </h3>
        {evidenceList.length === 0 || evidenceError ? (
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            {evidenceError ?? "No Evidence recorded for this Session."}
          </p>
        ) : (
          <div className="flex flex-col gap-1.5">
            {evidenceList.map((e) => (
              <Link
                key={e.id}
                href={`/evidence/${e.id}`}
                className="flex items-center justify-between px-3 py-2 rounded-md text-sm transition-colors hover:opacity-80"
                style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)" }}
              >
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                  {e.name}
                </span>
                <span className="flex items-center gap-3">
                  <span style={{ color: "var(--text-tertiary)" }}>{e.type}</span>
                  <span style={{ color: "var(--text-secondary)" }}>{e.processingState}</span>
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>

      <div className="shrink-0 border-t px-6 py-4" style={{ borderColor: "var(--border)" }}>
        <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Analysis
        </h3>
        <EmptyState
          icon={FileText}
          message={analysisResource.reason}
          actionLabel="Unsupported"
        />
      </div>

      <div className="shrink-0 border-t px-6 py-4" style={{ borderColor: "var(--border)" }}>
        <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Results
        </h3>
        {results.length === 0 ? (
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No Results yet.
          </p>
        ) : (
          <div className="flex flex-col gap-1.5">
            {results.map((r) => (
              <Link
                key={r.id}
                href={`/results/${r.id}`}
                className="flex items-center justify-between px-3 py-2 rounded-md text-sm transition-colors hover:opacity-80"
                style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)" }}
              >
                <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                  {r.title}
                </span>
                <span className="font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                  {r.generatedAt ?? "Unavailable"}
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>

      <div className="shrink-0 border-t px-6 py-4" style={{ borderColor: "var(--border)" }}>
        <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
          Provenance
        </h3>
        <ProvenanceChain evidenceCount={evidence.length} analysisCount={analysis.length} resultCount={results.length} />
      </div>
    </div>
  );
}

function ProvenanceChain({
  evidenceCount,
  analysisCount,
  resultCount,
}: {
  evidenceCount: number;
  analysisCount: number;
  resultCount: number;
}) {
  // Capture -> Session is always real for any Session that exists (this page
  // 404s otherwise), so the chain always has at least those two nodes.
  const nodes = ["Capture", "Session"];
  if (evidenceCount > 0) nodes.push("Evidence");
  if (analysisCount > 0) nodes.push("Analysis");
  if (resultCount > 0) nodes.push("Result");

  return (
    <div className="flex items-center flex-wrap gap-2">
      {nodes.map((node, i) => (
        <div key={node} className="flex items-center gap-2">
          <span
            className="px-3 py-1.5 rounded-full text-sm font-medium"
            style={{ background: "var(--bg-elevated)", color: "var(--text-primary)", border: "1px solid var(--border-subtle)" }}
          >
            {node}
          </span>
          {i < nodes.length - 1 && (
            <span style={{ color: "var(--text-tertiary)" }}>&rarr;</span>
          )}
        </div>
      ))}
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

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span style={{ color: "var(--text-tertiary)" }}>{label}</span>
      <span className="font-mono-num" style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}
