import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { EVIDENCE_TYPE_ICONS, humanizeEvidenceType } from "@/components/ui/EvidenceCard";
import { getEvidence, isApiError } from "@/lib/api";
import type { EvidenceRow } from "@/lib/types";

export default async function EvidenceViewerPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  let evidence: EvidenceRow;
  try {
    evidence = (await getEvidence(id)).row;
  } catch (e) {
    if (isApiError(e) && e.code === "not_found") notFound();
    throw e;
  }

  const Icon = EVIDENCE_TYPE_ICONS[evidence.type];

  return (
    <div className="flex flex-col h-full">
      <div className="flex flex-1 min-h-0">
        <div
          className="flex-1 relative flex flex-col items-center justify-center"
          style={{ background: "var(--bg-base)" }}
        >
          <Link
            href={`/evidence/${evidence.id}`}
            className="absolute top-4 left-4 inline-flex items-center gap-1.5 px-3 h-8 rounded-md text-sm font-medium transition-colors hover:opacity-80"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Back
          </Link>

          <div className="flex flex-col items-center gap-3 text-sm" style={{ color: "var(--text-tertiary)" }}>
            <Icon className="w-12 h-12" />
            <span>No preview available for this item.</span>
          </div>
        </div>

        <aside
          className="w-80 shrink-0 border-l overflow-y-auto p-5 flex flex-col gap-5"
          style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
        >
          <h2 className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>
            {evidence.name}
          </h2>

          <InfoSection title="Timestamps">
            <InfoRow label="Captured" value={evidence.capturedAt ?? "Unavailable"} />
            <InfoRow label="Uploaded" value={evidence.uploadedAt ?? "Unavailable"} />
            <InfoRow label="Processed" value={evidence.processedAt ?? "Unavailable"} />
          </InfoSection>

          <InfoSection title="Location">
            <InfoRow label="Place" value={evidence.location ?? "Unavailable"} />
          </InfoSection>

          <InfoSection title="Session">
            <div className="flex items-center justify-between text-sm">
              <span style={{ color: "var(--text-tertiary)" }}>Session</span>
              {evidence.sessionId ? (
                <Link
                  href={`/sessions/${evidence.sessionId}`}
                  className="font-mono-num hover:underline"
                  style={{ color: "var(--accent)" }}
                >
                  {evidence.sessionName ?? evidence.sessionId}
                </Link>
              ) : (
                <span className="font-mono-num" style={{ color: "var(--text-primary)" }}>
                  Standalone
                </span>
              )}
            </div>
          </InfoSection>

          <InfoSection title="World">
            <div className="flex items-center justify-between text-sm">
              <span style={{ color: "var(--text-tertiary)" }}>World</span>
              {evidence.worldId ? (
                <Link
                  href={`/worlds/${evidence.worldId}`}
                  className="font-mono-num hover:underline"
                  style={{ color: "var(--accent)" }}
                >
                  {evidence.worldName ?? evidence.worldId}
                </Link>
              ) : (
                <span className="font-mono-num" style={{ color: "var(--text-primary)" }}>
                  Unattached
                </span>
              )}
            </div>
          </InfoSection>

          <InfoSection title="Source">
            <InfoRow label="Type" value={humanizeEvidenceType(evidence.type)} />
          </InfoSection>
        </aside>
      </div>

      {(evidence.sessionId || evidence.worldId) && (
        <div
          className="shrink-0 border-t px-6 py-3 flex items-center gap-3"
          style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
        >
          {evidence.sessionId && (
            <Link
              href={`/sessions/${evidence.sessionId}`}
              className="inline-flex items-center px-3 h-7 rounded-md text-xs font-medium hover:underline"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}
            >
              View Session
            </Link>
          )}
          {evidence.worldId && (
            <Link
              href={`/worlds/${evidence.worldId}`}
              className="inline-flex items-center px-3 h-7 rounded-md text-xs font-medium hover:underline"
              style={{ background: "var(--accent-subtle)", color: "var(--accent)" }}
            >
              View World
            </Link>
          )}
        </div>
      )}
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
