import Link from "next/link";
import { notFound } from "next/navigation";
import PageHeader from "@/components/ui/PageHeader";
import { EVIDENCE_TYPE_ICONS, humanizeEvidenceType, EvidenceStatusPill } from "@/components/ui/EvidenceCard";
import { getEvidence, isApiError, evidenceArtifactUrl } from "@/lib/api";
import type { EvidenceRow } from "@/lib/types";

export default async function EvidenceDetailPage({
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
      <PageHeader
        title={evidence.name}
        action={
          <div className="flex items-center gap-2">
            <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
              {humanizeEvidenceType(evidence.type)}
            </span>
            <EvidenceStatusPill state={evidence.processingState} />
          </div>
        }
      />

      <div className="flex flex-1 min-h-0">
        <Link
          href={`/evidence/${evidence.id}/view`}
          className="flex-1 flex items-center justify-center overflow-hidden"
          style={{ background: "var(--bg-base)" }}
        >
          {evidence.type === "IMAGE" ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={evidenceArtifactUrl(evidence.id)}
              alt={evidence.name}
              className="max-h-full max-w-full object-contain"
            />
          ) : (
            <div className="flex flex-col items-center gap-2 text-sm" style={{ color: "var(--text-tertiary)" }}>
              <Icon className="w-8 h-8" />
              <span>No inline preview for {humanizeEvidenceType(evidence.type).toLowerCase()} items.</span>
            </div>
          )}
        </Link>

        <aside
          className="w-80 shrink-0 border-l overflow-y-auto p-5 flex flex-col gap-5"
          style={{ borderColor: "var(--border)" }}
        >
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
            <InfoRow label="Relationship" value={evidence.worldName ?? "Unattached"} />
          </InfoSection>
        </aside>
      </div>

      {evidence.sessionId && (
        <div className="shrink-0 border-t px-6 py-3" style={{ borderColor: "var(--border)" }}>
          <Link
            href={`/sessions/${evidence.sessionId}`}
            className="text-sm font-medium hover:underline"
            style={{ color: "var(--accent)" }}
          >
            View Session →
          </Link>
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
