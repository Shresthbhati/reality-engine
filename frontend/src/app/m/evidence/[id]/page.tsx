import Link from "next/link";
import { notFound } from "next/navigation";
import DesktopHandoff from "@/components/mobile/DesktopHandoff";
import DeleteEvidenceButton from "@/components/mobile/DeleteEvidenceButton";
import { MobileErrorState } from "@/components/mobile/AsyncState";
import { EVIDENCE_TYPE_ICONS, humanizeEvidenceType, EvidenceStatusPill } from "@/components/ui/EvidenceCard";
import { getEvidence, evidenceArtifactUrl, isApiError } from "@/lib/api";

export default async function MobileEvidenceDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  let evidence;
  try {
    evidence = (await getEvidence(id)).row;
  } catch (err) {
    if (isApiError(err) && err.code === "not_found") notFound();
    return (
      <div className="flex flex-col gap-5 p-4">
        <MobileErrorState message={isApiError(err) ? err.describe() : "Unexpected error loading this Evidence item."} />
      </div>
    );
  }

  const Icon = EVIDENCE_TYPE_ICONS[evidence.type];
  const canPreviewImage = evidence.type === "IMAGE" && evidence.processingState !== "FAILED";

  return (
    <div className="flex flex-col gap-5 p-4">
      <div>
        <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
          {evidence.name}
        </h1>
        <p className="text-sm mt-0.5" style={{ color: "var(--text-secondary)" }}>
          {humanizeEvidenceType(evidence.type)}
        </p>
      </div>

      {/* Spec §67: DesktopHandoff is this screen's primary action — placed
          right under the header, not buried at the bottom like other screens. */}
      <DesktopHandoff
        href={`/evidence/${evidence.id}`}
        label="Open on Desktop"
        worldName={evidence.worldName}
        sessionName={evidence.sessionName}
      />

      {canPreviewImage ? (
        // eslint-disable-next-line @next/next/no-img-element -- remote artifact, not a static asset
        <img
          src={evidenceArtifactUrl(evidence.id)}
          alt={evidence.name}
          className="w-full rounded-lg object-cover"
          style={{ height: 200, background: "var(--bg-elevated)" }}
        />
      ) : (
        <div
          className="flex flex-col items-center justify-center gap-2 rounded-lg"
          style={{ height: 160, background: "var(--bg-elevated)" }}
        >
          <Icon className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
          <span className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No preview available
          </span>
        </div>
      )}

      <section className="flex flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Details
        </h2>

        <Row label="Captured" value={evidence.capturedAt ?? "Unavailable"} mono />
        <Row label="Location" value={evidence.location ?? "Unavailable"} />

        <div className="flex items-center justify-between">
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Session
          </span>
          {evidence.sessionId ? (
            <Link href={`/m/sessions/${evidence.sessionId}`} className="text-sm font-medium" style={{ color: "var(--accent)" }}>
              {evidence.sessionName ?? evidence.sessionId}
            </Link>
          ) : (
            <span className="text-sm" style={{ color: "var(--text-primary)" }}>
              Unavailable
            </span>
          )}
        </div>

        <Row label="World" value={evidence.worldName ?? "Unavailable"} />

        <div className="flex items-center justify-between">
          <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Processing State
          </span>
          <EvidenceStatusPill state={evidence.processingState} />
        </div>
      </section>

      <DeleteEvidenceButton evidenceId={evidence.id} />
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
        {label}
      </span>
      <span className={`text-sm ${mono ? "font-mono-num" : ""}`} style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}
