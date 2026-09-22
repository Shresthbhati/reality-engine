import Link from "next/link";
import { notFound } from "next/navigation";
import { FileText } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import StatusBadge from "@/components/ui/StatusBadge";
import EmptyState from "@/components/ui/EmptyState";
import type { AnalysisRow } from "@/lib/types";
import { getAnalysisDetail, getResults, isApiError, type ResultRow } from "@/lib/api";

const STAGES = ["INPUTS", "CONFIGURATION", "VALIDATION", "RUN", "SESSION", "RESULT"] as const;

export default async function AnalysisDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const { row: analysis, resource } = await getAnalysisDetail(id);

  if (!analysis) {
    return (
      <div className="flex flex-col h-full">
        <PageHeader title="Analysis" />
        <div className="flex-1 flex items-center justify-center px-6 py-6">
          <EmptyState icon={FileText} message={resource.reason} />
        </div>
      </div>
    );
  }

  const { items: results } = await getResults();

  return (
    <div className="flex flex-col h-full">
      <PageHeader title={analysis.name} action={<StatusBadge state={analysis.state} />} />

      <LifecycleStrip analysis={analysis} />

      <div className="flex flex-1 min-h-0">
        <div className="flex-1 flex flex-col min-h-0 overflow-y-auto px-6 py-6">
          <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
            Results
          </h3>
          {results.length === 0 ? (
            <EmptyState icon={FileText} message="No Results yet." />
          ) : (
            <div className="flex flex-col gap-1.5">
              {results.map((r) => (
                <Link
                  key={r.id}
                  href={`/results/${r.id}`}
                  className="flex items-center justify-between px-3 h-9 rounded-md text-sm transition-colors"
                  style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
                >
                  <span>{r.title}</span>
                  <span className="font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                    {r.generatedAt ?? "Unavailable"}
                  </span>
                </Link>
              ))}
            </div>
          )}
        </div>

        <aside
          className="w-80 shrink-0 border-l overflow-y-auto p-5 flex flex-col gap-5"
          style={{ borderColor: "var(--border)" }}
        >
          <InfoSection title="Session">
            {analysis.sessionId ? (
              <div className="flex items-center justify-between text-sm">
                <span style={{ color: "var(--text-tertiary)" }}>Session</span>
                <Link
                  href={`/sessions/${analysis.sessionId}`}
                  className="font-mono-num hover:underline"
                  style={{ color: "var(--accent)" }}
                >
                  {analysis.sessionName ?? analysis.sessionId}
                </Link>
              </div>
            ) : (
              <InfoRow label="Session" value="No session" />
            )}
          </InfoSection>

          <InfoSection title="Time">
            <InfoRow label="Started" value={analysis.startedAt ?? "Unavailable"} />
            <InfoRow label="Completed" value={analysis.completedAt ?? "Unavailable"} />
          </InfoSection>

          <InfoSection title="Owner">
            <InfoRow label="Owner" value={analysis.owner ?? "Unassigned"} />
          </InfoSection>
        </aside>
      </div>
    </div>
  );
}

function stageIndex(analysis: AnalysisRow): number {
  if (analysis.state === "QUEUED") return 0;
  if (analysis.state === "COMPLETE") return STAGES.length - 1;
  // PROCESSING or FAILED — use the reported stage when we have one, otherwise
  // fall back to RUN, the point at which either state is legitimately reached.
  const fromStage = analysis.currentStage
    ? STAGES.findIndex((s) => s.toLowerCase() === analysis.currentStage!.toLowerCase())
    : -1;
  return fromStage >= 0 ? fromStage : 3;
}

function LifecycleStrip({ analysis }: { analysis: AnalysisRow }) {
  const active = stageIndex(analysis);

  return (
    <div
      className="flex items-center gap-2 px-6 py-4 border-b overflow-x-auto shrink-0"
      style={{ borderColor: "var(--border)" }}
    >
      {STAGES.map((stage, i) => {
        const reached = i <= active || analysis.state === "COMPLETE";
        const isActive = i === active && analysis.state !== "COMPLETE";
        const isFailed = isActive && analysis.state === "FAILED";
        const dotColor = isFailed ? "var(--error)" : reached ? "var(--accent)" : "var(--border)";
        return (
          <div key={stage} className="flex items-center gap-2">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: dotColor }} />
              <span
                className="text-xs font-medium tracking-wide uppercase whitespace-nowrap"
                style={{ color: reached ? "var(--text-primary)" : "var(--text-tertiary)" }}
              >
                {stage}
              </span>
            </div>
            {i < STAGES.length - 1 && (
              <span className="w-6 h-px shrink-0" style={{ background: "var(--border)" }} />
            )}
          </div>
        );
      })}
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
