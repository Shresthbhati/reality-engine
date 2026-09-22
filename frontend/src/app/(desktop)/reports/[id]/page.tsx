import Link from "next/link";
import { notFound } from "next/navigation";
import PageHeader from "@/components/ui/PageHeader";
import { getReport, getResult } from "@/lib/data";

export default async function ReportDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const report = getReport(id);

  if (!report) {
    notFound();
  }

  const results = report.resultIds
    .map((resultId) => getResult(resultId))
    .filter((r): r is NonNullable<typeof r> => r != null);

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title={report.title}
        action={
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled
              title="Not yet available"
              className="text-sm font-medium px-3 h-8 rounded-md transition-colors disabled:opacity-50"
              style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
            >
              Export
            </button>
            <button
              type="button"
              disabled
              title="Not yet available"
              className="text-sm font-medium px-3 h-8 rounded-md transition-colors disabled:opacity-50"
              style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
            >
              Share
            </button>
            <Link
              href={`/reports/${report.id}/edit`}
              className="text-sm font-medium px-3 h-8 flex items-center rounded-md transition-colors"
              style={{ background: "var(--bg-elevated)", color: "var(--text-secondary)", border: "1px solid var(--border)" }}
            >
              Edit
            </Link>
          </div>
        }
      />

      <div className="flex flex-col gap-6 px-6 py-6 overflow-y-auto">
        <div className="flex flex-col gap-1.5 max-w-md">
          <InfoRow label="Generated" value={report.generatedAt ?? "Unavailable"} />
          <InfoRow label="Updated" value={report.updatedAt ?? "Unavailable"} />
          <InfoRow label="Sessions" value={String(report.sessionCount)} />
        </div>

        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wide mb-3" style={{ color: "var(--text-tertiary)" }}>
            Results
          </h3>
          {results.length === 0 ? (
            <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
              No Results included in this Report.
            </p>
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
      </div>
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
