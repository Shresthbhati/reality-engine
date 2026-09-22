import Link from "next/link";
import { FileText } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import { RESULTS } from "@/lib/data";

export default function ResultsPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Results" />

      {RESULTS.length === 0 ? (
        <EmptyState icon={FileText} message="Results will appear here after Analysis completes." />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 p-6 overflow-y-auto">
          {RESULTS.map((r) => (
            <Link
              key={r.id}
              href={`/results/${r.id}`}
              className="flex flex-col gap-2.5 p-4 rounded-lg transition-colors"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium truncate" style={{ color: "var(--text-primary)" }}>
                  {r.title}
                </span>
                <span className="text-xs font-medium shrink-0" style={{ color: "var(--text-tertiary)" }}>
                  {r.type}
                </span>
              </div>
              {r.finding && (
                <p className="text-sm line-clamp-2" style={{ color: "var(--text-secondary)" }}>
                  {r.finding}
                </p>
              )}
              <div className="flex items-center gap-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
                <span className="font-mono-num">{r.generatedAt ?? "Unavailable"}</span>
                {r.sessionName && <span className="truncate">{r.sessionName}</span>}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
