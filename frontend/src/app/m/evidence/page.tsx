import Link from "next/link";
import { ShieldCheck } from "lucide-react";
import { listEvidence, isApiError } from "@/lib/api";
import { MobileEmptyState, MobileErrorState } from "@/components/mobile/AsyncState";

export default async function MobileEvidencePage({
  searchParams,
}: {
  searchParams: Promise<{ session?: string }>;
}) {
  const { session: sessionId } = await searchParams;

  let evidence;
  try {
    evidence = (await listEvidence({ sessionId })).rows;
  } catch (err) {
    return (
      <div className="flex flex-col gap-5 p-4">
        <Header />
        <MobileErrorState message={isApiError(err) ? err.describe() : "Unexpected error loading Evidence."} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 p-4">
      <Header />

      {evidence.length === 0 ? (
        <MobileEmptyState icon={ShieldCheck} message="No Evidence has been added yet." />
      ) : (
        <div className="flex flex-col gap-2">
          {evidence.map((e) => (
            <Link
              key={e.id}
              href={`/m/evidence/${e.id}`}
              className="flex flex-col justify-center gap-0.5 min-h-[56px] px-4 py-2.5 rounded-md"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
            >
              <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                {e.name}
              </span>
              {(e.location || e.sessionName) && (
                <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                  {[e.location, e.sessionName].filter(Boolean).join(" · ")}
                </span>
              )}
              {e.capturedAt && (
                <span className="text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                  Captured · {e.capturedAt}
                </span>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function Header() {
  return (
    <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
      Evidence
    </h1>
  );
}
