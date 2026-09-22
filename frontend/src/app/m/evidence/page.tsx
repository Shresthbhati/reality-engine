import Link from "next/link";
import { ShieldCheck } from "lucide-react";
import { EVIDENCE } from "@/lib/data";

export default function MobileEvidencePage() {
  return (
    <div className="flex flex-col gap-5 p-4">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Evidence
      </h1>

      {EVIDENCE.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-16 px-4 text-center">
          <ShieldCheck className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No Evidence has been added yet.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {EVIDENCE.map((e) => (
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
