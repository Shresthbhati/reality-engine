import Link from "next/link";
import { Globe } from "lucide-react";
import { WORLDS } from "@/lib/data";

export default function MobileWorldsPage() {
  return (
    <div className="flex flex-col gap-5 p-4">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Worlds
      </h1>

      {WORLDS.length === 0 ? (
        <div className="flex flex-col items-center gap-2 py-16 px-4 text-center">
          <Globe className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
          <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
            No Worlds yet.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {WORLDS.map((w) => {
            const metaParts: string[] = [];
            if (w.coverageKm2 !== null) metaParts.push(`${w.coverageKm2} km²`);
            metaParts.push(`${w.sessionCount} Sessions`);
            metaParts.push(`${w.evidenceCount} Evidence`);

            return (
              <Link
                key={w.id}
                href={`/m/worlds/${w.id}`}
                className="flex flex-col justify-center gap-0.5 min-h-[56px] px-4 py-2.5 rounded-md"
                style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
              >
                <span className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
                  {w.name}
                </span>
                {w.location && (
                  <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    {w.location}
                  </span>
                )}
                <span className="text-xs font-mono-num" style={{ color: "var(--text-tertiary)" }}>
                  {metaParts.join(" · ")}
                </span>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
