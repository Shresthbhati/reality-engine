import Link from "next/link";
import { Globe } from "lucide-react";
import { listWorlds, isApiError } from "@/lib/api";
import { MobileEmptyState, MobileErrorState } from "@/components/mobile/AsyncState";

export default async function MobileWorldsPage() {
  let worlds;
  try {
    worlds = (await listWorlds()).rows;
  } catch (err) {
    return (
      <div className="flex flex-col gap-5 p-4">
        <Header />
        <MobileErrorState message={isApiError(err) ? err.describe() : "Unexpected error loading Worlds."} />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 p-4">
      <Header />

      {worlds.length === 0 ? (
        <MobileEmptyState icon={Globe} message="No Worlds yet." />
      ) : (
        <div className="flex flex-col gap-2">
          {worlds.map((w) => {
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

function Header() {
  return (
    <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
      Worlds
    </h1>
  );
}
