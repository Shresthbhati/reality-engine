import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import WorldMap from "@/components/map/WorldMap";
import { getSession } from "@/lib/data";

export default async function SessionMapPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const session = getSession(id);
  if (!session) notFound();

  const hasCoverage = session.lat != null && session.lng != null;

  return (
    <div className="relative h-full w-full">
      <WorldMap
        center={hasCoverage ? [session.lng!, session.lat!] : undefined}
        zoom={hasCoverage ? 14 : undefined}
      />

      <div className="absolute top-4 left-4 flex items-center gap-2 z-10">
        <Link
          href={`/sessions/${session.id}`}
          className="flex items-center gap-1.5 h-9 px-3 rounded-md text-sm font-medium backdrop-blur-sm"
          style={{ background: "rgba(14,16,19,0.9)", color: "var(--text-primary)", border: "1px solid var(--border)" }}
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          {session.name}
        </Link>
      </div>

      {!hasCoverage && (
        <div
          className="absolute bottom-4 left-4 right-4 z-10 flex items-center justify-center h-10 rounded-md text-sm backdrop-blur-sm"
          style={{ background: "rgba(14,16,19,0.9)", color: "var(--text-tertiary)", border: "1px solid var(--border)" }}
        >
          No coverage geometry recorded for this Session — showing the base map only.
        </div>
      )}
    </div>
  );
}
