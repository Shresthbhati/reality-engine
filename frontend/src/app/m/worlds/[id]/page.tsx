import Link from "next/link";
import { notFound } from "next/navigation";
import { Camera, List } from "lucide-react";
import WorldMap from "@/components/map/WorldMap";
import DesktopHandoff from "@/components/mobile/DesktopHandoff";
import { getWorld } from "@/lib/data";

export default async function MobileWorldDetailPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const world = getWorld(id);

  if (!world) {
    notFound();
  }

  const hasCoords = world.lat !== null && world.lng !== null;
  const hasTimeRange = world.timeRangeStart !== null || world.timeRangeEnd !== null;

  return (
    <div className="flex flex-col gap-5 p-4">
      <div>
        <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
          {world.name}
        </h1>
        <p className="text-sm mt-0.5" style={{ color: world.location ? "var(--text-secondary)" : "var(--text-tertiary)" }}>
          {world.location ?? "Unavailable"}
        </p>
      </div>

      {hasCoords && (
        <div style={{ height: 180 }} className="rounded-lg overflow-hidden">
          <WorldMap center={[world.lng as number, world.lat as number]} />
        </div>
      )}

      <section className="flex flex-col gap-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
          Metrics
        </h2>
        <Row label="Coverage" value={world.coverageKm2 !== null ? `${world.coverageKm2} km²` : "Unavailable"} />
        <Row label="Sessions" value={String(world.sessionCount)} />
        <Row label="Evidence" value={String(world.evidenceCount)} />
        {hasTimeRange && (
          <Row label="Time Range" value={`${world.timeRangeStart ?? "Unavailable"} – ${world.timeRangeEnd ?? "Unavailable"}`} />
        )}
      </section>

      <div className="flex flex-col gap-3">
        <Link
          href={`/m/camera?world=${world.id}`}
          className="flex items-center justify-center gap-2 h-12 rounded-lg text-sm font-semibold"
          style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
        >
          <Camera className="w-4 h-4" />
          Capture Here
        </Link>
        <Link
          href="/m/sessions"
          className="flex items-center justify-center gap-2 h-11 rounded-md text-sm font-medium"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
        >
          <List className="w-4 h-4" />
          View Sessions
        </Link>
      </div>

      <DesktopHandoff href={`/worlds/${world.id}`} />
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
        {label}
      </span>
      <span className="text-sm font-mono-num" style={{ color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}
