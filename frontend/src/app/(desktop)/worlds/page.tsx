import Link from "next/link";
import { Globe, Plus, TriangleAlert } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import WorldCard from "@/components/ui/WorldCard";
import { isApiError, listWorlds, type WorldRow } from "@/lib/api";

export default async function WorldsPage() {
  let worlds: WorldRow[] = [];
  let failure: string | null = null;
  try {
    worlds = (await listWorlds()).rows;
  } catch (err) {
    failure = isApiError(err) ? err.describe() : (err as Error).message;
  }

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Worlds"
        action={
          <Link
            href="/worlds/new"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" />
            Create World
          </Link>
        }
      />

      {failure && (
        <div
          className="mx-6 mt-4 flex items-start gap-2 rounded-md px-3 py-2 text-sm"
          style={{ background: "var(--error-subtle)", color: "var(--error)" }}
          role="alert"
        >
          <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" />
          <span>{failure}</span>
        </div>
      )}

      {worlds.length === 0 && !failure ? (
        <EmptyState icon={Globe} message="No Worlds yet." actionLabel="Create World" actionHref="/worlds/new" />
      ) : (
        <div className="grid gap-4 p-6" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
          {worlds.map((w) => (
            <WorldCard key={w.id} world={w} />
          ))}
        </div>
      )}
    </div>
  );
}

