import Link from "next/link";
import { Globe, Plus, TriangleAlert, Archive, Trash2, CheckCircle2 } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import WorldCard from "@/components/ui/WorldCard";
import { isApiError, listWorlds, type WorldRow } from "@/lib/api";

export default async function WorldsPage({
  searchParams,
}: {
  searchParams: Promise<{ status?: string }>;
}) {
  const { status: rawStatus } = await searchParams;
  const status = rawStatus === "archived" || rawStatus === "trash" ? rawStatus : "active";

  let worlds: WorldRow[] = [];
  let failure: string | null = null;
  try {
    worlds = (await listWorlds(status)).rows;
  } catch (err) {
    failure = isApiError(err) ? err.describe() : (err as Error).message;
  }

  const tabs: Array<{ id: "active" | "archived" | "trash"; label: string; href: string }> = [
    { id: "active", label: "Active Worlds", href: "/worlds" },
    { id: "archived", label: "Archived", href: "/worlds?status=archived" },
    { id: "trash", label: "Trash", href: "/worlds?status=trash" },
  ];

  return (
    <div className="flex flex-col h-full">
      <PageHeader
        title="Worlds"
        action={
          <Link
            href="/worlds/new"
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{
              background: "var(--accent-subtle)",
              color: "var(--accent)",
              border: "1px solid var(--accent-border)",
            }}
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Create World</span>
          </Link>
        }
      />

      {/* Status Filter Tabs */}
      <div className="flex items-center gap-2 px-6 pt-4 border-b border-zinc-800">
        {tabs.map((tab) => {
          const isActive = status === tab.id;
          return (
            <Link
              key={tab.id}
              href={tab.href}
              className={`px-3 py-1.5 text-xs font-medium rounded-t-md transition-colors border-b-2 -mb-px ${
                isActive
                  ? "border-indigo-500 text-zinc-100 bg-zinc-800/40"
                  : "border-transparent text-zinc-400 hover:text-zinc-200 hover:border-zinc-700"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>

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
        <EmptyState
          icon={status === "trash" ? Trash2 : status === "archived" ? Archive : Globe}
          message={
            status === "trash"
              ? "Trash is empty."
              : status === "archived"
              ? "No archived worlds."
              : "No Worlds yet."
          }
          actionLabel={status === "active" ? "Create World" : undefined}
          actionHref={status === "active" ? "/worlds/new" : undefined}
        />
      ) : (
        <div
          className="grid gap-4 p-6"
          style={{ gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))" }}
        >
          {worlds.map((w) => (
            <WorldCard key={w.id} world={w} statusView={status} />
          ))}
        </div>
      )}
    </div>
  );
}
