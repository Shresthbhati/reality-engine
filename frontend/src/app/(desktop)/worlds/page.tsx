import Link from "next/link";
import { Globe, Plus } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import WorldCard from "@/components/ui/WorldCard";
import { WORLDS } from "@/lib/data";

export default function WorldsPage() {
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

      {WORLDS.length === 0 ? (
        <EmptyState icon={Globe} message="No Worlds yet." actionLabel="Create World" actionHref="/worlds/new" />
      ) : (
        <div className="grid gap-4 p-6" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
          {WORLDS.map((w) => (
            <WorldCard key={w.id} world={w} />
          ))}
        </div>
      )}
    </div>
  );
}
