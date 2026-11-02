"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Map as MapIcon, Plus, TriangleAlert } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";
import { createSite, isApiError, useSites } from "@/lib/api";
import { DEFAULT_GRID_CELL_M, parsePositiveM } from "@/lib/sites/site-grid-model";

const inputStyle = { background: "var(--bg-base)", borderColor: "var(--border)", color: "var(--text-primary)" } as const;

export default function SitesPage() {
  const router = useRouter();
  const { data: sites, isLoading, error, refetch } = useSites();
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [cell, setCell] = useState(String(DEFAULT_GRID_CELL_M));
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);

  const gridCell = parsePositiveM(cell);
  const canCreate = name.trim().length > 0 && gridCell !== null && !busy;

  async function create(e: React.FormEvent) {
    e.preventDefault();
    if (!canCreate || gridCell === null) return;
    setBusy(true);
    setFailure(null);
    try {
      const site = await createSite({ name: name.trim(), grid_cell_m: gridCell });
      router.push(`/sites/${encodeURIComponent(site.id)}`);
    } catch (err) {
      setFailure(isApiError(err) ? err.describe() : (err as Error).message);
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <PageHeader
        title="Sites"
        action={
          <button
            type="button"
            onClick={() => setCreating((v) => !v)}
            aria-expanded={creating}
            className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md transition-colors"
            style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}
          >
            <Plus className="w-3.5 h-3.5" aria-hidden="true" />
            New site
          </button>
        }
      />

      <p className="px-6 pt-3 text-xs" style={{ color: "var(--text-tertiary)" }}>
        A site arranges reconstructed worlds on a metric grid by hand. You place them; nothing is geo-registered automatically.
      </p>

      {creating && (
        <form onSubmit={create} className="mx-6 mt-4 grid gap-3 sm:grid-cols-[1fr_160px_auto] items-end p-4 rounded-lg border" style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}>
          <div>
            <label htmlFor="site-name" className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>Site name</label>
            <input id="site-name" autoFocus value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Agra riverfront" className="w-full h-9 px-3 rounded-md border text-sm" style={inputStyle} />
          </div>
          <div>
            <label htmlFor="site-cell" className="block text-xs mb-1" style={{ color: "var(--text-secondary)" }}>Grid cell (m)</label>
            <input id="site-cell" inputMode="decimal" value={cell} onChange={(e) => setCell(e.target.value)} aria-invalid={gridCell === null} className="w-full h-9 px-3 rounded-md border text-sm" style={{ ...inputStyle, borderColor: gridCell === null ? "var(--error)" : "var(--border)" }} />
          </div>
          <button type="submit" disabled={!canCreate} className="h-9 px-4 rounded-md text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed" style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>
            {busy ? "Creating…" : "Create"}
          </button>
          {failure && (
            <div role="alert" className="sm:col-span-3 text-sm rounded-md px-3 py-2" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
              {failure}
            </div>
          )}
        </form>
      )}

      {error && (
        <div role="alert" className="mx-6 mt-4 flex items-start gap-2 rounded-md px-3 py-2 text-sm" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
          <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
          <span className="flex-1">{isApiError(error) ? error.describe() : error.message}</span>
          <button type="button" className="underline" onClick={refetch}>Retry</button>
        </div>
      )}

      {isLoading && !sites && <p className="px-6 py-6 text-sm" style={{ color: "var(--text-secondary)" }}>Loading sites…</p>}

      {sites && sites.length === 0 && !creating && (
        <EmptyState icon={MapIcon} message="No sites yet. Create one, then place your reconstructed worlds on its grid." />
      )}

      {sites && sites.length > 0 && (
        <ul className="grid gap-4 p-6" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))" }}>
          {sites.map((s) => (
            <li key={s.id}>
              <Link href={`/sites/${encodeURIComponent(s.id)}`} className="block rounded-lg border p-4 hover:bg-[var(--bg-hover)] transition-colors" style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}>
                <p className="text-sm font-semibold truncate" style={{ color: "var(--text-primary)" }}>{s.name}</p>
                <p className="text-xs mt-1" style={{ color: "var(--text-secondary)" }}>
                  {s.placement_count} {s.placement_count === 1 ? "world" : "worlds"} · {s.grid_cell_m} m cells
                </p>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
