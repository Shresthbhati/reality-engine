"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, Plus, Trash2, TriangleAlert } from "lucide-react";
import SiteGrid from "@/components/sites/SiteGrid";
import AddWorldDialog from "@/components/sites/AddWorldDialog";
import PlacementInspector from "@/components/sites/PlacementInspector";
import ComposeWorldDialog from "@/components/sites/ComposeWorldDialog";
import AnchorMapPanel from "@/components/sites/AnchorMapPanel";
import { MAP_CONTEXT_LABEL } from "@/lib/sites/compose-model";
import {
  addPlacement,
  deleteSite,
  getSite,
  isApiError,
  removePlacement,
  updatePlacement,
  useSite,
  type Site,
  type UpdatePlacementInput,
} from "@/lib/api";
import { centerM, type PlacementPosition } from "@/lib/sites/site-grid-model";

const errText = (e: unknown) => (isApiError(e) ? e.describe() : (e as Error).message);

type Confirm = { kind: "placement"; id: string } | { kind: "site" } | null;

export default function SiteComposerPage() {
  const params = useParams<{ id: string }>();
  const id = decodeURIComponent(params.id);
  const router = useRouter();
  const { data, isLoading, error, refetch } = useSite(id);

  // `local` holds optimistic edits and post-save server truth; `data` is the first load.
  const [local, setLocal] = useState<Site | null>(null);
  const site = local ?? data;
  const siteRef = useRef<Site | null>(null);
  siteRef.current = site;

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const [composing, setComposing] = useState(false);
  const [confirm, setConfirm] = useState<Confirm>(null);
  const [snap, setSnap] = useState(true);
  const [opError, setOpError] = useState<string | null>(null);
  const [pending, setPending] = useState(0);

  // Mutations run one at a time (rapid arrow-key nudges must not race); the authoritative
  // site (including server-computed overlaps) is re-read once the queue drains.
  const queue = useRef<Promise<void>>(Promise.resolve());
  const inflight = useRef(0);
  const run = useCallback(
    (op: () => Promise<unknown>) => {
      inflight.current += 1;
      setPending(inflight.current);
      queue.current = queue.current
        .then(async () => {
          try {
            await op();
            setOpError(null);
          } catch (e) {
            setOpError(errText(e));
          }
        })
        .then(async () => {
          inflight.current -= 1;
          if (inflight.current === 0) {
            try {
              setLocal(await getSite(id));
            } catch (e) {
              setOpError(errText(e));
            }
          }
          setPending(inflight.current);
        });
    },
    [id],
  );

  useEffect(() => {
    if (selectedId && site && !site.placements.some((p) => p.id === selectedId)) setSelectedId(null);
  }, [site, selectedId]);

  const patch = useCallback(
    (placementId: string, body: UpdatePlacementInput) => {
      const cur = siteRef.current;
      if (cur) {
        // Optimistic: reflect the edit immediately; the re-read replaces it with server values.
        setLocal({
          ...cur,
          placements: cur.placements.map((p) => {
            if (p.id !== placementId) return p;
            const next = { ...p, ...body };
            return { ...next, center_m: centerM(next, cur.grid_cell_m) };
          }),
        });
      }
      run(() => updatePlacement(id, placementId, body));
    },
    [id, run],
  );

  const move = useCallback((pid: string, pos: PlacementPosition) => patch(pid, pos), [patch]);

  const doRemove = (pid: string) => {
    setConfirm(null);
    setSelectedId(null);
    run(() => removePlacement(id, pid));
  };

  if (error && !site) {
    const notFound = isApiError(error) && error.code === "not_found";
    return (
      <div className="p-6">
        <Link href="/sites" className="inline-flex items-center gap-1 text-sm mb-4" style={{ color: "var(--accent)" }}>
          <ArrowLeft className="w-4 h-4" aria-hidden="true" /> Sites
        </Link>
        <div role="alert" className="flex items-start gap-2 rounded-md px-3 py-2 text-sm" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
          <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
          <span className="flex-1">{notFound ? "This site does not exist." : errText(error)}</span>
          {!notFound && <button type="button" className="underline" onClick={refetch}>Retry</button>}
        </div>
      </div>
    );
  }

  if (!site) {
    return <p className="p-6 text-sm" style={{ color: "var(--text-secondary)" }}>{isLoading ? "Loading site…" : "Site unavailable."}</p>;
  }

  const selected = site.placements.find((p) => p.id === selectedId) ?? null;
  const confirmPlacement = confirm?.kind === "placement" ? site.placements.find((p) => p.id === confirm.id) : null;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 sm:px-6 min-h-14 py-2 border-b shrink-0" style={{ borderColor: "var(--border)" }}>
        <Link href="/sites" aria-label="Back to sites" className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-sm font-semibold tracking-wide truncate" style={{ color: "var(--text-primary)" }}>{site.name}</h1>
        <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          {site.placements.length} placed · {site.grid_cell_m} m cells{pending > 0 ? " · saving…" : ""}
        </span>
        <div className="ml-auto flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: "var(--text-secondary)" }}>
            <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} />
            Snap drags
          </label>
          <button type="button" onClick={() => setConfirm({ kind: "site" })} aria-label="Delete site" title="Delete site" className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--error-subtle)]" style={{ color: "var(--error)" }}>
            <Trash2 className="w-4 h-4" />
          </button>
          <button type="button" onClick={() => setComposing(true)} className="text-sm font-medium px-3 h-8 rounded-md border hover:bg-[var(--bg-hover)]" style={{ borderColor: "var(--border)", color: "var(--text-primary)" }}>
            Compose world
          </button>
          <button type="button" onClick={() => setAdding(true)} className="flex items-center gap-1.5 text-sm font-medium px-3 h-8 rounded-md" style={{ background: "var(--accent-subtle)", color: "var(--accent)", border: "1px solid var(--accent-border)" }}>
            <Plus className="w-3.5 h-3.5" aria-hidden="true" />
            Add world
          </button>
        </div>
      </div>

      <p className="px-4 sm:px-6 py-2 text-xs shrink-0" style={{ color: "var(--text-tertiary)" }}>
        You position worlds by hand on this grid; they are not geo-registered automatically. Drag, or select one and use the arrow keys.
      </p>

      {opError && (
        <div role="alert" className="mx-4 sm:mx-6 mb-2 flex items-start gap-2 rounded-md px-3 py-2 text-sm shrink-0" style={{ background: "var(--error-subtle)", color: "var(--error)" }}>
          <TriangleAlert className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
          <span className="flex-1">{opError}</span>
          <button type="button" className="underline" onClick={() => setOpError(null)}>Dismiss</button>
        </div>
      )}

      <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-4 px-4 sm:px-6 pb-4 overflow-y-auto lg:overflow-hidden">
        <div className="flex-1 min-h-[360px] lg:min-h-0 relative">
          <SiteGrid
            gridCellM={site.grid_cell_m}
            placements={site.placements}
            overlaps={site.overlaps}
            selectedId={selectedId}
            snapM={snap ? site.grid_cell_m / 10 : 0}
            onSelect={setSelectedId}
            onMove={move}
            onRemoveRequest={(pid) => setConfirm({ kind: "placement", id: pid })}
          />
        </div>

        <aside aria-label="Placement details" className="lg:w-80 shrink-0 rounded-lg border p-4 lg:overflow-y-auto" style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}>
          {selected ? (
            <PlacementInspector
              placement={selected}
              placements={site.placements}
              overlaps={site.overlaps}
              gridCellM={site.grid_cell_m}
              busy={pending > 0}
              onChange={(body) => patch(selected.id, body)}
              onRemoveRequest={() => setConfirm({ kind: "placement", id: selected.id })}
            />
          ) : (
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--text-secondary)" }}>Placements</h2>
              {site.placements.length === 0 ? (
                <p className="text-sm" style={{ color: "var(--text-secondary)" }}>No worlds placed. Choose “Add world”, pick a world, then a grid cell.</p>
              ) : (
                <ul className="flex flex-col gap-1">
                  {site.placements.map((p) => (
                    <li key={p.id}>
                      <button type="button" onClick={() => setSelectedId(p.id)} className="w-full text-left px-3 py-2 rounded-md border text-sm hover:bg-[var(--bg-hover)]" style={{ borderColor: "var(--border-subtle)", color: "var(--text-primary)" }}>
                        <span className="block truncate font-medium">{p.label}</span>
                        <span className="block text-xs" style={{ color: "var(--text-tertiary)" }}>
                          {p.world_name} · cell {p.cell_x}, {p.cell_z}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <p className="text-xs mt-3" style={{ color: "var(--text-tertiary)" }}>Select a placement on the grid to move, rotate, scale, rename or remove it.</p>
            </div>
          )}
          <div className="mt-4 pt-4 border-t" style={{ borderColor: "var(--border-subtle)" }}>
            <AnchorMapPanel
              site={site}
              onAddMapWorld={async (worldId) => {
                const created = await addPlacement(id, { world_id: worldId, label: MAP_CONTEXT_LABEL, cell_x: 0, cell_z: 0 });
                setLocal(await getSite(id));
                setSelectedId(created.id);
              }}
            />
          </div>
        </aside>
      </div>

      {composing && <ComposeWorldDialog siteId={id} siteName={site.name} placements={site.placements} onClose={() => setComposing(false)} />}

      {adding && (
        <AddWorldDialog
          gridCellM={site.grid_cell_m}
          placements={site.placements}
          onClose={() => setAdding(false)}
          onAdd={async (input) => {
            const created = await addPlacement(id, input);
            setLocal(await getSite(id));
            setSelectedId(created.id);
          }}
        />
      )}

      {confirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "color-mix(in srgb, var(--bg-base) 70%, transparent)" }} onKeyDown={(e) => e.key === "Escape" && setConfirm(null)}>
          <div role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" className="w-full max-w-sm rounded-xl border p-5" style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}>
            <h2 id="confirm-title" className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              {confirm.kind === "site" ? `Delete site “${site.name}”?` : `Remove “${confirmPlacement?.label ?? "placement"}”?`}
            </h2>
            <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
              {confirm.kind === "site"
                ? "All placements on this site are removed. The worlds themselves are not deleted."
                : "It is removed from this site only. The world itself is not deleted."}
            </p>
            <div className="flex justify-end gap-2 mt-5">
              <button type="button" autoFocus onClick={() => setConfirm(null)} className="h-9 px-3 rounded-md text-sm hover:bg-[var(--bg-hover)]" style={{ color: "var(--text-secondary)" }}>
                Cancel
              </button>
              <button
                type="button"
                onClick={async () => {
                  if (confirm.kind === "placement") return doRemove(confirm.id);
                  try {
                    await deleteSite(id);
                    router.push("/sites");
                  } catch (e) {
                    setConfirm(null);
                    setOpError(errText(e));
                  }
                }}
                className="h-9 px-4 rounded-md text-sm font-medium"
                style={{ background: "var(--error-subtle)", color: "var(--error)", border: "1px solid var(--error)" }}
              >
                {confirm.kind === "site" ? "Delete site" : "Remove"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
