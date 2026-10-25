"use client";

import { useCallback, useState } from "react";
import { createPortal } from "react-dom";
import { Layers, Loader2, Trash2, X } from "lucide-react";
import { isApiError } from "@/lib/api/client";
import { deleteGenerated, listGenerated, updateGenerated, type GeneratedKind } from "@/lib/api/procedural";
import {
  H_MAX,
  H_MIN,
  MAX_UI_STOREYS,
  apartmentEditFrom,
  apartmentPatch,
  buildingForm,
  buildingPatch,
  cityFormFrom,
  corridorFormFrom,
  dependencyLines,
  hasOpeningErrors,
  parseDependents,
  resizeHeights,
  resultLine,
  roomFormFrom,
  summarize,
  validateApartmentEdit,
  validateBuildingForm,
  validateOpenings,
  type ApartmentEditForm,
  type BuildingForm,
  type CityForm,
  type CorridorForm,
  type GeneratedItem,
  type RoomForm,
} from "@/lib/procedural/generated-model";
import {
  districtEditFrom,
  districtPatch,
  validateDistrictEdit,
  type DistrictEditForm,
} from "@/lib/procedural/district-model";
import OpeningsEditor from "@/components/workspace/OpeningsEditor";

const inputCls =
  "w-full bg-transparent outline-none text-sm px-2 h-8 rounded border border-[var(--border)] text-[var(--text-primary,#fff)]";

function errText(err: unknown, fallback: string): string {
  return isApiError(err) ? err.message : fallback;
}

/** Toolbar entry + dialog listing every stored generated object. Select one to edit it
 * (PUT -> new version) or delete it. `onChanged` must refetch the WorldIR. */
export default function GeneratedPanel({ worldId, onChanged }: { worldId: string; onChanged: () => void }) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<GeneratedItem[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoadError(null);
    try {
      setItems(await listGenerated(worldId));
    } catch (err) {
      setLoadError(errText(err, "Could not load generated objects."));
    }
  }, [worldId]);

  const current = items?.find((i) => i.entity_id === selected) ?? null;

  return (
    <>
      <button
        type="button"
        onClick={() => {
          setOpen(true);
          void load();
        }}
        title="Edit or delete generated rooms, corridors, cities, apartments and buildings"
        data-testid="open-generated"
        className="flex items-center gap-1 px-2.5 h-7 rounded text-xs font-medium bg-[var(--bg-elevated)] hover:bg-neutral-800 text-neutral-300 hover:text-white border border-[var(--border)] transition-colors cursor-pointer"
      >
        <Layers className="w-3.5 h-3.5 text-[var(--accent)]" />
        <span className="hidden sm:inline">Generated</span>
      </button>

      {/* Portaled: the toolbar is a z-30 stacking context, which would paint the status footer over a tall dialog. */}
      {open && createPortal(
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="Generated">
          <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--bg-base)] p-4 flex flex-col gap-3 shadow-2xl">
            <div className="flex items-center justify-between">
              <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
                <Layers className="w-4 h-4 text-[var(--accent)]" /> Generated objects
                <span className="text-[10px] font-normal text-[var(--warning)] border border-[var(--warning)] rounded px-1">PROCEDURAL</span>
              </h3>
              <button type="button" onClick={() => setOpen(false)} aria-label="Close" className="cursor-pointer">
                <X className="w-4 h-4 text-neutral-400" />
              </button>
            </div>

            {loadError && (
              <div role="alert" className="text-xs text-[var(--error)] flex items-center gap-2">
                {loadError}
                <button type="button" onClick={() => void load()} className="underline cursor-pointer">Retry</button>
              </div>
            )}
            {!items && !loadError && (
              <p className="text-xs text-neutral-400 flex items-center gap-2"><Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading…</p>
            )}
            {items && items.length === 0 && (
              <p className="text-xs text-neutral-400" data-testid="generated-empty">
                Nothing generated in this world yet. Use Build to add a room, corridor, city, apartment or building.
              </p>
            )}

            {items && items.length > 0 && (
              <ul className="flex flex-col gap-1.5" data-testid="generated-list">
                {items.map((it) => {
                  const lines = dependencyLines(it, items);
                  return (
                    <li key={it.entity_id}>
                      <button
                        type="button"
                        onClick={() => setSelected(it.entity_id)}
                        aria-pressed={selected === it.entity_id}
                        className={`w-full text-left rounded border px-2.5 py-1.5 cursor-pointer ${
                          selected === it.entity_id
                            ? "border-[var(--accent)] bg-[var(--bg-elevated)]"
                            : "border-[var(--border)] hover:bg-[var(--bg-surface)]"
                        }`}
                      >
                        <div className="flex items-center gap-2 text-sm text-white">
                          <span className="text-[10px] uppercase tracking-wide text-[var(--accent)] w-[4.5rem]">{it.kind}</span>
                          <span className="font-medium">{it.name}</span>
                          <span className="text-xs text-neutral-400 ml-auto">{summarize(it)}</span>
                        </div>
                        {lines.map((l) => (
                          <div key={l} className="text-[11px] text-neutral-500 pl-[5.25rem]">{l}</div>
                        ))}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}

            {current && (
              <EditForm
                key={current.entity_id}
                worldId={worldId}
                item={current}
                onSaved={async () => {
                  onChanged();
                  await load();
                }}
                onDeleted={async () => {
                  setSelected(null);
                  onChanged();
                  await load();
                }}
              />
            )}
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

function EditForm({
  worldId,
  item,
  onSaved,
  onDeleted,
}: {
  worldId: string;
  item: GeneratedItem;
  onSaved: () => Promise<void>;
  onDeleted: () => Promise<void>;
}) {
  const kind: GeneratedKind = item.kind;
  const [room, setRoom] = useState<RoomForm>(() => roomFormFrom(item));
  const [corridor, setCorridor] = useState<CorridorForm>(() => corridorFormFrom(item));
  const [city, setCity] = useState<CityForm>(() => cityFormFrom(item));
  const [building, setBuilding] = useState<BuildingForm>(() => buildingForm(item));
  const [district, setDistrict] = useState<DistrictEditForm>(() => districtEditFrom(item.params));
  const apt0 = apartmentEditFrom(item);
  const [apt, setApt] = useState<ApartmentEditForm>(apt0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [dependents, setDependents] = useState<{ names: string[]; message: string } | null>(null);
  const [confirmCascade, setConfirmCascade] = useState(false);

  const nums =
    kind === "room"
      ? [room.width, room.depth, room.height, room.seed, room.x, room.y]
      : kind === "corridor"
        ? [corridor.length, corridor.width, corridor.height, corridor.x, corridor.y]
        : kind === "city"
          ? Object.values(city)
          : [];
  const numbersOk = nums.every(Number.isFinite);
  const openingsBad = kind === "room" && hasOpeningErrors(validateOpenings(room.openings, room));
  const problems =
    kind === "building"
      ? validateBuildingForm(building)
      : kind === "apartment"
        ? validateApartmentEdit(apt)
        : kind === "district"
          ? validateDistrictEdit(district)
          : [];
  const aptChanges = apartmentPatch(apt0, apt);
  const noChange = kind === "apartment" && Object.keys(aptChanges).length === 0;
  const canSave = numbersOk && !openingsBad && problems.length === 0 && !noChange && !busy;

  function patch(): Record<string, unknown> {
    if (kind === "building") return buildingPatch(building);
    if (kind === "apartment") return aptChanges;
    if (kind === "district") return districtPatch(district);
    if (kind === "room") {
      return {
        width: room.width, depth: room.depth, height: room.height, seed: room.seed,
        openings: room.openings, origin: [room.x, room.y, item.origin?.[2] ?? 0],
      };
    }
    if (kind === "corridor") {
      return {
        length: corridor.length, width: corridor.width, height: corridor.height, axis: corridor.axis,
        ...(corridor.attach ? { attach: corridor.attach } : { origin: [corridor.x, corridor.y, item.origin?.[2] ?? 0] }),
      };
    }
    return {
      blocks_x: city.blocks_x, blocks_y: city.blocks_y, lot_size: city.lot_size,
      street_width: city.street_width, min_height: city.min_height, max_height: city.max_height,
      seed: city.seed, origin: [city.x, city.y, item.origin?.[2] ?? 0],
    };
  }

  async function save() {
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const r = await updateGenerated(worldId, kind, item.name, patch());
      await onSaved();
      setStatus(resultLine(r));
    } catch (err) {
      setError(errText(err, `Failed to save ${item.name}.`));
    } finally {
      setBusy(false);
    }
  }

  async function remove(cascade: boolean) {
    setBusy(true);
    setError(null);
    try {
      await deleteGenerated(worldId, kind, item.name, cascade);
      await onDeleted();
    } catch (err) {
      if (isApiError(err) && err.status === 409 && !cascade) {
        const message = err.message;
        setDependents({ names: parseDependents(message), message });
        setConfirmDelete(false);
      } else {
        setError(errText(err, `Failed to delete ${item.name}.`));
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="rounded border border-[var(--border)] bg-[var(--bg-surface)] p-3 flex flex-col gap-3" data-testid="generated-edit">
      <h4 className="text-xs font-semibold text-white">
        Edit {item.kind} <span className="text-[var(--accent)]">{item.name}</span>
      </h4>

      {kind === "room" && (
        <>
          <div className="grid grid-cols-3 gap-2">
            <N label="Width (m)" value={room.width} onChange={(v) => setRoom({ ...room, width: v })} />
            <N label="Depth (m)" value={room.depth} onChange={(v) => setRoom({ ...room, depth: v })} />
            <N label="Height (m)" value={room.height} onChange={(v) => setRoom({ ...room, height: v })} />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <N label="Position x (m)" value={room.x} onChange={(v) => setRoom({ ...room, x: v })} />
            <N label="Position y (m)" value={room.y} onChange={(v) => setRoom({ ...room, y: v })} />
            <N label="Seed" value={room.seed} step={1} onChange={(v) => setRoom({ ...room, seed: v })} />
          </div>
          <OpeningsEditor openings={room.openings} room={room} onChange={(openings) => setRoom({ ...room, openings })} />
        </>
      )}

      {kind === "corridor" && (
        <>
          <div className="grid grid-cols-3 gap-2">
            <N label="Length (m)" value={corridor.length} onChange={(v) => setCorridor({ ...corridor, length: v })} />
            <N label="Width (m)" value={corridor.width} onChange={(v) => setCorridor({ ...corridor, width: v })} />
            <N label="Height (m)" value={corridor.height} onChange={(v) => setCorridor({ ...corridor, height: v })} />
          </div>
          <label className="flex flex-col gap-1 text-xs text-neutral-400">
            Runs along
            <select value={corridor.axis} onChange={(e) => setCorridor({ ...corridor, axis: e.target.value as "x" | "y" })} className={inputCls}>
              <option value="x">East-West (x)</option>
              <option value="y">North-South (y)</option>
            </select>
          </label>
          {corridor.attach ? (
            <label className="flex flex-col gap-1 text-xs text-neutral-400">
              Attached side (position follows the room)
              <select
                value={corridor.attach.side}
                onChange={(e) => setCorridor({ ...corridor, attach: { to: corridor.attach!.to, side: e.target.value } })}
                className={inputCls}
              >
                {["east", "west", "north", "south"].map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              <N label="Position x (m)" value={corridor.x} onChange={(v) => setCorridor({ ...corridor, x: v })} />
              <N label="Position y (m)" value={corridor.y} onChange={(v) => setCorridor({ ...corridor, y: v })} />
            </div>
          )}
        </>
      )}

      {kind === "city" && (
        <div className="grid grid-cols-3 gap-2">
          <N label="Blocks east-west" value={city.blocks_x} step={1} onChange={(v) => setCity({ ...city, blocks_x: v })} />
          <N label="Blocks north-south" value={city.blocks_y} step={1} onChange={(v) => setCity({ ...city, blocks_y: v })} />
          <N label="Lot size (m)" value={city.lot_size} onChange={(v) => setCity({ ...city, lot_size: v })} />
          <N label="Street width (m)" value={city.street_width} onChange={(v) => setCity({ ...city, street_width: v })} />
          <N label="Min height (m)" value={city.min_height} onChange={(v) => setCity({ ...city, min_height: v })} />
          <N label="Max height (m)" value={city.max_height} onChange={(v) => setCity({ ...city, max_height: v })} />
          <N label="Seed" value={city.seed} step={1} onChange={(v) => setCity({ ...city, seed: v })} />
          <N label="Position x (m)" value={city.x} onChange={(v) => setCity({ ...city, x: v })} />
          <N label="Position y (m)" value={city.y} onChange={(v) => setCity({ ...city, y: v })} />
        </div>
      )}

      {kind === "building" && (
        <fieldset className="flex flex-col gap-2">
          <legend className="text-xs text-neutral-400 mb-1">Storeys and storey heights (metres)</legend>
          <N
            label={`Storeys (1-${MAX_UI_STOREYS})`}
            value={building.storeys}
            step={1}
            onChange={(v) => setBuilding({ storeys: v, heights: resizeHeights(building.heights, v) })}
          />
          <div className="grid grid-cols-3 gap-2" data-testid="building-heights">
            {building.heights.map((h, i) => (
              <N
                key={i}
                label={`Storey ${i + 1} height (m, ${H_MIN}-${H_MAX})`}
                value={h}
                onChange={(v) => setBuilding({ ...building, heights: building.heights.map((x, j) => (j === i ? v : x)) })}
              />
            ))}
          </div>
        </fieldset>
      )}

      {kind === "district" && (
        <fieldset className="flex flex-col gap-2" data-testid="district-edit">
          <legend className="text-xs text-neutral-400 mb-1 flex items-center gap-2">
            Seed and detail policy
            <span className="text-[10px] font-semibold text-[var(--warning)] border border-[var(--warning)] rounded px-1">GENERATED</span>
          </legend>
          <div className="grid grid-cols-3 gap-2">
            <N label="Seed" value={district.seed} step={1} onChange={(v) => setDistrict({ ...district, seed: v })} />
            <N label="Fully detailed buildings (empty = all)" value={district.maxFull} step={1} onChange={(v) => setDistrict({ ...district, maxFull: v })} />
            <N label="Full detail within (m, empty = no limit)" value={district.fullWithin} step={1} onChange={(v) => setDistrict({ ...district, fullWithin: v })} />
          </div>
          <p className="text-[11px] text-neutral-500">
            Buildings beyond the policy become one box each (footprint only). Layout, sizes and families are fixed here: delete and rebuild to change them.
          </p>
        </fieldset>
      )}

      {kind === "apartment" && (
        <div className="grid grid-cols-2 gap-2">
          <N label="Seed" value={apt.seed} step={1} onChange={(v) => setApt({ ...apt, seed: v })} />
          <N label="Plan index (0-9)" value={apt.plan_index} step={1} onChange={(v) => setApt({ ...apt, plan_index: v })} />
          <p className="col-span-2 text-[11px] text-neutral-500">
            Changing either re-solves the layout from the stored brief. Rooms and adjacency are fixed here: delete and rebuild to change them.
          </p>
        </div>
      )}

      {problems.length > 0 && (
        <ul className="text-xs text-[var(--warning)] list-disc pl-4" data-testid="generated-problems">
          {problems.map((p) => <li key={p}>{p}</li>)}
        </ul>
      )}

      {error && <p role="alert" className="text-xs text-[var(--error)]">{error}</p>}
      {status && <p role="status" data-testid="generated-result" className="text-xs text-[var(--success)]">{status}</p>}

      {dependents && (
        <div role="alert" className="rounded border border-[var(--warning)] p-2 text-xs text-neutral-200 flex flex-col gap-2" data-testid="delete-dependents">
          <p>{dependents.message}</p>
          {dependents.names.length > 0 && (
            <ul className="list-disc pl-4">
              {dependents.names.map((n) => <li key={n}>{n}</li>)}
            </ul>
          )}
          {!confirmCascade ? (
            <button type="button" onClick={() => setConfirmCascade(true)} className="self-start h-7 px-2 rounded border border-[var(--error)] text-[var(--error)] cursor-pointer">
              Delete with dependents
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <span>Delete {item.name} and {dependents.names.length || "its"} dependent(s)? This cannot be undone here.</span>
              <button type="button" disabled={busy} onClick={() => void remove(true)} className="h-7 px-2 rounded bg-[var(--error)] text-white disabled:opacity-40 cursor-pointer">
                Confirm delete all
              </button>
              <button type="button" onClick={() => { setConfirmCascade(false); setDependents(null); }} className="h-7 px-2 rounded border border-[var(--border)] cursor-pointer">
                Cancel
              </button>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => void save()}
          disabled={!canSave}
          className="flex items-center gap-2 h-8 px-3 rounded-md text-xs font-semibold bg-[var(--primary)] text-white hover:bg-[var(--primary-hover)] disabled:opacity-40 cursor-pointer"
        >
          {busy && <Loader2 className="w-3.5 h-3.5 animate-spin" />} Save changes
        </button>
        {!confirmDelete ? (
          <button
            type="button"
            onClick={() => { setConfirmDelete(true); setDependents(null); setConfirmCascade(false); }}
            className="flex items-center gap-1 h-8 px-3 rounded-md text-xs border border-[var(--border)] text-neutral-300 hover:text-white cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" /> Delete
          </button>
        ) : (
          <span className="flex items-center gap-2 text-xs text-neutral-300">
            Delete {item.name}?
            <button type="button" disabled={busy} onClick={() => void remove(false)} className="h-7 px-2 rounded bg-[var(--error)] text-white disabled:opacity-40 cursor-pointer">
              Confirm delete
            </button>
            <button type="button" onClick={() => setConfirmDelete(false)} className="h-7 px-2 rounded border border-[var(--border)] cursor-pointer">
              Cancel
            </button>
          </span>
        )}
      </div>
    </div>
  );
}

function N({ label, value, onChange, step = 0.1 }: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-neutral-400">
      {label}
      <input
        type="number"
        step={step}
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(e.target.value === "" ? NaN : Number(e.target.value))}
        className={inputCls}
      />
    </label>
  );
}
