"use client";

import { useEffect, useState } from "react";
import { Hammer, X, Loader2 } from "lucide-react";
import {
  createCity,
  createCorridor,
  createRoom,
  listRoomTemplates,
  type RoomTemplate,
} from "@/lib/api/procedural";
import { isApiError } from "@/lib/api/client";
import ApartmentBuilder from "@/components/workspace/ApartmentBuilder";
import OpeningsEditor from "@/components/workspace/OpeningsEditor";
import {
  duplicateNameMessage,
  hasOpeningErrors,
  validateOpenings,
  type OpeningDraft,
} from "@/lib/procedural/generated-model";

type Kind = "room" | "corridor" | "city" | "apartment";

export interface BuildRoomOption {
  id: string;
  name: string;
}

/** Procedural builder: Room / Corridor / City forms -> real POST to
 * /api/worlds/{id}/{rooms|corridors|cities} -> new WorldStore version.
 * Nothing exists locally until the backend has committed it. */
export default function BuildModal({
  worldId,
  rooms,
  isOpen,
  onClose,
  onCreated,
}: {
  worldId: string;
  rooms: BuildRoomOption[];
  isOpen: boolean;
  onClose: () => void;
  onCreated: () => void;
}) {
  const [kind, setKind] = useState<Kind>("room");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [name, setName] = useState("room-1");
  const [x, setX] = useState(0);
  const [y, setY] = useState(0);
  // room
  const [width, setWidth] = useState(4);
  const [depth, setDepth] = useState(3);
  const [height, setHeight] = useState(2.4);
  // corridor
  const [length, setLength] = useState(8);
  const [corridorWidth, setCorridorWidth] = useState(2);
  const [axis, setAxis] = useState<"x" | "y">("x");
  const [connects, setConnects] = useState<string[]>([]);
  // city
  const [blocksX, setBlocksX] = useState(3);
  const [blocksY, setBlocksY] = useState(3);
  const [lotSize, setLotSize] = useState(20);
  const [streetWidth, setStreetWidth] = useState(8);
  const [minHeight, setMinHeight] = useState(6);
  const [maxHeight, setMaxHeight] = useState(30);
  const [seed, setSeed] = useState(0);
  // room openings + templates (default dimensions only; they just fill the fields)
  const [openings, setOpenings] = useState<OpeningDraft[]>([]);
  const [templates, setTemplates] = useState<RoomTemplate[]>([]);
  const [templatesNote, setTemplatesNote] = useState<string | null>(null);
  // corridor attach (preferred) vs manual position (Advanced)
  const [attachTo, setAttachTo] = useState("");
  const [attachSide, setAttachSide] = useState<"east" | "west" | "north" | "south">("east");

  useEffect(() => {
    if (!isOpen) return;
    let live = true;
    listRoomTemplates()
      .then((r) => {
        if (!live) return;
        setTemplates(r.templates);
        setTemplatesNote(r.label);
      })
      .catch(() => {
        if (live) setTemplatesNote("Templates unavailable (API did not answer); enter sizes by hand.");
      });
    return () => {
      live = false;
    };
  }, [isOpen]);

  if (!isOpen) return null;

  function pick(next: Kind) {
    setKind(next);
    setName({ room: "room-1", corridor: "corridor-1", city: "city-1", apartment: "apartment-1" }[next]);
    setError(null);
  }

  function applyTemplate(tplName: string) {
    const t = templates.find((c) => c.name === tplName);
    if (!t) return;
    setWidth(t.width);
    setDepth(t.depth);
    setHeight(t.height);
  }

  const openingErrors = kind === "room" && hasOpeningErrors(validateOpenings(openings, { width, depth }));
  const attached = kind === "corridor" && attachTo !== "";

  async function submit() {
    setSubmitting(true);
    setError(null);
    const origin: [number, number, number] = [x, y, 0];
    try {
      if (kind === "room") {
        await createRoom(worldId, { name, width, depth, height, origin, openings });
      } else if (kind === "corridor") {
        await createCorridor(worldId, {
          name, length, width: corridorWidth, height, axis, connects,
          ...(attached ? { attach: { to: attachTo, side: attachSide } } : { origin }),
        });
      } else {
        await createCity(worldId, {
          name, blocks_x: blocksX, blocks_y: blocksY, lot_size: lotSize,
          street_width: streetWidth, min_height: minHeight, max_height: maxHeight, seed, origin,
        });
      }
      onCreated();
      onClose();
    } catch (err) {
      if (isApiError(err) && err.status === 409 && /already exists/i.test(err.message)) {
        setError(`${duplicateNameMessage(kind, name)} (${err.message})`);
      } else {
        setError(isApiError(err) ? err.describe() : `Failed to create ${kind}.`);
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" role="dialog" aria-modal="true" aria-label="Build">
      <div className={`w-full ${kind === "apartment" ? "max-w-2xl" : "max-w-md"} max-h-[90vh] overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--bg-base)] p-4 flex flex-col gap-3 shadow-2xl`}>
        <div className="flex items-center justify-between">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
            <Hammer className="w-4 h-4 text-[var(--accent)]" /> Build
          </h3>
          <button type="button" onClick={onClose} aria-label="Close" className="cursor-pointer">
            <X className="w-4 h-4 text-neutral-400" />
          </button>
        </div>

        <div className="flex gap-1" role="tablist">
          {(["room", "corridor", "city", "apartment"] as Kind[]).map((k) => (
            <button
              key={k}
              type="button"
              role="tab"
              aria-selected={kind === k}
              onClick={() => pick(k)}
              className={`flex-1 h-8 rounded text-xs font-medium capitalize border cursor-pointer ${
                kind === k
                  ? "bg-[var(--primary)] text-white border-[var(--primary)]"
                  : "bg-[var(--bg-elevated)] text-neutral-300 border-[var(--border)] hover:text-white"
              }`}
            >
              {k}
            </button>
          ))}
        </div>

        {kind === "apartment" ? (
          <ApartmentBuilder worldId={worldId} onCreated={onCreated} onClose={onClose} />
        ) : (
        <>
        <Field label="Name">
          <input value={name} onChange={(e) => setName(e.target.value)} className={inputCls} />
        </Field>

        {kind === "room" && (
          <>
            <Field label="Template (default sizes only - edit freely)">
              <select
                value=""
                onChange={(e) => applyTemplate(e.target.value)}
                aria-label="Room template"
                disabled={templates.length === 0}
                className={inputCls}
              >
                <option value="">{templates.length ? "Choose a template…" : "No templates available"}</option>
                {templates.map((t) => (
                  <option key={t.name} value={t.name}>
                    {t.name} ({t.width} x {t.depth} x {t.height} m)
                  </option>
                ))}
              </select>
            </Field>
            {templatesNote && <p className="text-[10px] text-neutral-500 -mt-2">{templatesNote}</p>}
            <div className="grid grid-cols-3 gap-2">
              <Num label="Width (m)" value={width} onChange={setWidth} />
              <Num label="Depth (m)" value={depth} onChange={setDepth} />
              <Num label="Height (m)" value={height} onChange={setHeight} />
            </div>
            <OpeningsEditor openings={openings} room={{ width, depth }} onChange={setOpenings} />
          </>
        )}

        {kind === "corridor" && (
          <>
            <div className="grid grid-cols-3 gap-2">
              <Num label="Length (m)" value={length} onChange={setLength} />
              <Num label="Width (m)" value={corridorWidth} onChange={setCorridorWidth} />
              <Num label="Height (m)" value={height} onChange={setHeight} />
            </div>
            <Field label="Runs along">
              <select value={axis} onChange={(e) => setAxis(e.target.value as "x" | "y")} className={inputCls}>
                <option value="x">East-West (x)</option>
                <option value="y">North-South (y)</option>
              </select>
            </Field>
            <Field label="Attach to room (position follows it when resized)">
              <div className="grid grid-cols-2 gap-2">
                <select
                  value={attachTo}
                  onChange={(e) => setAttachTo(e.target.value)}
                  aria-label="Attach to room"
                  className={inputCls}
                >
                  <option value="">Not attached (set position)</option>
                  {rooms.map((r) => (
                    <option key={r.id} value={r.id}>{r.name}</option>
                  ))}
                </select>
                <select
                  value={attachSide}
                  onChange={(e) => setAttachSide(e.target.value as typeof attachSide)}
                  aria-label="Attach side"
                  disabled={!attached}
                  className={inputCls}
                >
                  {["east", "west", "north", "south"].map((s) => (
                    <option key={s} value={s}>{s} side</option>
                  ))}
                </select>
              </div>
            </Field>
            <Field label={rooms.length ? "Connects rooms" : "Connects rooms (no rooms yet)"}>
              <div className="flex flex-wrap gap-1.5">
                {rooms.map((r) => {
                  const on = connects.includes(r.id);
                  return (
                    <button
                      key={r.id}
                      type="button"
                      aria-pressed={on}
                      onClick={() => setConnects(on ? connects.filter((c) => c !== r.id) : [...connects, r.id])}
                      className={`px-2 h-6 rounded text-[11px] border cursor-pointer ${
                        on ? "border-[var(--accent)] text-[var(--accent)]" : "border-[var(--border)] text-neutral-400"
                      }`}
                    >
                      {r.name}
                    </button>
                  );
                })}
              </div>
            </Field>
          </>
        )}

        {kind === "city" && (
          <>
            <div className="grid grid-cols-2 gap-2">
              <Num label="Blocks east-west" value={blocksX} onChange={setBlocksX} step={1} />
              <Num label="Blocks north-south" value={blocksY} onChange={setBlocksY} step={1} />
              <Num label="Lot size (m)" value={lotSize} onChange={setLotSize} />
              <Num label="Street width (m)" value={streetWidth} onChange={setStreetWidth} />
              <Num label="Min building height (m)" value={minHeight} onChange={setMinHeight} />
              <Num label="Max building height (m)" value={maxHeight} onChange={setMaxHeight} />
            </div>
            <Num label="Seed (same seed = same city)" value={seed} onChange={setSeed} step={1} />
          </>
        )}

        {kind === "corridor" ? (
          <details className="text-xs text-neutral-400" open={!attached}>
            <summary className="cursor-pointer">Advanced: manual position{attached ? " (ignored while attached)" : ""}</summary>
            <div className="grid grid-cols-2 gap-2 mt-2">
              <Num label="Position x (m)" value={x} onChange={setX} />
              <Num label="Position y (m)" value={y} onChange={setY} />
            </div>
          </details>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <Num label="Position x (m)" value={x} onChange={setX} />
            <Num label="Position y (m)" value={y} onChange={setY} />
          </div>
        )}

        {error && <p role="alert" className="text-xs text-red-400">{error}</p>}

        <button
          type="button"
          onClick={submit}
          disabled={submitting || !name.trim() || openingErrors}
          className="flex items-center justify-center gap-2 h-9 rounded-md text-sm font-semibold bg-[var(--primary)] text-white hover:bg-[var(--primary-hover)] disabled:opacity-40 cursor-pointer"
        >
          {submitting && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
          {submitting ? "Building…" : `Build ${kind}`}
        </button>
        </>
        )}
      </div>
    </div>
  );
}

const inputCls =
  "w-full bg-transparent outline-none text-sm px-2 h-8 rounded border border-[var(--border)] text-white";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-neutral-400">
      {label}
      {children}
    </label>
  );
}

function Num({ label, value, onChange, step = 0.1 }: {
  label: string; value: number; onChange: (v: number) => void; step?: number;
}) {
  return (
    <Field label={label}>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={inputCls}
      />
    </Field>
  );
}
