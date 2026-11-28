"use client";

import { Fragment } from "react";
import type { Measurement, InteriorSpace } from "@/lib/interior/model";
import { KIND_LABELS } from "@/lib/interior/model";

/** One labelled fact. `value` null means "the engine did not record this". */
function Fact({
  label,
  value,
  unit,
  provenance,
}: {
  label: string;
  value: number | null;
  unit?: string;
  provenance?: string | null;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <span className="text-[11px] text-neutral-500">{label}</span>
      <span className="text-[11px] font-mono text-neutral-200 text-right">
        {value === null ? (
          <span className="text-neutral-600">not measured</span>
        ) : (
          <>
            {Number.isInteger(value) ? value : value.toFixed(2)}
            {unit && <span className="text-neutral-500"> {unit}</span>}
            {provenance && (
              <span className="text-neutral-600 text-[10px] ml-1">{provenance}</span>
            )}
          </>
        )}
      </span>
    </div>
  );
}

function measurementValue(
  m: Record<string, Measurement>,
  key: string,
): Measurement | null {
  return m[key] ?? null;
}

function IdList({
  label,
  items,
  emptyNote,
}: {
  label: string;
  items: Array<{ id: string; name: string }>;
  emptyNote: string;
}) {
  return (
    <div className="py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
        {label} ({items.length})
      </div>
      {items.length === 0 ? (
        <div className="text-[11px] text-neutral-600">{emptyNote}</div>
      ) : (
        <ul className="space-y-0.5">
          {items.map((i) => (
            <li key={i.id} className="text-[11px] text-neutral-300 truncate">
              {i.name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
/**
 * Room / corridor inspection, showing only what the engine recorded.
 *
 * Every field maps to a real WorldIR fact: measurements come from the room
 * promotion's `custom_properties`, connectivity from actual relationship
 * edges, and confidence/provenance pass through. Anything the engine did not
 * record renders "not measured" / "no recorded ..." rather than a plausible
 * default.
 */
export default function InteriorSpaceInspector({
  space,
  onSelectEntity,
}: {
  space: InteriorSpace;
  onSelectEntity?: (entityId: string) => void;
}) {
  const m = space.measurements;
  const area = measurementValue(m, "floor_area_m2");
  const dimU = measurementValue(m, "floor_dimension_u_m");
  const dimV = measurementValue(m, "floor_dimension_v_m");
  const height = measurementValue(m, "height_m");

  const isCorridor = space.kind === "corridor";
  const length = dimU?.value ?? null;
  const width = dimV?.value ?? null;

  return (
    <div className="space-y-3" data-testid="interior-space-inspector">
      <header>
        <h3 className="text-sm font-semibold text-white">{space.name}</h3>
        <p className="text-[11px] font-mono text-neutral-500 mt-0.5">
          {KIND_LABELS[space.kind]} &middot; {space.rawType} &middot; {space.id}
        </p>
      </header>

      <div className="rounded border border-neutral-800 divide-y divide-neutral-800/60 px-2.5 py-1">
        {isCorridor ? (
          <>
            <Fact label="Length" value={length} unit="m" provenance={dimU?.provenance} />
            <Fact label="Width" value={width} unit="m" provenance={dimV?.provenance} />
            <Fact label="Height" value={height?.value ?? null} unit="m" provenance={height?.provenance} />
          </>
        ) : (
          <>
            <Fact label="Floor area" value={area?.value ?? null} unit="m2" provenance={area?.provenance} />
            <Fact label="Dimension U" value={dimU?.value ?? null} unit="m" provenance={dimU?.provenance} />
            <Fact label="Dimension V" value={dimV?.value ?? null} unit="m" provenance={dimV?.provenance} />
            <Fact label="Height" value={height?.value ?? null} unit="m" provenance={height?.provenance} />
          </>
        )}
        <Fact
          label="Confidence"
          value={space.confidence === null ? null : Number((space.confidence * 100).toFixed(0))}
          unit="%"
          provenance={space.provenance}
        />
      </div>

      {space.bounds && (
        <div className="rounded border border-neutral-800 px-2.5 py-1.5">
          <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
            Bounding geometry
          </div>
          <div className="text-[11px] font-mono text-neutral-300">
            {space.bounds.size.x.toFixed(2)} x {space.bounds.size.y.toFixed(2)} x{" "}
            {space.bounds.size.z.toFixed(2)} m
          </div>
          <div className="text-[10px] font-mono text-neutral-600 mt-0.5">
            min [{space.bounds.min.x.toFixed(2)}, {space.bounds.min.y.toFixed(2)},{" "}
            {space.bounds.min.z.toFixed(2)}] &middot; max [
            {space.bounds.max.x.toFixed(2)}, {space.bounds.max.y.toFixed(2)},{" "}
            {space.bounds.max.z.toFixed(2)}]
          </div>
        </div>
      )}

      <div className="divide-y divide-neutral-800/60">
        <IdList
          label="Connected rooms"
          items={space.connectedRooms}
          emptyNote="No recorded relationship to another room."
        />
        <IdList
          label="Corridor connections"
          items={space.corridorConnections}
          emptyNote="No recorded corridor connection."
        />
        <IdList
          label="Openings (doors / windows)"
          items={space.openings}
          emptyNote="No door or window entities were reconstructed."
        />
      </div>

      <div className="pt-2 border-t border-neutral-800">
        <div className="text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
          Contained elements ({space.containsElements.length})
        </div>
        {space.containsElements.length === 0 ? (
          <div className="text-[11px] text-neutral-600">
            This space has no recorded CONTAINS edges.
          </div>
        ) : (
          <ul className="space-y-0.5">
            {space.containsElements.map((c) => (
              <li key={c.id}>
                <button
                  type="button"
                  onClick={() => onSelectEntity?.(c.id)}
                  className="text-[11px] text-neutral-300 hover:text-[#00e5ff] transition-colors text-left"
                >
                  {KIND_LABELS[c.kind]} &middot; {c.name}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="pt-2 border-t border-neutral-800 text-[10px] text-neutral-600 leading-relaxed">
        {space.observationCount} observation(s), {space.relationshipCount}{" "}
        relationship(s) recorded on this entity.
      </div>
    </div>
  );
}