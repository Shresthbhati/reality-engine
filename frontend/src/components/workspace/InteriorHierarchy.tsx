"use client";

import { Fragment, useMemo, type ReactNode } from "react";
import {
  Building2,
  Layers,
  DoorOpen,
  Grid3x3,
  PanelTop,
  Footprints,
  ChevronsUpDown,
  Ruler,
  Ban,
} from "lucide-react";
import {
  KIND_LABELS,
  type InteriorEntity,
  type InteriorKind,
  type InteriorModel,
  type InteriorSpace,
} from "@/lib/interior/model";

const KIND_ICONS: Record<InteriorKind, typeof Building2> = {
  building: Building2,
  level: Layers,
  room: Grid3x3,
  corridor: ChevronsUpDown,
  wall: Ruler,
  door: DoorOpen,
  window: Grid3x3,
  floor: Footprints,
  ceiling: PanelTop,
  stairs: Layers,
  other: Grid3x3,
};

/** Classes rendered as expandable groups inside a room. */
const GROUPED_KINDS: InteriorKind[] = [
  "wall",
  "door",
  "window",
  "floor",
  "ceiling",
  "stairs",
];

interface InteriorHierarchyProps {
  model: InteriorModel;
  selectedEntityId: string | null;
  onSelect: (entityId: string) => void;
}

function confidencePct(c: number | null): string {
  return c === null ? "--" : `${Math.round(c * 100)}%`;
}

function EntityRow({
  entity,
  depth,
  selected,
  onSelect,
}: {
  entity: InteriorEntity;
  depth: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const Icon = KIND_ICONS[entity.kind] ?? Grid3x3;
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={selected ? "true" : undefined}
      style={{ paddingLeft: `${8 + depth * 14}px` }}
      className={`w-full flex items-center gap-2 py-1.5 pr-2 rounded text-left transition-colors ${
        selected
          ? "bg-[#00e5ff]/15 text-[#00e5ff]"
          : "text-neutral-300 hover:bg-neutral-800"
      }`}
    >
      <Icon className="w-3.5 h-3.5 shrink-0 opacity-70" />
      <span className="text-[11px] truncate flex-1">{entity.name}</span>
      <span className="text-[10px] font-mono text-neutral-500 shrink-0">
        {confidencePct(entity.confidence)}
      </span>
    </button>
  );
}

function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <section>
      <h4 className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400 mb-1.5">
        {label}
      </h4>
      <div className="space-y-0.5">{children}</div>
    </section>
  );
}

function Empty({ text, hint }: { text: string; hint?: string }) {
  return (
    <div className="text-[11px] text-neutral-500 border border-dashed border-neutral-800 rounded px-2.5 py-2 leading-relaxed">
      <div>{text}</div>
      {hint && <div className="text-neutral-600 mt-0.5">{hint}</div>}
    </div>
  );
}
/**
 * The interior hierarchy, built entirely from the real WorldIR version.
 *
 * Classes the engine has no producer for (doors, windows, stairs, levels and
 * corridors today) are listed with an explicit reason rather than silently
 * omitted, so a reviewer can tell "not reconstructed" from "not implemented"
 * without reading the backend.
 */
export default function InteriorHierarchy({
  model,
  selectedEntityId,
  onSelect,
}: InteriorHierarchyProps) {
  const rooms = useMemo(
    () => model.spaces.filter((s) => s.kind === "room"),
    [model.spaces],
  );
  const corridors = useMemo(
    () => model.spaces.filter((s) => s.kind === "corridor"),
    [model.spaces],
  );
  const ownedIds = useMemo(
    () => new Set(model.spaces.flatMap((s) => s.contains)),
    [model.spaces],
  );
  const ungrouped = useMemo(
    () =>
      model.elements.filter(
        (e) => !GROUPED_KINDS.includes(e.kind) && !ownedIds.has(e.id),
      ),
    [model.elements, ownedIds],
  );

  const ofKind = (kind: InteriorKind) => model.all.filter((e) => e.kind === kind);

  const renderSpace = (space: InteriorSpace) => (
    <Fragment key={space.id}>
      <EntityRow
        entity={space}
        depth={0}
        selected={selectedEntityId === space.id}
        onSelect={() => onSelect(space.id)}
      />
      {GROUPED_KINDS.map((kind) => {
        const children = space.containsElements.filter((e) => e.kind === kind);
        if (children.length === 0) return null;
        return (
          <Fragment key={kind}>
            <div
              style={{ paddingLeft: "22px" }}
              className="text-[10px] uppercase tracking-wider text-neutral-500 py-1"
            >
              {KIND_LABELS[kind]}
            </div>
            {children.map((c) => (
              <EntityRow
                key={c.id}
                entity={c}
                depth={2}
                selected={selectedEntityId === c.id}
                onSelect={() => onSelect(c.id)}
              />
            ))}
          </Fragment>
        );
      })}
    </Fragment>
  );

  return (
    <div className="space-y-3" data-testid="interior-hierarchy">
      <Section label={`Rooms (${rooms.length})`}>
        {rooms.length > 0 ? (
          rooms.map(renderSpace)
        ) : (
          <Empty
            text="The engine closed no room from this evidence."
            hint="A room requires a walkable floor enclosed by contacting walls."
          />
        )}
      </Section>

      {corridors.length > 0 && (
        <Section label={`Corridors (${corridors.length})`}>
          {corridors.map(renderSpace)}
        </Section>
      )}

      {(["wall", "floor", "ceiling"] as InteriorKind[]).map((kind) => {
        const items = ofKind(kind);
        if (items.length === 0) return null;
        return (
          <Section key={kind} label={`${KIND_LABELS[kind]} (${items.length})`}>
            {items.map((e) => (
              <EntityRow
                key={e.id}
                entity={e}
                depth={0}
                selected={selectedEntityId === e.id}
                onSelect={() => onSelect(e.id)}
              />
            ))}
          </Section>
        );
      })}

      {ungrouped.length > 0 && (
        <Section label={`Other (${ungrouped.length})`}>
          {ungrouped.map((e) => (
            <EntityRow
              key={e.id}
              entity={e}
              depth={0}
              selected={selectedEntityId === e.id}
              onSelect={() => onSelect(e.id)}
            />
          ))}
        </Section>
      )}

      {model.unsupported.length > 0 && (
        <div className="mt-4 pt-3 border-t border-neutral-800">
          <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-neutral-500 mb-2">
            <Ban className="w-3 h-3" />
            Not produced by the current engine
          </div>
          <ul className="space-y-1">
            {model.unsupported.map((u) => (
              <li
                key={u.kind}
                className="text-[10px] text-neutral-500 leading-relaxed"
              >
                <span className="text-neutral-400">{u.label}</span> - {u.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}