"use client";

import { useState, useEffect, useMemo } from "react";
import { fetchEntityProvenance, type EntityProvenance } from "@/lib/api/provenance";
import {
  Armchair,
  BrickWall,
  Check,
  ChevronRight,
  Download,
  DoorOpen,
  Edit3,
  Eye,
  AppWindow,
  GitBranch,
  House,
  Image as ImageIcon,
  Layers,
  Link2,
  Maximize2,
  Plus,
  Sliders,
  Square,
  Workflow,
  X,
  AlertTriangle,
  Box,
  Building,
  FileSearch,
  Shield,
  Sparkles,
} from "lucide-react";
import type { Entity, Geometry, WorldIR } from "@/types/worldir";
import {
  buildEntityEvidence, fmtConfidence, NOT_RECORDED,
} from "@/lib/viewport/inspector-evidence-model";
import { buildConfidenceFacets, readConfidence, roomMembers, TONE_TOKEN } from "@/lib/inspector/confidence-model";
import {
  allDimensionRows, anyUncalibrated, countWorld, describeWorldCounts, displayName, entityKind, fmtValue,
  hostedOpeningIds, isEvidencePlane, kindLabel, materialSwatches, openingHostId, primaryDimensionRows,
  roomFacts, scaleSentence, stateWord, type EntityKind, type Fact,
} from "@/lib/inspector/entity-facts";
import { describeScale } from "@/lib/api/scale-model";
import {
  ChipGroup, ConfidenceBars, DimensionRows, Disclosure, EntityChip, FactList, KV, MaterialCards, Muted, SectionTitle,
} from "./InspectorParts";

interface AdaptiveInspectorProps {
  world: WorldIR | null;
  worldId?: string;
  activeVersion?: string;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
  onTraceEvidence?: (evidenceId: string) => void;
  onCommitCorrection?: (
    entityId: string,
    changes: Record<string, unknown>,
    commitMessage: string
  ) => Promise<{ version_id: string; parent_version_id?: string; world_id?: string } | void>;
  onOpenDiff?: (base?: string, head?: string) => void;
  onOpenRoomConstruction?: () => void;
  onExport?: (format: "worldir" | "ply" | "cameras" | "report") => void;
  /** Set when `world` is null because the WorldIR fetch failed (e.g. a 404
   * racing a just-succeeded reconstruction), not because none was ever
   * compiled -- these must not render the same "not compiled yet" message,
   * or a real reconstruction success looks silently erased. */
  loadError?: string | null;
  onRetryLoad?: () => void;
}

type InspectorTab = "overview" | "dimensions" | "materials" | "evidence";
const TABS: ReadonlyArray<{ id: InspectorTab; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "dimensions", label: "Dimensions" },
  { id: "materials", label: "Materials" },
  { id: "evidence", label: "Evidence" },
];

const PROVENANCE_WORDS: Record<string, string> = {
  OBSERVED: "Observed directly", RECONSTRUCTED: "Reconstructed from observations", INFERRED: "Inferred", PROCEDURAL: "Generated from rules", SYNTHESIZED: "Synthesised",
};

const KIND_ICON: Record<EntityKind, typeof Box> = {
  room: House, wall: BrickWall, window: AppWindow, door: DoorOpen, floor: Layers, ceiling: Square, furniture: Armchair, other: Box,
};

export default function AdaptiveInspector({
  world,
  worldId = "current-world",
  activeVersion,
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
  onTraceEvidence,
  onCommitCorrection,
  onOpenDiff,
  onOpenRoomConstruction,
  onExport,
  loadError,
  onRetryLoad,
}: AdaptiveInspectorProps) {
  const [tab, setTab] = useState<InspectorTab>("overview");
  const [isEditing, setIsEditing] = useState(false);
  const [previewImageId, setPreviewImageId] = useState<string | null>(null);

  const entity: Entity | null =
    world && selectedEntityId && world.entities
      ? world.entities[selectedEntityId] ?? null
      : null;

  const geometry: Geometry | null =
    world && entity && entity.geometry_ids && entity.geometry_ids.length > 0 && world.geometries
      ? world.geometries[entity.geometry_ids[0]] ?? null
      : null;

  return (
    <aside
      aria-label="Properties"
      className="w-full h-full flex flex-col overflow-hidden bg-[var(--bg-surface)] text-[var(--text-primary)]"
    >
      {entity ? (
        <EntityDetails
          key={entity.id}
          world={world}
          worldId={worldId}
          activeVersion={activeVersion}
          entity={entity}
          geometry={geometry}
          tab={tab}
          setTab={setTab}
          onSelectEntity={onSelectEntity}
          onFrame={() => onFrameEntity(entity.id)}
          onFrameEntity={onFrameEntity}
          onTraceEvidence={onTraceEvidence}
          onCommitCorrection={onCommitCorrection}
          onOpenDiff={onOpenDiff}
          isEditing={isEditing}
          setIsEditing={setIsEditing}
          previewImageId={previewImageId}
          setPreviewImageId={setPreviewImageId}
        />
      ) : (
        <WorldOverview
          world={world}
          worldId={worldId}
          activeVersion={activeVersion}
          onOpenRoomConstruction={onOpenRoomConstruction}
          onExport={onExport}
          loadError={loadError}
          onRetryLoad={onRetryLoad}
        />
      )}
    </aside>
  );
}

function EntityDetails({
  world,
  worldId,
  activeVersion,
  entity,
  geometry,
  tab,
  setTab,
  onSelectEntity,
  onFrame,
  onFrameEntity,
  onTraceEvidence,
  onCommitCorrection,
  onOpenDiff,
  isEditing,
  setIsEditing,
  previewImageId,
  setPreviewImageId,
}: {
  world: WorldIR | null;
  worldId: string;
  activeVersion?: string;
  entity: Entity;
  geometry: Geometry | null;
  tab: InspectorTab;
  setTab: (t: InspectorTab) => void;
  onSelectEntity: (id: string | null) => void;
  onFrame: () => void;
  onFrameEntity?: (id: string) => void;
  onTraceEvidence?: (evidenceId: string) => void;
  onCommitCorrection?: (
    entityId: string,
    changes: Record<string, unknown>,
    commitMessage: string
  ) => Promise<{ version_id: string; parent_version_id?: string; world_id?: string } | void>;
  onOpenDiff?: (base?: string, head?: string) => void;
  isEditing: boolean;
  setIsEditing: (v: boolean) => void;
  previewImageId: string | null;
  setPreviewImageId: (id: string | null) => void;
}) {
  // Confidence is a measurement the compiler makes, or it is absent. It is
  // never defaulted: a fabricated 0.5 was rendered as "50.0%" and, worse,
  // was sent on every correction commit -- writing an invented number into
  // WorldIR and the new WorldStore version.
  const conf: number | null =
    typeof entity.confidence === "number" && Number.isFinite(entity.confidence)
      ? entity.confidence
      : null;

  const pos = entity.transform?.position;
  const observations = entity.observations || [];
  const relationships = entity.relationships || [];

  // Multi-mode Correction Form State
  const [editedType, setEditedType] = useState(entity.type);
  // A blank field means "leave the recorded confidence alone" -- it is never
  // substituted with a default, so a correction cannot invent a measurement.
  const [editedConfidence, setEditedConfidence] = useState(
    conf === null ? "" : String(conf)
  );
  const parsedConfidence =
    editedConfidence.trim() === "" ? null : Number(editedConfidence);
  const confidenceIsValid =
    parsedConfidence !== null && Number.isFinite(parsedConfidence) && parsedConfidence > 0 && parsedConfidence <= 1;
  const confidenceChange =
    conf !== null && confidenceIsValid && parsedConfidence !== conf
      ? { confidence: parsedConfidence }
      : {};
  const [editedLabels, setEditedLabels] = useState((entity.semantic_labels || []).join(", "));
  const [editedBoundaryIds, setEditedBoundaryIds] = useState<string[]>(
    Array.isArray(entity.custom_properties?.boundary_element_ids)
      ? (entity.custom_properties.boundary_element_ids as string[])
      : Array.isArray(entity.custom_properties?.wall_ids)
      ? (entity.custom_properties.wall_ids as string[])
      : []
  );
  const [editedConnectedRoomIds, setEditedConnectedRoomIds] = useState<string[]>(
    Array.isArray(entity.custom_properties?.connected_room_ids)
      ? (entity.custom_properties.connected_room_ids as string[])
      : []
  );
  const [editedConnectedLevelIds, setEditedConnectedLevelIds] = useState<string[]>(
    Array.isArray(entity.custom_properties?.connected_level_ids)
      ? (entity.custom_properties.connected_level_ids as string[])
      : Array.isArray(entity.custom_properties?.storey_ids)
      ? (entity.custom_properties.storey_ids as string[])
      : []
  );
  const [selectedWallToAdd, setSelectedWallToAdd] = useState<string>("");
  const [selectedRoomToAdd, setSelectedRoomToAdd] = useState<string>("");
  const [selectedLevelToAdd, setSelectedLevelToAdd] = useState<string>("");

  const initialCorrectionTab =
    entity.type === "room"
      ? "boundary"
      : entity.type === "corridor"
      ? "connected_rooms"
      : entity.type === "door" || entity.type === "window"
      ? "opening"
      : entity.type === "stairs" || entity.type === "stair"
      ? "stair_levels"
      : "classification";
  const [correctionTab, setCorrectionTab] = useState<string>(initialCorrectionTab);

  const [commitMessage, setCommitMessage] = useState(`Review & verify ${entity.id}`);
  // F6: a typed real dimension is VERIFIED evidence, carried across rebuilds by the server.
  const dimensionEntries = Object.entries(
    ((entity.custom_properties?.dimensions as Record<string, { value?: number; unit?: string; state?: string }>) || {})
  ).filter(([, v]) => v && typeof v.value === "number");
  const [dimensionProp, setDimensionProp] = useState<string>(dimensionEntries[0]?.[0] ?? "");
  const [dimensionValue, setDimensionValue] = useState<string>("");
  const selectedDim = dimensionEntries.find(([k]) => k === dimensionProp)?.[1];
  const dimensionIsValid = dimensionValue.trim() !== "" && Number(dimensionValue) > 0 && Number.isFinite(Number(dimensionValue));
  const [submitting, setSubmitting] = useState(false);
  const [committedSuccess, setCommittedSuccess] = useState(false);
  const [committedVersionResult, setCommittedVersionResult] = useState<{
    version_id: string;
    parent_version_id?: string;
  } | null>(null);
  const [commitError, setCommitError] = useState<string | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);

  const childrenList = useMemo<Entity[]>(() => {
    if (!world?.entities) return [];
    return Object.values(world.entities).filter(
      (e) =>
        e.parent_id === entity.id ||
        (entity.relationships || []).some(
          (r) =>
            (r.type === "contains" || (r as { kind?: string }).kind === "contains") &&
            (r.target_id === e.id || (r as { target_entity_id?: string }).target_entity_id === e.id)
        )
    );
  }, [world, entity]);

  const parentEntity = useMemo(() => {
    if (!entity.parent_id || !world?.entities) return null;
    return world.entities[entity.parent_id] || null;
  }, [entity.parent_id, world]);

  const handleTogglePreview = () => {
    if (isPreviewing) {
      window.dispatchEvent(new CustomEvent("clear-correction-preview"));
      setIsPreviewing(false);
    } else {
      window.dispatchEvent(
        new CustomEvent("preview-correction", {
          detail: {
            entityId: entity.id,
            type: editedType,
            // Preview shows the real recorded confidence; an unset field
            // previews the entity as it is, not as 50%.
            confidence: confidenceIsValid ? parsedConfidence : conf,
          },
        })
      );
      setIsPreviewing(true);
    }
  };

  // Real, entity-specific trace-to-evidence -- refetched whenever the
  // selected entity (or world) changes.
  const [provenance, setProvenance] = useState<EntityProvenance | null | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    fetchEntityProvenance(worldId, entity.id).then((p) => {
      if (!cancelled) setProvenance(p);
    });
    return () => {
      cancelled = true;
    };
  }, [worldId, entity.id]);
  const provenanceLoading = provenance === undefined;

  const handleSaveCorrection = async (overrideMode?: string) => {
    if (!onCommitCorrection) return;
    setSubmitting(true);
    setCommitError(null);
    window.dispatchEvent(new CustomEvent("clear-correction-preview"));
    setIsPreviewing(false);
    try {
      const mode = overrideMode || correctionTab;
      let payload: Record<string, unknown> = {};

      if (mode === "boundary") {
        payload = { boundary_element_ids: editedBoundaryIds };
      } else if (mode === "connected_rooms") {
        payload = {
          type: editedType,
          connected_room_ids: editedConnectedRoomIds,
          ...confidenceChange,
        };
      } else if (mode === "opening") {
        payload = {
          type: editedType,
          ...confidenceChange,
        };
      } else if (mode === "stair_levels") {
        payload = {
          connected_level_ids: editedConnectedLevelIds,
        };
      } else {
        payload = {
          type: editedType,
          ...confidenceChange,
          semantic_labels: editedLabels.split(",").map((s) => s.trim()).filter(Boolean),
        };
      }
      if (dimensionProp && dimensionIsValid) {
        payload.user_correction = {
          entity_id: entity.id,
          property: dimensionProp,
          value: Number(dimensionValue),
          note: commitMessage,
        };
      }

      const res = await onCommitCorrection(entity.id, payload, commitMessage);
      if (res && typeof res === "object" && "version_id" in res) {
        setCommittedVersionResult({
          version_id: res.version_id,
          parent_version_id: (res as { parent_version_id?: string }).parent_version_id,
        });
      }
      setCommittedSuccess(true);
      setIsEditing(false);
    } catch (e) {
      console.error("Commit failed", e);
      setCommitError(e instanceof Error ? e.message : "Correction commit failed.");
    } finally {
      setSubmitting(false);
    }
  };

  const extentDimensions = geometry?.bounds_min && geometry?.bounds_max ? {
    x: Math.abs(geometry.bounds_max.x - geometry.bounds_min.x),
    y: Math.abs(geometry.bounds_max.y - geometry.bounds_min.y),
    z: Math.abs(geometry.bounds_max.z - geometry.bounds_min.z),
    volume:
      Math.abs(geometry.bounds_max.x - geometry.bounds_min.x) *
      Math.abs(geometry.bounds_max.y - geometry.bounds_min.y) *
      Math.abs(geometry.bounds_max.z - geometry.bounds_min.z),
  } : null;

  const evidence = useMemo(() => buildEntityEvidence(entity), [entity]);

  const entities = (world?.entities ?? {}) as Record<string, Entity>;
  const kind = entityKind(entity);
  const title = displayName(entity);
  const KindIcon = KIND_ICON[kind];
  const subtype = typeof entity.custom_properties?.subtype === "string" ? (entity.custom_properties.subtype as string) : null;
  const subtitle = subtype ? `${kindLabel(entity)} · ${subtype}` : kindLabel(entity);

  const facets = useMemo(() => buildConfidenceFacets(entity, entities), [entity, world]); // eslint-disable-line react-hooks/exhaustive-deps
  const dimRows = useMemo(() => primaryDimensionRows(entity, evidence.dimensions), [entity, evidence.dimensions]);
  const allRows = useMemo(() => allDimensionRows(entity, evidence.dimensions), [entity, evidence.dimensions]);
  const swatches = useMemo(() => materialSwatches(entity), [entity]);
  const materialReading = readConfidence(entity.custom_properties?.material_confidence);
  const photos = (provenance?.evidence ?? []).filter((ev) => ev.evidence_type === "photo");
  const otherEvidence = (provenance?.evidence ?? []).filter((ev) => ev.evidence_type !== "photo");

  const stateSuffix = (s: Parameters<typeof stateWord>[0]) => (s === "MEASURED" || s === "VERIFIED" || s === null ? "" : ` (${stateWord(s).toLowerCase()})`);
  const rowFacts: Fact[] = dimRows.map((r) => ({
    label: r.assumedDefault ? `${r.label} (assumed default)` : `${r.label}${stateSuffix(r.state)}`,
    value: fmtValue(r.value, r.unit),
  }));
  const parentLevel = entity.parent_id ? entities[entity.parent_id] : undefined;
  const levelName = parentLevel && (parentLevel.type === "storey" || parentLevel.type === "level") ? displayName(parentLevel) : null;
  const overviewFacts: Fact[] = kind === "room" ? roomFacts(entity, evidence.dimensions, levelName) : rowFacts;

  const roomOf = (e: Entity): string | null => {
    const direct = e.custom_properties?.room_id;
    if (typeof direct === "string" && entities[direct]) return direct;
    const rel = (e.relationships ?? []).find((r) => entities[r.target_id]?.type === "room" && (r.kind === "part_of" || r.type === "part_of"));
    return rel ? rel.target_id : null;
  };
  const idList = (key: string): string[] => {
    const v = entity.custom_properties?.[key];
    return Array.isArray(v) ? (v as unknown[]).filter((x): x is string => typeof x === "string" && !!entities[x]) : [];
  };

  const linked: Array<{ title: string; ids: string[] }> = [];
  if (kind === "room") {
    const members = roomMembers(entity, entities as never).filter((m) => !isEvidencePlane(m as never)) as Entity[];
    const by = (k: EntityKind) => members.filter((m) => entityKind(m) === k).map((m) => m.id);
    linked.push(
      { title: "Walls", ids: by("wall") }, { title: "Doors", ids: by("door") }, { title: "Windows", ids: by("window") },
      { title: "Furniture", ids: by("furniture") }, { title: "Floor", ids: by("floor") }, { title: "Ceiling", ids: by("ceiling") },
      { title: "Connected spaces", ids: [...idList("connected_corridor_ids"), ...idList("adjacent_room_ids")] },
    );
  } else if (kind === "wall") {
    linked.push({ title: "Openings in this wall", ids: hostedOpeningIds(entity as never, entities as never) });
    const rid = roomOf(entity);
    if (rid) linked.push({ title: "Part of", ids: [rid] });
  } else if (kind === "window" || kind === "door") {
    const host = openingHostId(entity as never, entities as never);
    if (host) linked.push({ title: "In wall", ids: [host] });
    const rid = roomOf(entity);
    if (rid) linked.push({ title: "Part of", ids: [rid] });
  } else if (kind === "furniture") {
    const rests = (entity.relationships ?? []).filter((r) => (r.kind ?? r.type) === "rests_on" && entities[r.target_id]).map((r) => r.target_id);
    linked.push({ title: "Resting on", ids: rests });
    const rid = roomOf(entity);
    if (rid) linked.push({ title: "Part of", ids: [rid] });
  } else if (kind === "floor" || kind === "ceiling") {
    const rid = roomOf(entity);
    if (rid) linked.push({ title: "Part of", ids: [rid] });
  } else {
    linked.push(
      { title: "Levels", ids: idList("storey_ids") }, { title: "Rooms", ids: idList("room_ids") },
      { title: "Connected rooms", ids: idList("connected_room_ids") }, { title: "Connected levels", ids: idList("connected_level_ids") },
      { title: "Doors", ids: idList("door_ids") }, { title: "Windows", ids: idList("window_ids") },
    );
  }
  if (kind === "other") {
    if (parentEntity) linked.push({ title: "Part of", ids: [parentEntity.id] });
    linked.push({ title: "Contains", ids: childrenList.map((c) => c.id) });
  }
  const shownLinked = linked.map((l) => ({ ...l, ids: [...new Set(l.ids)].filter((id) => entities[id] && !isEvidencePlane(entities[id] as never)) })).filter((l) => l.ids.length > 0);

  const focusEntity = (id: string) => {
    onSelectEntity(id);
    onFrameEntity?.(id);
  };

  const toggleEditing = () => {
    if (isEditing && isPreviewing) {
      window.dispatchEvent(new CustomEvent("clear-correction-preview"));
      setIsPreviewing(false);
    }
    setIsEditing(!isEditing);
  };
  const correctDimension = (key: string) => {
    if (dimensionEntries.some(([k]) => k === key)) setDimensionProp(key);
    setIsEditing(true);
    requestAnimationFrame(() => document.querySelector('[data-testid="verified-dimension"]')?.scrollIntoView({ block: "nearest" }));
  };

  const iconBtn =
    "p-1.5 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer";

  return (
    <>
      <header className="shrink-0 border-b border-[var(--border)] bg-[var(--bg-surface)] px-4 pt-3">
        <div className="flex items-center gap-3">
          <span aria-hidden className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-[var(--accent-subtle)] text-[var(--accent)]">
            <KindIcon className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[15px] font-semibold leading-tight text-[var(--text-primary)]" title={title}>{title}</h2>
            <p className="truncate text-[12px] text-[var(--text-secondary)]">{subtitle}</p>
          </div>
          <button
            type="button"
            onClick={toggleEditing}
            aria-pressed={isEditing}
            title={isEditing ? "Close the correction form" : "Correct this item"}
            aria-label="Correct this item"
            className={`${iconBtn} ${isEditing ? "bg-[var(--accent-subtle)] text-[var(--accent)]" : ""}`}
          >
            <Edit3 className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => {
              setIsEditing(false);
              onSelectEntity(null);
            }}
            title="Deselect [Esc]"
            aria-label="Deselect"
            className={iconBtn}
          >
            <X className="h-4 w-4" />
          </button>
          <button type="button" onClick={onFrame} title="Show in 3D view [F]" aria-label="Show in 3D view" className={iconBtn}>
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
        <div role="tablist" aria-label="Item properties" className="mt-3 flex gap-5">
          {TABS.map((t) => (
            <button
              key={t.id}
              id={`insp-tab-${t.id}`}
              role="tab"
              type="button"
              aria-selected={tab === t.id}
              aria-controls="insp-panel"
              onClick={() => setTab(t.id)}
              className={`-mb-px border-b-2 pb-2 text-[12px] transition-colors cursor-pointer ${
                tab === t.id
                  ? "border-[var(--accent)] font-medium text-[var(--text-primary)]"
                  : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </header>

      <div id="insp-panel" role="tabpanel" aria-labelledby={`insp-tab-${tab}`} className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4 text-xs">
      {/* Commit Failure Error Banner */}
      {commitError && (
        <div className="p-3 rounded-lg border border-[var(--error)]/40 bg-[var(--error-subtle)] space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[var(--error)] font-semibold">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Correction failed</span>
            </span>
            <button
              type="button"
              onClick={() => setCommitError(null)}
              className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          <p className="text-[11px] text-[var(--text-primary)] font-mono">{commitError}</p>
        </div>
      )}

      {/* Post-Commit WorldStore Version Banner */}
      {committedVersionResult && (
        <div className="p-3 rounded-lg border border-[var(--success)]/40 bg-[var(--success-subtle)] space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[var(--success)] font-semibold">
              <Check className="w-3.5 h-3.5" />
              <span>Correction saved</span>
            </span>
            <button
              type="button"
              onClick={() => {
                setCommittedVersionResult(null);
                setCommittedSuccess(false);
              }}
              className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          <div className="text-[11px] font-mono text-[var(--text-primary)] space-y-0.5">
            <div>Version: <span className="text-[var(--accent)] font-semibold">{committedVersionResult.version_id}</span></div>
            {committedVersionResult.parent_version_id && (
              <div className="text-[10px] text-[var(--text-secondary)]">Parent: {committedVersionResult.parent_version_id}</div>
            )}
          </div>
          {onOpenDiff && (
            <button
              type="button"
              onClick={() => onOpenDiff(committedVersionResult.parent_version_id, committedVersionResult.version_id)}
              className="w-full py-1.5 rounded bg-[var(--accent)]/20 hover:bg-[var(--accent)]/30 text-[var(--accent)] border border-[var(--accent)]/40 text-xs font-mono flex items-center justify-center gap-1.5 cursor-pointer font-medium"
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>See what changed</span>
            </button>
          )}
        </div>
      )}

      {/* Review / Architectural Correction Panel */}
      {isEditing && (
        <div className="p-3 rounded-lg border border-[var(--accent)]/40 bg-[var(--bg-surface)] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[var(--accent)] flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              <span>Correct this item</span>
            </span>
            <button
              type="button"
              onClick={() => {
                if (isPreviewing) {
                  window.dispatchEvent(new CustomEvent("clear-correction-preview"));
                  setIsPreviewing(false);
                }
                setIsEditing(false);
              }}
              className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>

          {/* Submode Selection Tabs */}
          <div className="flex items-center gap-1 border-b border-[var(--border-subtle)] pb-2 overflow-x-auto text-[10px]">
            {(entity.type === "room" || entity.id.startsWith("room")) && (
              <>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("boundary")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "boundary"
                      ? "bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Room Boundaries ({editedBoundaryIds.length})
                </button>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("classification")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "classification"
                      ? "bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Classification
                </button>
              </>
            )}

            {(entity.type === "corridor" || entity.id.startsWith("corridor")) && (
              <>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("connected_rooms")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "connected_rooms"
                      ? "bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Connected Rooms ({editedConnectedRoomIds.length})
                </button>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("classification")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "classification"
                      ? "bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
                      : "text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  Classification
                </button>
              </>
            )}

            {(entity.type === "door" || entity.type === "window") && (
              <button
                type="button"
                onClick={() => setCorrectionTab("opening")}
                className="px-2 py-0.5 rounded font-mono bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
              >
                Opening Classification
              </button>
            )}

            {(entity.type === "stairs" || entity.type === "stair" || entity.id.startsWith("stair")) && (
              <button
                type="button"
                onClick={() => setCorrectionTab("stair_levels")}
                className="px-2 py-0.5 rounded font-mono bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
              >
                Connected Levels ({editedConnectedLevelIds.length})
              </button>
            )}

            {entity.type !== "room" &&
              !entity.id.startsWith("room") &&
              entity.type !== "corridor" &&
              !entity.id.startsWith("corridor") &&
              entity.type !== "door" &&
              entity.type !== "window" &&
              entity.type !== "stairs" &&
              entity.type !== "stair" &&
              !entity.id.startsWith("stair") && (
                <button
                  type="button"
                  onClick={() => setCorrectionTab("classification")}
                  className="px-2 py-0.5 rounded font-mono bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 font-semibold"
                >
                  Classification
                </button>
              )}
          </div>

          {/* 1. ROOM BOUNDARY CORRECTION */}
          {correctionTab === "boundary" && (
            <div className="space-y-2.5">
              <div>
                <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider block mb-1">
                  Enclosing Boundary Walls
                </label>
                {editedBoundaryIds.length === 0 ? (
                  <p className="text-[11px] text-[var(--text-tertiary)] italic">No boundary walls assigned yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {editedBoundaryIds.map((wId) => (
                      <span
                        key={wId}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[var(--warning)]/20 text-[var(--warning)] border border-[var(--warning)]/40 text-[10px] font-mono"
                      >
                        <span>{wId}</span>
                        <button
                          type="button"
                          onClick={() => setEditedBoundaryIds(editedBoundaryIds.filter((id) => id !== wId))}
                          className="hover:text-[var(--error)] cursor-pointer"
                        >
                          <X className="w-2.5 h-2.5" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                <div className="flex gap-1.5 mt-1">
                  <select
                    value={selectedWallToAdd}
                    onChange={(e) => setSelectedWallToAdd(e.target.value)}
                    className="flex-1 h-7 px-2 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none cursor-pointer"
                  >
                    <option value="">Select wall from WorldIR to add...</option>
                    {Object.entries(world?.entities || {})
                      .filter(([id, e]) => (e.type === "wall" || id.startsWith("wall")) && !editedBoundaryIds.includes(id))
                      .map(([id]) => (
                        <option key={id} value={id}>
                          {id}
                        </option>
                      ))}
                  </select>
                  <button
                    type="button"
                    disabled={!selectedWallToAdd}
                    onClick={() => {
                      if (selectedWallToAdd && !editedBoundaryIds.includes(selectedWallToAdd)) {
                        setEditedBoundaryIds([...editedBoundaryIds, selectedWallToAdd]);
                        setSelectedWallToAdd("");
                      }
                    }}
                    className="px-2.5 h-7 rounded bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] text-[var(--accent)] border border-[var(--border)] text-xs flex items-center gap-1 cursor-pointer disabled:opacity-40"
                  >
                    <Plus className="w-3 h-3" />
                    <span>Add</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* 2. CORRIDOR CONNECTED ROOMS CORRECTION */}
          {correctionTab === "connected_rooms" && (
            <div className="space-y-2.5">
              <div>
                <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider block mb-1">
                  Connected Rooms & Spaces
                </label>
                {editedConnectedRoomIds.length === 0 ? (
                  <p className="text-[11px] text-[var(--text-tertiary)] italic">No connected rooms linked yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {editedConnectedRoomIds.map((rId) => (
                      <span
                        key={rId}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[var(--accent)]/20 text-[var(--accent)] border border-[var(--accent)]/40 text-[10px] font-mono"
                      >
                        <span>{rId}</span>
                        <button
                          type="button"
                          onClick={() => setEditedConnectedRoomIds(editedConnectedRoomIds.filter((id) => id !== rId))}
                          className="hover:text-[var(--error)] cursor-pointer"
                        >
                          <X className="w-2.5 h-2.5" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                <div className="flex gap-1.5 mt-1">
                  <select
                    value={selectedRoomToAdd}
                    onChange={(e) => setSelectedRoomToAdd(e.target.value)}
                    className="flex-1 h-7 px-2 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none cursor-pointer"
                  >
                    <option value="">Select room to connect...</option>
                    {Object.entries(world?.entities || {})
                      .filter(([id, e]) => (e.type === "room" || id.startsWith("room")) && id !== entity.id && !editedConnectedRoomIds.includes(id))
                      .map(([id]) => (
                        <option key={id} value={id}>
                          {id}
                        </option>
                      ))}
                  </select>
                  <button
                    type="button"
                    disabled={!selectedRoomToAdd}
                    onClick={() => {
                      if (selectedRoomToAdd && !editedConnectedRoomIds.includes(selectedRoomToAdd)) {
                        setEditedConnectedRoomIds([...editedConnectedRoomIds, selectedRoomToAdd]);
                        setSelectedRoomToAdd("");
                      }
                    }}
                    className="px-2.5 h-7 rounded bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] text-[var(--accent)] border border-[var(--border)] text-xs flex items-center gap-1 cursor-pointer disabled:opacity-40"
                  >
                    <Plus className="w-3 h-3" />
                    <span>Add</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* 3. OPENING CLASSIFICATION (DOOR VS WINDOW) */}
          {correctionTab === "opening" && (
            <div className="space-y-2.5">
              <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider block">Opening Classification</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setEditedType("door")}
                  className={`py-2 rounded border text-xs font-mono font-medium flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                    editedType === "door"
                      ? "bg-[var(--success)]/20 border-[var(--success)] text-[var(--success)]"
                      : "bg-[var(--bg-surface)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <Building className="w-3.5 h-3.5" />
                  <span>Door (Passage)</span>
                </button>
                <button
                  type="button"
                  onClick={() => setEditedType("window")}
                  className={`py-2 rounded border text-xs font-mono font-medium flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                    editedType === "window"
                      ? "bg-[var(--info)]/20 border-[var(--info)] text-[var(--info)]"
                      : "bg-[var(--bg-surface)] border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  <Box className="w-3.5 h-3.5" />
                  <span>Window (Glazing)</span>
                </button>
              </div>
            </div>
          )}

          {/* 4. STAIR LEVEL CONNECTIONS */}
          {correctionTab === "stair_levels" && (
            <div className="space-y-2.5">
              <div>
                <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider block mb-1">
                  Connected Levels / Storeys
                </label>
                {editedConnectedLevelIds.length === 0 ? (
                  <p className="text-[11px] text-[var(--text-tertiary)] italic">No connected levels assigned yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {editedConnectedLevelIds.map((lvlId) => (
                      <span
                        key={lvlId}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[var(--purple)]/20 text-[var(--purple)] border border-[var(--purple)]/40 text-[10px] font-mono"
                      >
                        <span>{lvlId}</span>
                        <button
                          type="button"
                          onClick={() => setEditedConnectedLevelIds(editedConnectedLevelIds.filter((id) => id !== lvlId))}
                          className="hover:text-[var(--error)] cursor-pointer"
                        >
                          <X className="w-2.5 h-2.5" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}

                <div className="flex gap-1.5 mt-1">
                  <select
                    value={selectedLevelToAdd}
                    onChange={(e) => setSelectedLevelToAdd(e.target.value)}
                    className="flex-1 h-7 px-2 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none cursor-pointer"
                  >
                    <option value="">Select storey/level to connect...</option>
                    {Object.entries(world?.entities || {})
                      .filter(
                        ([id, e]) =>
                          (e.type === "storey" || e.type === "level" || id.startsWith("storey") || id.startsWith("level")) &&
                          !editedConnectedLevelIds.includes(id)
                      )
                      .map(([id]) => (
                        <option key={id} value={id}>
                          {id}
                        </option>
                      ))}
                  </select>
                  <button
                    type="button"
                    disabled={!selectedLevelToAdd}
                    onClick={() => {
                      if (selectedLevelToAdd && !editedConnectedLevelIds.includes(selectedLevelToAdd)) {
                        setEditedConnectedLevelIds([...editedConnectedLevelIds, selectedLevelToAdd]);
                        setSelectedLevelToAdd("");
                      }
                    }}
                    className="px-2.5 h-7 rounded bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] text-[var(--purple)] border border-[var(--border)] text-xs flex items-center gap-1 cursor-pointer disabled:opacity-40"
                  >
                    <Plus className="w-3 h-3" />
                    <span>Add</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* 5. CLASSIFICATION & LABELS (FOR GENERAL OR ROOM/CORRIDOR CLASSIFICATION TAB) */}
          {correctionTab === "classification" && (
            <div className="space-y-2.5">
              <div className="space-y-1">
                <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider">Semantic Classification</label>
                <select
                  value={editedType}
                  onChange={(e) => setEditedType(e.target.value)}
                  className="w-full h-7 px-2 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none cursor-pointer"
                >
                  <option value="room">Room (Enclosed Space)</option>
                  <option value="corridor">Corridor (Circulation Space)</option>
                  <option value="wall">Wall (Vertical Partition)</option>
                  <option value="door">Door (Passage Opening)</option>
                  <option value="window">Window (Glazing Opening)</option>
                  <option value="floor">Floor (Walking Surface)</option>
                  <option value="ceiling">Ceiling (Upper Boundary)</option>
                  <option value="stairs">Stair (Circulation Element)</option>
                  <option value="roof">Roof (Exterior Covering)</option>
                  <option value="column">Column (Vertical Structural Member)</option>
                  <option value="beam">Beam (Horizontal Structural Member)</option>
                  <option value="object">Object (Furniture / Equipment)</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider">Semantic Labels</label>
                <input
                  type="text"
                  value={editedLabels}
                  onChange={(e) => setEditedLabels(e.target.value)}
                  placeholder="e.g. office, meeting_room, primary_corridor"
                  className="w-full h-7 px-2 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:border-[var(--accent)] focus:outline-none font-mono"
                />
              </div>
            </div>
          )}

          {/* F6: correct a measured dimension with a real one (VERIFIED, survives rebuilds) */}
          {dimensionEntries.length > 0 && (
            <div className="space-y-1 rounded border border-[var(--border)] p-2" data-testid="verified-dimension">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)]">Verified dimension</span>
                <span className="text-[10px] text-[var(--text-tertiary)]">tape-measured</span>
              </div>
              <div className="flex gap-1.5">
                <select
                  value={dimensionProp}
                  onChange={(e) => setDimensionProp(e.target.value)}
                  className="h-7 min-w-0 flex-1 rounded border border-[var(--border)] bg-[var(--bg-surface)] px-1 text-[11px] text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none"
                >
                  {dimensionEntries.map(([k, v]) => (
                    <option key={k} value={k}>{k} ({v.unit ?? "?"})</option>
                  ))}
                </select>
                <input
                  type="number"
                  min="0"
                  step="0.001"
                  value={dimensionValue}
                  onChange={(e) => setDimensionValue(e.target.value)}
                  placeholder={selectedDim?.value !== undefined ? selectedDim.value.toFixed(3) : "real value"}
                  className="h-7 w-24 rounded border border-[var(--border)] bg-[var(--bg-surface)] px-1.5 text-[11px] text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:border-[var(--accent)] focus:outline-none"
                />
                <span className="flex h-7 items-center text-[10px] text-[var(--text-secondary)]">{selectedDim?.unit ?? ""}</span>
              </div>
              <p className="text-[10px] leading-snug text-[var(--text-tertiary)]">
                {dimensionValue.trim() === ""
                  ? "Leave blank to keep the measured value. A typed value is marked VERIFIED and replayed after a rebuild."
                  : "This value will be marked VERIFIED (user-entered), not measured by the engine."}
              </p>
            </div>
          )}

          {/* Common fields: Confidence & Commit Message */}
          <div className="space-y-1 pt-1 border-t border-[var(--border-subtle)]">
            <div className="flex justify-between text-[10px] text-[var(--text-secondary)] uppercase tracking-wider">
              <span>Adjusted Confidence</span>
              <span className="font-mono text-[var(--accent)]">
                {confidenceIsValid ? `${(parsedConfidence * 100).toFixed(0)}%` : "unchanged"}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={confidenceIsValid ? String(parsedConfidence) : conf ?? "0.5"}
              onChange={(e) => setEditedConfidence(e.target.value)}
              className="w-full accent-[var(--accent)] cursor-pointer"
            />
            <p className="text-[10px] text-[var(--text-tertiary)]">
              {conf === null
                ? "This entity has no recorded confidence. Leaving this at \"unchanged\" keeps it unrecorded - the correction will not invent a value."
                : "Leave unchanged to keep the recorded value."}
            </p>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-[var(--text-secondary)] uppercase tracking-wider">Reason for the change</label>
            <input
              type="text"
              value={commitMessage}
              onChange={(e) => setCommitMessage(e.target.value)}
              placeholder="e.g. Correct boundary wall assignment and connectivity"
              className="w-full h-7 px-2 rounded bg-[var(--bg-surface)] border border-[var(--border)] text-xs text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:border-[var(--accent)] focus:outline-none font-mono"
            />
          </div>

          <div className="flex items-center gap-2 pt-1">
            <button
              type="button"
              onClick={handleTogglePreview}
              className={`flex-1 py-1.5 rounded font-medium text-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer border ${
                isPreviewing
                  ? "bg-[var(--warning)]/20 text-[var(--warning)] border-[var(--warning)]/40"
                  : "bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] text-[var(--text-primary)] border-[var(--border)]"
              }`}
            >
              <Eye className="w-3.5 h-3.5" />
              <span>{isPreviewing ? "Revert 3D" : "3D Preview"}</span>
            </button>

            <button
              type="button"
              disabled={submitting}
              onClick={() => handleSaveCorrection()}
              className={`flex-1 py-1.5 rounded font-medium text-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                committedSuccess
                  ? "bg-[var(--success)] text-[var(--bg-base)]"
                  : "bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-[var(--text-primary)]"
              }`}
            >
              {submitting ? (
                <span>Saving...</span>
              ) : committedSuccess ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Saved</span>
                </>
              ) : (
                <span>Save correction</span>
              )}
            </button>
          </div>
        </div>
      )}

        {tab === "overview" && (
          <div className="space-y-5">
            <div className="flex items-start gap-3">
              {photos[0] && <PhotoThumb worldId={worldId} evidenceId={photos[0].evidence_id} name={photos[0].evidence_name} className="h-24 w-24 shrink-0" />}
              <div className="min-w-0 flex-1">
                {overviewFacts.length > 0 ? <FactList facts={overviewFacts} /> : <Muted>No measurements are recorded for this item.</Muted>}
              </div>
            </div>

            {(kind === "window" || kind === "door") && conf !== null && conf < 0.5 && (
              <p className="flex items-start gap-1.5 text-[12px] text-[var(--warning)]">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
                <span>Not enough evidence to be sure this is a {kind} (the depth fit was weak or the frame was hidden).</span>
              </p>
            )}

            {shownLinked.map((l) => (
              <ChipGroup key={l.title} title={l.title}>
                {l.ids.map((id) => (
                  <EntityChip key={id} label={displayName(entities[id])} hint={`Show ${displayName(entities[id])}`} onClick={() => focusEntity(id)} />
                ))}
              </ChipGroup>
            ))}

            <section aria-label="Confidence">
              <SectionTitle>Confidence</SectionTitle>
              <ConfidenceBars facets={facets} />
              <p className="mt-2 text-[11px] text-[var(--text-tertiary)]">Grey means no confidence was recorded. It is not a score.</p>
            </section>
          </div>
        )}

        {tab === "dimensions" && (
          <section aria-label="Dimensions">
            {dimRows.length > 0 ? (
              <>
                <DimensionRows rows={dimRows} onCorrect={onCommitCorrection ? correctDimension : undefined} />
                {dimRows.some((r) => r.unit?.startsWith("model_units")) && (
                  <p className="mt-3 text-[11px] leading-relaxed text-[var(--text-tertiary)]">
                    These sizes are in model units: the world has no metric scale yet, so they are not metres.
                  </p>
                )}
              </>
            ) : extentDimensions ? (
              <div className="space-y-2">
                <Muted>No dimensions are recorded for this item. Its 3D box measures:</Muted>
                <FactList
                  facts={[
                    { label: "Box width", value: `${extentDimensions.x.toFixed(2)}` },
                    { label: "Box depth", value: `${extentDimensions.y.toFixed(2)}` },
                    { label: "Box height", value: `${extentDimensions.z.toFixed(2)}` },
                  ]}
                />
                <Muted>Taken from the model geometry, not from a recorded measurement.</Muted>
              </div>
            ) : (
              <Muted>No dimensions are recorded for this item.</Muted>
            )}
          </section>
        )}

        {tab === "materials" && (
          <section aria-label="Materials">
            {swatches.length > 0 ? (
              <MaterialCards swatches={swatches} confidenceLabel={materialReading.label} confidenceToken={TONE_TOKEN[materialReading.tone]} />
            ) : (
              <Muted>No colour or material is recorded for this item.</Muted>
            )}
          </section>
        )}

        {tab === "evidence" && (
          <div className="space-y-5">
            <section aria-label="Supporting photographs">
              <SectionTitle>Supporting photographs</SectionTitle>
              {provenanceLoading ? (
                <Muted>Looking for supporting photographs&hellip;</Muted>
              ) : photos.length === 0 ? (
                <Muted>{provenance?.reason || "No photographs are linked to this item."}</Muted>
              ) : (
                <>
                  {provenance?.trace_level === "session" && (
                    <p className="mb-2 text-[11px] text-[var(--text-tertiary)]">
                      Linked to the capture session this came from, not to a specific frame.
                    </p>
                  )}
                  <ul className="grid grid-cols-2 gap-3">
                    {photos.map((ev) => {
                      const big = previewImageId === ev.evidence_id;
                      return (
                        <li key={ev.evidence_id} className={`space-y-1 ${big ? "col-span-2" : ""}`}>
                          <button
                            type="button"
                            onClick={() => setPreviewImageId(big ? null : ev.evidence_id)}
                            aria-label={big ? "Shrink photo" : "Enlarge photo"}
                            className="block w-full cursor-pointer"
                          >
                            <PhotoThumb worldId={worldId} evidenceId={ev.evidence_id} name={ev.evidence_name} className={`w-full ${big ? "h-44" : "h-20"}`} />
                          </button>
                          <div className="flex items-center justify-between gap-2 text-[11px]">
                            <span className="truncate font-mono text-[var(--text-secondary)]" title={ev.evidence_name}>{ev.evidence_name}</span>
                            <a
                              href={`/evidence/${ev.evidence_id}`}
                              onClick={() => onTraceEvidence?.(ev.evidence_id)}
                              className="shrink-0 text-[var(--accent)] hover:underline"
                            >
                              Open
                            </a>
                          </div>
                        </li>
                      );
                    })}
                  </ul>
                </>
              )}
              {otherEvidence.length > 0 && (
                <ul className="mt-3 space-y-1 text-[12px]">
                  {otherEvidence.map((ev) => (
                    <li key={ev.evidence_id} className="flex items-center justify-between gap-2">
                      <span className="truncate text-[var(--text-primary)]">{ev.evidence_name}</span>
                      <a href={`/evidence/${ev.evidence_id}`} onClick={() => onTraceEvidence?.(ev.evidence_id)} className="shrink-0 text-[var(--accent)] hover:underline">
                        Open {ev.evidence_type}
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </section>

      {evidence.isTarget && (
        <section className="space-y-2" aria-label={`Evidence for ${title}`} data-testid="entity-evidence">
          <SectionTitle>Evidence summary</SectionTitle>
          <FactList
            facts={[
              { label: "Confidence", value: fmtConfidence(evidence.confidence), muted: evidence.confidence === null },
              { label: "How it was derived", value: evidence.provenance ? PROVENANCE_WORDS[evidence.provenance] ?? evidence.provenance : NOT_RECORDED, muted: !evidence.provenance },
              { label: "Colour", value: evidence.colour.state === "measured" ? "Measured from photos" : "Not measured", muted: evidence.colour.state !== "measured" },
              { label: "Photos used", value: evidence.supportingViews.length ? evidence.supportingViews.join(", ") : NOT_RECORDED, muted: evidence.supportingViews.length === 0 },
            ]}
          />
          <p className="text-[11px] text-[var(--text-tertiary)]">
            {evidence.colour.basis}
            {evidence.colour.coverage !== null ? ` · ${Math.round(evidence.colour.coverage * 100)}% of the surface observed` : ""}
            {evidence.colour.hasTexture ? " · textured" : ""}
          </p>
        </section>
      )}
          </div>
        )}

        <Disclosure title="Details" testId="inspector-details">
          <div className="space-y-2">
            <KV k="Identifier" v={entity.id} wrap />
            <KV k="Type" v={entity.type} />
            <KV k="How it was derived" v={entity.provenance || "Not recorded"} />
            <KV
              k="Parent"
              v={
                entity.parent_id ? (
                  <button type="button" onClick={() => focusEntity(entity.parent_id!)} className="inline-flex items-center gap-1 text-[var(--accent)] hover:underline cursor-pointer" title={`Show parent: ${entity.parent_id}`}>
                    <Link2 className="h-3 w-3" aria-hidden />
                    {parentEntity?.name || entity.parent_id}
                  </button>
                ) : (
                  "None"
                )
              }
            />
            <KV k="Children" v={childrenList.length > 0 ? childrenList.map((c) => c.name || c.id).slice(0, 4).join(", ") + (childrenList.length > 4 ? ` +${childrenList.length - 4}` : "") : "None"} />
            <KV k="Coordinate frame" v={String(world?.coordinate_frame || world?.coordinate_system || "metric_enu")} />
            <KV k="World version" v={provenanceLoading ? "…" : provenance?.version_id || activeVersion || "Not compiled"} wrap />
            <KV k="Position (XYZ)" v={pos ? `${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}` : "Unavailable"} />
            <KV k="Bounding volume" v={extentDimensions ? `${extentDimensions.volume.toFixed(2)} m³` : "Unavailable"} />
            {entity.semantic_labels && entity.semantic_labels.length > 0 && <KV k="Labels" v={entity.semantic_labels.join(", ")} />}
            {Array.isArray(entity.custom_properties?.evidence_plane_ids) && (
              <KV k="Fitted from" v={`${(entity.custom_properties.evidence_plane_ids as unknown[]).length} surface patch(es)`} />
            )}
          </div>

          {allRows.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-tertiary)]">Every recorded dimension</h4>
              {allRows.map((r) => (
                <KV key={r.key} k={r.label} v={`${fmtValue(r.value, r.unit)} · ${r.assumedDefault ? "assumed default" : stateWord(r.state)}`} />
              ))}
            </div>
          )}

      {/* Geometry Section */}
      {(
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)] flex items-center gap-1.5">
            <Box className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span>Geometry & Bounds</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[var(--bg-elevated)] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between items-start">
              <span className="text-[var(--text-secondary)] shrink-0">
                Geometry ID{entity.geometry_ids && entity.geometry_ids.length > 1 ? `s (${entity.geometry_ids.length})` : ""}
              </span>
              <div className="flex flex-col items-end gap-0.5">
                {entity.geometry_ids && entity.geometry_ids.length > 0 ? (
                  entity.geometry_ids.map((gid) => (
                    <span key={gid} className="font-mono text-[var(--text-primary)] truncate max-w-[150px]">
                      {gid}
                    </span>
                  ))
                ) : (
                  <span className="font-mono text-[var(--text-tertiary)]">Unavailable</span>
                )}
              </div>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Dimensions (L × W × H)</span>
              <span className="font-mono text-[var(--text-primary)]">
                {extentDimensions
                  ? `${extentDimensions.x.toFixed(2)} × ${extentDimensions.y.toFixed(2)} × ${extentDimensions.z.toFixed(2)}`
                  : "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Inlier / Vertex Count</span>
              <span className="font-mono text-[var(--text-primary)]">
                {geometry?.vertex_count != null
                  ? geometry.vertex_count.toLocaleString()
                  : "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">LOD Representation</span>
              <span className="font-mono text-[var(--text-primary)]">
                {geometry?.lod_level != null ? `LOD ${geometry.lod_level}` : "Unavailable"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Spatial Topology Graph Section */}
      {(
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)] flex items-center gap-1.5">
              <Workflow className="w-3.5 h-3.5 text-[var(--accent)]" />
              <span>Spatial Topology Graph</span>
            </h4>
            <span className="text-[10px] text-[var(--text-tertiary)] font-mono">
              {relationships.length} connections
            </span>
          </div>

          <div
            className="p-3 rounded-lg border bg-[var(--bg-elevated)] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            {relationships.length === 0 ? (
              <p className="text-[var(--text-tertiary)] italic">No topological relationships recorded in WorldIR.</p>
            ) : (
              <div className="space-y-1.5">
                {relationships.map((rel: { predicate?: string; kind?: string; type?: string; target_entity_id?: string; target_id?: string }, idx) => {
                  const targetId = String(rel.target_entity_id || rel.target_id || "");
                  const targetEnt = world?.entities?.[targetId];
                  const relKind = String(rel.kind || rel.type || "rel").toLowerCase();

                  return (
                    <div
                      key={idx}
                      onClick={() => targetId && onSelectEntity(targetId)}
                      className="group p-1.5 rounded bg-[var(--bg-surface)] border border-[var(--border)] hover:border-[var(--accent)] transition-colors flex items-center justify-between font-mono cursor-pointer"
                    >
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="text-[9px] px-1 py-0.2 rounded bg-[var(--warning)]/15 text-[var(--warning)] uppercase font-semibold">
                          {relKind}
                        </span>
                        <span className="text-[var(--text-primary)] group-hover:text-[var(--text-primary)] truncate">
                          {targetEnt?.name || targetId || "unknown"}
                        </span>
                      </div>
                      <div className="flex items-center gap-1 text-[10px] text-[var(--text-tertiary)] shrink-0">
                        {targetEnt && <span className="capitalize">{targetEnt.type}</span>}
                        <Maximize2 className="w-3 h-3 text-[var(--text-tertiary)] group-hover:text-[var(--accent)]" />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Canonical Provenance Pipeline: ENTITY → EVIDENCE → SESSION → RECONSTRUCTION */}
      <div className="p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-elevated)] space-y-2.5">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-secondary)] flex items-center justify-between">
          <span>Provenance Pipeline</span>
          <span className="font-mono text-[var(--accent)] text-[9px]">REAL DATA CHAIN</span>
        </div>
        <div className="flex items-center justify-between text-[10px] font-mono">
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[var(--accent)]" />
            <span className="text-[var(--text-primary)] font-semibold">ENTITY</span>
            <span className="text-[var(--text-tertiary)] max-w-[65px] truncate">{entity.id}</span>
          </div>
          <span className="text-[var(--text-tertiary)]">→</span>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[var(--success)]" />
            <span className="text-[var(--text-primary)] font-semibold">EVIDENCE</span>
            <span className="text-[var(--text-tertiary)] max-w-[65px] truncate">
              {provenance?.evidence[0]?.evidence_name || (provenance?.trace_level === "none" ? "None" : "—")}
            </span>
          </div>
          <span className="text-[var(--text-tertiary)]">→</span>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[var(--purple)]" />
            <span className="text-[var(--text-primary)] font-semibold">SESSION</span>
            <span className="text-[var(--text-tertiary)] max-w-[65px] truncate">
              {provenance?.source_session_ids?.[0] || (provenance?.trace_level === "none" ? "None" : "—")}
            </span>
          </div>
          <span className="text-[var(--text-tertiary)]">→</span>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[var(--warning)]" />
            <span className="text-[var(--text-primary)] font-semibold">RECON</span>
            <span className="text-[var(--text-tertiary)] max-w-[65px] truncate">
              {String(world?.metadata?.reconstruction?.backend ?? "unrecorded")}
            </span>
          </div>
        </div>
      </div>

      {/* Observations: WHERE THIS CAME FROM */}
      {(
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)] flex items-center gap-1.5">
            <FileSearch className="w-3.5 h-3.5 text-[var(--warning)]" />
            <span>Observations ({observations.length})</span>
          </h4>
          {observations.length === 0 ? (
            <p className="text-[var(--text-tertiary)] italic">No explicit observations attached.</p>
          ) : (
            observations.map((obs) => (
              <div
                key={obs.id}
                className="p-3 rounded-lg border bg-[var(--bg-elevated)] space-y-2 text-[11px]"
                style={{ borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-medium text-[var(--accent)]">
                    {obs.sensor_type}
                  </span>
                  {obs.confidence != null && (
                    <span className="text-[var(--text-secondary)] font-mono-num">
                      {(obs.confidence * 100).toFixed(0)}% conf
                    </span>
                  )}
                </div>
                {obs.metadata && (
                  <div className="space-y-1 pt-1 border-t border-[var(--border-subtle)]">
                    {Object.entries(obs.metadata).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <span className="text-[var(--text-secondary)] truncate">{k}</span>
                        <span className="font-mono text-[var(--text-primary)] text-right truncate">
                          {typeof v === "number"
                            ? Number.isInteger(v)
                              ? v
                              : v.toFixed(3)
                            : String(v ?? "—")}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {/* Provenance & Lineage */}
      {(
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)] flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-[var(--success)]" />
            <span>Provenance & Lineage</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[var(--bg-elevated)] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Derivation</span>
              <span className="font-mono font-medium text-[var(--text-primary)]">
                {entity.provenance || "Not recorded"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Source Session</span>
              <span className="text-[var(--text-primary)] font-mono">
                {provenance?.source_session_ids?.join(", ") || "Unassigned"}
              </span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-[var(--text-secondary)] shrink-0">WorldStore Version</span>
              <span className="text-[var(--accent)] font-mono break-all text-right">
                {provenanceLoading ? "…" : provenance?.version_id ?? "Not compiled"}
              </span>
            </div>
            {entity.semantic_labels && entity.semantic_labels.length > 0 && (
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Semantic Labels</span>
                <span className="font-mono text-[var(--text-primary)]">
                  {entity.semantic_labels.join(", ")}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Uncertainty */}
      {(
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-secondary)] flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-[var(--warning)]" />
            <span>Uncertainty Model</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[var(--bg-elevated)] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between">
              <span className="text-[var(--text-secondary)]">Certainty Score</span>
              <span className="font-mono text-[var(--text-primary)]">
                {conf !== null ? `${conf.toFixed(3)} (${(conf * 100).toFixed(1)}%)` : "Not available"}
              </span>
            </div>
            {typeof (entity.custom_properties?.inlier_fraction ?? entity.custom_properties?.inlier_ratio) === "number" && (
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Inlier Fraction</span>
                <span className="font-mono text-[var(--text-primary)]">
                  {(Number(entity.custom_properties?.inlier_fraction ?? entity.custom_properties?.inlier_ratio) * 100).toFixed(1)}%
                </span>
              </div>
            )}
            {typeof (entity.custom_properties?.residual_m ?? entity.custom_properties?.fit_residual_m) === "number" && (
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Fit Residual</span>
                <span className="font-mono text-[var(--text-primary)]">
                  {(Number(entity.custom_properties?.residual_m ?? entity.custom_properties?.fit_residual_m) * 1000).toFixed(1)} mm
                </span>
              </div>
            )}
            {typeof entity.custom_properties?.uncertainty_sigma_m === "number" && (
              <div className="flex justify-between">
                <span className="text-[var(--text-secondary)]">Spatial StdDev (σ)</span>
                <span className="font-mono text-[var(--text-primary)]">
                  {(Number(entity.custom_properties?.uncertainty_sigma_m) * 100).toFixed(2)} cm
                </span>
              </div>
            )}
            {entity.custom_properties?.inlier_fraction === undefined &&
              entity.custom_properties?.inlier_ratio === undefined &&
              entity.custom_properties?.residual_m === undefined &&
              entity.custom_properties?.fit_residual_m === undefined &&
              entity.custom_properties?.uncertainty_sigma_m === undefined && (
                <div className="text-[10px] text-[var(--text-tertiary)] pt-1 border-t border-[var(--border-subtle)]">
                  No spatial covariance matrix or residual metrics recorded for this entity.
                </div>
            )}
          </div>
        </div>
      )}
        </Disclosure>
      </div>
    </>
  );
}

function PhotoThumb({ worldId, evidenceId, name, className }: { worldId: string; evidenceId: string; name: string; className?: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <span className={`${className ?? ""} grid place-items-center rounded border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-tertiary)]`}>
        <ImageIcon className="h-4 w-4" aria-label="Photo unavailable" />
      </span>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={`/api/worlds/${worldId}/evidence/${evidenceId}/image`}
      alt={`Photograph ${name}`}
      className={`${className ?? ""} rounded border border-[var(--border)] object-cover`}
      onError={() => setFailed(true)}
    />
  );
}

function WorldOverview({
  world,
  worldId,
  activeVersion,
  onOpenRoomConstruction,
  onExport,
  loadError,
  onRetryLoad,
}: {
  world: WorldIR | null;
  worldId: string;
  activeVersion?: string;
  onOpenRoomConstruction?: () => void;
  onExport?: (format: "worldir" | "ply" | "cameras" | "report") => void;
  loadError?: string | null;
  onRetryLoad?: () => void;
}) {
  const actionBtn =
    "w-full py-2 rounded font-medium text-xs bg-[var(--bg-elevated)] hover:bg-[var(--bg-hover)] border border-[var(--border)] text-[var(--text-primary)] transition-colors flex items-center justify-center gap-1.5 cursor-pointer";

  if (!world && loadError) {
    return (
      <div className="space-y-3 p-4 text-center text-xs text-[var(--error)]">
        <p>Couldn&apos;t load the compiled world: {loadError}</p>
        {onRetryLoad && (
          <button
            type="button"
            onClick={onRetryLoad}
            className="cursor-pointer rounded border border-[var(--error)]/40 bg-[var(--error-subtle)] px-3.5 py-1.5 text-xs text-[var(--error)] transition-colors hover:bg-[var(--error)]/20"
          >
            Retry
          </button>
        )}
      </div>
    );
  }

  if (!world) {
    return (
      <div className="space-y-3 p-4 text-center text-xs text-[var(--text-tertiary)]">
        <p>No 3D world has been built for this project yet.</p>
        {onOpenRoomConstruction && (
          <button
            type="button"
            onClick={onOpenRoomConstruction}
            className="cursor-pointer rounded bg-[var(--primary)] px-3.5 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--primary-hover)]"
          >
            Open room construction
          </button>
        )}
      </div>
    );
  }

  const meta = world.metadata || {};
  const counts = countWorld(world.entities as never);
  const scaleView = describeScale(meta.scale);
  const scaleKind = scaleView.kind === "unknown" && anyUncalibrated(world.entities as never) ? "relative" : scaleView.kind;
  const nEnt = Object.keys(world.entities || {}).length;
  const stats: Array<[string, number]> = [
    ["Rooms", counts.rooms], ["Walls", counts.walls], ["Doors", counts.doors], ["Windows", counts.windows], ["Furniture", counts.furniture],
  ];

  return (
    <>
      <header className="shrink-0 border-b border-[var(--border)] bg-[var(--bg-surface)] px-4 py-3">
        <div className="flex items-center gap-3">
          <span aria-hidden className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-[var(--accent-subtle)] text-[var(--accent)]">
            <House className="h-5 w-5" />
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-[15px] font-semibold leading-tight text-[var(--text-primary)]">{world.name || "This world"}</h2>
            <p className="truncate text-[12px] text-[var(--text-secondary)]">Nothing selected</p>
          </div>
        </div>
      </header>

      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-4 py-4 text-xs">
        <section aria-label="What this world is">
          <SectionTitle>What this is</SectionTitle>
          <p className="text-[13px] leading-relaxed text-[var(--text-primary)]">
            {counts.total > 0 ? `A reconstructed space with ${describeWorldCounts(counts)}.` : "A reconstructed space with nothing recognised in it yet."}
          </p>
          <p className="mt-1 text-[12px] text-[var(--text-tertiary)]">Select a wall, window or object in the 3D view to see its properties.</p>
          <dl className="mt-3 grid grid-cols-5 divide-x divide-[var(--border-subtle)] border-y border-[var(--border-subtle)]" data-testid="world-counts">
            {stats.map(([label, n]) => (
              <div key={label} className="px-1 py-2 text-center">
                <dd className={`font-mono-num text-[15px] ${n === 0 ? "text-[var(--text-tertiary)]" : "text-[var(--text-primary)]"}`}>{n}</dd>
                <dt className="text-[10px] text-[var(--text-tertiary)]">{label}</dt>
              </div>
            ))}
          </dl>
        </section>

        <section aria-label="Scale">
          <SectionTitle>Scale</SectionTitle>
          <p className="flex items-start gap-2 text-[12px] leading-relaxed text-[var(--text-secondary)]" data-scale-kind={scaleKind}>
            <span
              aria-hidden
              className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
              style={{ background: scaleKind === "measured" ? "var(--conf-high)" : scaleKind === "estimated" ? "var(--warning)" : "var(--conf-unknown)" }}
            />
            <span>{scaleSentence({ kind: scaleKind })}</span>
          </p>
        </section>

        <div className="space-y-2">
          {onOpenRoomConstruction && (
            <button type="button" onClick={onOpenRoomConstruction} className={actionBtn}>
              <Workflow className="h-3.5 w-3.5 text-[var(--accent)]" />
              <span>Open room construction</span>
            </button>
          )}
          {onExport && (
            <button type="button" onClick={() => onExport("worldir")} className={actionBtn}>
              <Download className="h-3.5 w-3.5 text-[var(--accent)]" />
              <span>Export world data (JSON)</span>
            </button>
          )}
        </div>

        <Disclosure title="Details" testId="inspector-details">
          <div className="space-y-2">
            <KV k="World" v={world.id || worldId} wrap />
            {activeVersion && <KV k="Version" v={activeVersion} wrap />}
            <KV k="Items compiled" v={nEnt} />
            <KV k="Reconstruction backend" v={meta.reconstruction?.backend || "Unavailable"} />
            <KV k="Registration" v={meta.reconstruction?.registration_status || "Unavailable"} />
            <KV
              k="Cameras registered"
              v={meta.reconstruction?.cameras_registered != null ? `${meta.reconstruction.cameras_registered} / ${meta.reconstruction.cameras_input ?? "—"}` : "Unavailable"}
            />
            <KV k="Scale state" v={meta.scale?.state || "Unavailable"} />
          </div>
          {meta.scale?.note && <p className="text-[11px] leading-relaxed text-[var(--text-secondary)]">{meta.scale.note}</p>}
          {meta.depth && (
            <div className="space-y-2">
              <h4 className="text-[11px] font-medium uppercase tracking-wider text-[var(--text-tertiary)]">Depth estimation</h4>
              <KV k="Model" v={meta.depth.model || "Unavailable"} />
              <KV k="Depth maps scaled" v={meta.depth.metricized != null ? `${meta.depth.metricized} / ${meta.depth.maps ?? "—"}` : "Unavailable"} />
              {meta.depth.note && <p className="text-[11px] leading-relaxed text-[var(--text-secondary)]">{meta.depth.note}</p>}
            </div>
          )}
        </Disclosure>
      </div>
    </>
  );
}
