"use client";

import { useState, useEffect } from "react";
import { fetchEntityProvenance, type EntityProvenance } from "@/lib/api/provenance";
import {
  Box,
  Compass,
  FileSearch,
  Layers,
  Maximize2,
  Shield,
  Sparkles,
  X,
  Camera,
  Activity,
  Edit3,
  Check,
  Sliders,
  Workflow,
  Download,
  Image as ImageIcon,
  Building,
  AlertTriangle,
  Eye,
  Plus,
  Trash2,
  Link2,
} from "lucide-react";
import type {
  Entity,
  Geometry,
  WorldIR,
} from "@/types/worldir";
import { TYPE_COLORS } from "@/lib/viewport/three-scene";

interface AdaptiveInspectorProps {
  world: WorldIR | null;
  worldId?: string;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
  onTraceEvidence?: (evidenceId: string) => void;
  onCommitCorrection?: (
    entityId: string,
    changes: Record<string, unknown>,
    commitMessage: string
  ) => Promise<void>;
  onOpenRoomConstruction?: () => void;
  onExport?: (format: "worldir" | "ply" | "cameras" | "report") => void;
}

type InspectorTab = "all" | "geometry" | "evidence" | "observations" | "provenance" | "uncertainty";

export default function AdaptiveInspector({
  world,
  worldId = "current-world",
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
  onTraceEvidence,
  onCommitCorrection,
  onOpenRoomConstruction,
  onExport,
}: AdaptiveInspectorProps) {
  const [tab, setTab] = useState<InspectorTab>("all");
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
      className="w-full h-full flex flex-col overflow-hidden bg-[#0e1013] select-none"
      style={{ borderColor: "var(--border)" }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 h-12 border-b shrink-0 bg-[#12141a]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400 shrink-0">
            {entity ? "Entity Inspector" : "World Inspector"}
          </span>
          {entity && (
            <span
              className="text-[10px] font-mono px-1.5 py-0.5 rounded truncate"
              style={{
                background: "rgba(0, 229, 255, 0.12)",
                color: "#00e5ff",
                border: "1px solid rgba(0, 229, 255, 0.3)",
              }}
            >
              {entity.type}
            </span>
          )}
        </div>

        {entity && (
          <div className="flex items-center gap-1 shrink-0">
            <button
              type="button"
              onClick={() => onFrameEntity(entity.id)}
              title="Frame Entity in Viewport [F]"
              className="p-1 rounded text-neutral-400 hover:text-[#00e5ff] hover:bg-neutral-800 transition-colors cursor-pointer"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => {
                setIsEditing(false);
                onSelectEntity(null);
              }}
              title="Deselect [Esc]"
              className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Tabs */}
      {entity && (
        <div
          className="flex items-center px-3 gap-1 h-9 border-b shrink-0 overflow-x-auto text-xs bg-[#101217]"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          {(
            [
              { id: "all", label: "Overview" },
              { id: "geometry", label: "Geometry" },
              { id: "evidence", label: "Evidence" },
              { id: "observations", label: "Observations" },
              { id: "provenance", label: "Provenance" },
              { id: "uncertainty", label: "Uncertainty" },
            ] as const
          ).map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`px-2 py-1 rounded transition-colors whitespace-nowrap cursor-pointer ${
                tab === t.id
                  ? "text-[#00e5ff] font-medium bg-[rgba(0,229,255,0.12)]"
                  : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      {/* Body Content */}
      <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4 text-xs">
        {entity ? (
          <EntityDetails
            key={entity.id}
            world={world}
            worldId={worldId}
            entity={entity}
            geometry={geometry}
            tab={tab}
            onSelectEntity={onSelectEntity}
            onFrame={() => onFrameEntity(entity.id)}
            onTraceEvidence={onTraceEvidence}
            onCommitCorrection={onCommitCorrection}
            isEditing={isEditing}
            setIsEditing={setIsEditing}
            previewImageId={previewImageId}
            setPreviewImageId={setPreviewImageId}
          />
        ) : (
          <WorldOverview
            world={world}
            worldId={worldId}
            onOpenRoomConstruction={onOpenRoomConstruction}
            onExport={onExport}
          />
        )}
      </div>
    </aside>
  );
}

function EntityDetails({
  world,
  worldId,
  entity,
  geometry,
  tab,
  onSelectEntity,
  onFrame,
  onTraceEvidence,
  onCommitCorrection,
  isEditing,
  setIsEditing,
  previewImageId,
  setPreviewImageId,
}: {
  world: WorldIR | null;
  worldId: string;
  entity: Entity;
  geometry: Geometry | null;
  tab: InspectorTab;
  onSelectEntity: (id: string | null) => void;
  onFrame: () => void;
  onTraceEvidence?: (evidenceId: string) => void;
  onCommitCorrection?: (
    entityId: string,
    changes: Record<string, unknown>,
    commitMessage: string
  ) => Promise<void>;
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
  const hexColor = (TYPE_COLORS[entity.type] || TYPE_COLORS.default)
    .toString(16)
    .padStart(6, "0");

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
  const [submitting, setSubmitting] = useState(false);
  const [committedSuccess, setCommittedSuccess] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);

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

      await onCommitCorrection(entity.id, payload, commitMessage);
      setCommittedSuccess(true);
      setTimeout(() => {
        setCommittedSuccess(false);
        setIsEditing(false);
      }, 1500);
    } catch (e) {
      console.error("Commit failed", e);
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

  // Derivation status. A provenance the compiler did not stamp is reported as
  // unrecorded -- it is never guessed to be "inferred", which would claim a
  // derivation method the pipeline never ran.
  const derivationBadge = !entity.provenance
    ? { label: "Derivation Not Recorded", color: "text-neutral-400 bg-neutral-800/40 border-neutral-600/40" }
    : entity.provenance === "OBSERVED"
    ? { label: "Directly Observed", color: "text-[#35d07f] bg-[#35d07f]/10 border-[#35d07f]/30" }
    : entity.provenance === "PROCEDURAL"
    ? { label: "Procedurally Extruded", color: "text-[#b28dff] bg-[#b28dff]/10 border-[#b28dff]/30" }
    : entity.provenance === "RECONSTRUCTED"
    ? { label: "Reconstructed", color: "text-[#00e5ff] bg-[#00e5ff]/10 border-[#00e5ff]/30" }
    : { label: `Inferred (${entity.provenance})`, color: "text-[#00e5ff] bg-[#00e5ff]/10 border-[#00e5ff]/30" };

  return (
    <div className="space-y-4">
      {/* Title Card */}
      <div
        className="p-3 rounded-lg border bg-[#151821] space-y-3"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className="w-3 h-3 rounded-sm shrink-0"
              style={{ background: `#${hexColor}` }}
            />
            <span className="font-mono font-semibold text-white truncate text-sm">
              {entity.name || entity.id}
            </span>
          </div>
          <span
            className={`text-[10px] font-mono px-1.5 py-0.5 rounded border shrink-0 ${derivationBadge.color}`}
          >
            {derivationBadge.label}
          </span>
        </div>

        {/* Identity & Classification */}
        <div className="space-y-1 text-[11px] pt-1 border-t border-neutral-800">
          <div className="flex justify-between items-center">
            <span className="text-neutral-400">Entity Identity</span>
            <span className="font-mono text-neutral-200">{entity.id}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-400">Semantic Class</span>
            <span className="font-mono text-[#00e5ff] capitalize">{entity.type}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-400">Derivation Status</span>
            <span className="font-mono text-white">
              {entity.provenance || "Not recorded"}
            </span>
          </div>
        </div>

        {/* Confidence Meter */}
        <div>
          <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
            <span>Spatial Confidence</span>
            {conf !== null ? (
              <span className="font-mono-num text-white">{(conf * 100).toFixed(1)}%</span>
            ) : (
              <span className="font-mono text-neutral-500">Not recorded</span>
            )}
          </div>
          {conf !== null ? (
            <div className="w-full h-1.5 rounded-full bg-neutral-800 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${Math.max(5, conf * 100)}%`,
                  background: conf > 0.7 ? "#2ecc71" : conf > 0.4 ? "#f5a623" : "#e74c3c",
                }}
              />
            </div>
          ) : (
            <div className="w-full h-1.5 rounded-full bg-neutral-800 overflow-hidden">
              <div
                className="h-full w-full rounded-full"
                style={{
                  background:
                    "repeating-linear-gradient(90deg, #2a2a2a 0px, #2a2a2a 6px, #1a1a1a 6px, #1a1a1a 12px)",
                }}
              />
            </div>
          )}
        </div>

        {/* Position & Volume */}
        <div className="pt-2 border-t border-neutral-800 space-y-1 text-[11px]">
          <div className="flex justify-between items-center">
            <span className="text-neutral-400">Position (XYZ m)</span>
            <span className="font-mono text-white">
              {pos
                ? `${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}`
                : "Unavailable"}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-400">Bounding Volume</span>
            <span className="font-mono text-[#00e5ff]">
              {extentDimensions ? `${extentDimensions.volume.toFixed(2)} m³` : "Unavailable"}
            </span>
          </div>
        </div>

        {/* Action Button Row */}
        <div className="flex items-center gap-2 pt-2 border-t border-neutral-800">
          <button
            type="button"
            onClick={() => onFrame()}
            className="flex-1 flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-200 hover:text-white transition-colors text-[11px] font-medium cursor-pointer"
          >
            <Maximize2 className="w-3 h-3 text-[#00e5ff]" />
            <span>Frame [F]</span>
          </button>

          <button
            type="button"
            onClick={() => {
              if (isEditing && isPreviewing) {
                window.dispatchEvent(new CustomEvent("clear-correction-preview"));
                setIsPreviewing(false);
              }
              setIsEditing(!isEditing);
            }}
            className={`flex-1 flex items-center justify-center gap-1.5 px-2.5 py-1.5 rounded transition-colors text-[11px] font-medium cursor-pointer ${
              isEditing
                ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40"
                : "bg-neutral-800 hover:bg-neutral-700 text-neutral-200 hover:text-white"
            }`}
          >
            <Edit3 className="w-3 h-3 text-[#ffb84d]" />
            <span>{isEditing ? "Editing..." : "Correct"}</span>
          </button>
        </div>
      </div>

      {/* Review / Architectural Correction Panel */}
      {isEditing && (
        <div className="p-3 rounded-lg border border-[#00e5ff]/40 bg-[#121622] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#00e5ff] flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              <span>Architectural Correction (WorldStore CAS)</span>
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
              className="text-neutral-400 hover:text-white cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>

          {/* Submode Selection Tabs */}
          <div className="flex items-center gap-1 border-b border-neutral-800 pb-2 overflow-x-auto text-[10px]">
            {(entity.type === "room" || entity.id.startsWith("room")) && (
              <>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("boundary")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "boundary"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
                      : "text-neutral-400 hover:text-white"
                  }`}
                >
                  Room Boundaries ({editedBoundaryIds.length})
                </button>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("classification")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "classification"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
                      : "text-neutral-400 hover:text-white"
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
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
                      : "text-neutral-400 hover:text-white"
                  }`}
                >
                  Connected Rooms ({editedConnectedRoomIds.length})
                </button>
                <button
                  type="button"
                  onClick={() => setCorrectionTab("classification")}
                  className={`px-2 py-0.5 rounded font-mono transition-colors cursor-pointer ${
                    correctionTab === "classification"
                      ? "bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
                      : "text-neutral-400 hover:text-white"
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
                className="px-2 py-0.5 rounded font-mono bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
              >
                Opening Classification
              </button>
            )}

            {(entity.type === "stairs" || entity.type === "stair" || entity.id.startsWith("stair")) && (
              <button
                type="button"
                onClick={() => setCorrectionTab("stair_levels")}
                className="px-2 py-0.5 rounded font-mono bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
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
                  className="px-2 py-0.5 rounded font-mono bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 font-semibold"
                >
                  Classification
                </button>
              )}
          </div>

          {/* 1. ROOM BOUNDARY CORRECTION */}
          {correctionTab === "boundary" && (
            <div className="space-y-2.5">
              <div>
                <label className="text-[10px] text-neutral-400 uppercase tracking-wider block mb-1">
                  Enclosing Boundary Walls
                </label>
                {editedBoundaryIds.length === 0 ? (
                  <p className="text-[11px] text-neutral-500 italic">No boundary walls assigned yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {editedBoundaryIds.map((wId) => (
                      <span
                        key={wId}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#f59e0b]/20 text-[#f59e0b] border border-[#f59e0b]/40 text-[10px] font-mono"
                      >
                        <span>{wId}</span>
                        <button
                          type="button"
                          onClick={() => setEditedBoundaryIds(editedBoundaryIds.filter((id) => id !== wId))}
                          className="hover:text-red-400 cursor-pointer"
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
                    className="flex-1 h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white focus:border-[#00e5ff] focus:outline-none cursor-pointer"
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
                    className="px-2.5 h-7 rounded bg-neutral-800 hover:bg-neutral-700 text-[#00e5ff] border border-neutral-700 text-xs flex items-center gap-1 cursor-pointer disabled:opacity-40"
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
                <label className="text-[10px] text-neutral-400 uppercase tracking-wider block mb-1">
                  Connected Rooms & Spaces
                </label>
                {editedConnectedRoomIds.length === 0 ? (
                  <p className="text-[11px] text-neutral-500 italic">No connected rooms linked yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {editedConnectedRoomIds.map((rId) => (
                      <span
                        key={rId}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/40 text-[10px] font-mono"
                      >
                        <span>{rId}</span>
                        <button
                          type="button"
                          onClick={() => setEditedConnectedRoomIds(editedConnectedRoomIds.filter((id) => id !== rId))}
                          className="hover:text-red-400 cursor-pointer"
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
                    className="flex-1 h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white focus:border-[#00e5ff] focus:outline-none cursor-pointer"
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
                    className="px-2.5 h-7 rounded bg-neutral-800 hover:bg-neutral-700 text-[#00e5ff] border border-neutral-700 text-xs flex items-center gap-1 cursor-pointer disabled:opacity-40"
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
              <label className="text-[10px] text-neutral-400 uppercase tracking-wider block">Opening Classification</label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setEditedType("door")}
                  className={`py-2 rounded border text-xs font-mono font-medium flex items-center justify-center gap-1.5 cursor-pointer transition-colors ${
                    editedType === "door"
                      ? "bg-[#10b981]/20 border-[#10b981] text-[#10b981]"
                      : "bg-[#0e1013] border-neutral-700 text-neutral-400 hover:text-white"
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
                      ? "bg-[#06b6d4]/20 border-[#06b6d4] text-[#06b6d4]"
                      : "bg-[#0e1013] border-neutral-700 text-neutral-400 hover:text-white"
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
                <label className="text-[10px] text-neutral-400 uppercase tracking-wider block mb-1">
                  Connected Levels / Storeys
                </label>
                {editedConnectedLevelIds.length === 0 ? (
                  <p className="text-[11px] text-neutral-500 italic">No connected levels assigned yet.</p>
                ) : (
                  <div className="flex flex-wrap gap-1 mb-2">
                    {editedConnectedLevelIds.map((lvlId) => (
                      <span
                        key={lvlId}
                        className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#a855f7]/20 text-[#a855f7] border border-[#a855f7]/40 text-[10px] font-mono"
                      >
                        <span>{lvlId}</span>
                        <button
                          type="button"
                          onClick={() => setEditedConnectedLevelIds(editedConnectedLevelIds.filter((id) => id !== lvlId))}
                          className="hover:text-red-400 cursor-pointer"
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
                    className="flex-1 h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white focus:border-[#00e5ff] focus:outline-none cursor-pointer"
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
                    className="px-2.5 h-7 rounded bg-neutral-800 hover:bg-neutral-700 text-[#a855f7] border border-neutral-700 text-xs flex items-center gap-1 cursor-pointer disabled:opacity-40"
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
                <label className="text-[10px] text-neutral-400 uppercase tracking-wider">Semantic Classification</label>
                <select
                  value={editedType}
                  onChange={(e) => setEditedType(e.target.value)}
                  className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white focus:border-[#00e5ff] focus:outline-none cursor-pointer"
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
                <label className="text-[10px] text-neutral-400 uppercase tracking-wider">Semantic Labels</label>
                <input
                  type="text"
                  value={editedLabels}
                  onChange={(e) => setEditedLabels(e.target.value)}
                  placeholder="e.g. office, meeting_room, primary_corridor"
                  className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none font-mono"
                />
              </div>
            </div>
          )}

          {/* Common fields: Confidence & Commit Message */}
          <div className="space-y-1 pt-1 border-t border-neutral-800">
            <div className="flex justify-between text-[10px] text-neutral-400 uppercase tracking-wider">
              <span>Adjusted Confidence</span>
              <span className="font-mono text-[#00e5ff]">
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
              className="w-full accent-[#00e5ff] cursor-pointer"
            />
            <p className="text-[10px] text-neutral-500">
              {conf === null
                ? "This entity has no recorded confidence. Leaving this at \"unchanged\" keeps it unrecorded - the correction will not invent a value."
                : "Leave unchanged to keep the recorded value."}
            </p>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-neutral-400 uppercase tracking-wider">Commit Rationale (WorldStore Version)</label>
            <input
              type="text"
              value={commitMessage}
              onChange={(e) => setCommitMessage(e.target.value)}
              placeholder="e.g. Correct boundary wall assignment and connectivity"
              className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none font-mono"
            />
          </div>

          <div className="flex items-center gap-2 pt-1">
            <button
              type="button"
              onClick={handleTogglePreview}
              className={`flex-1 py-1.5 rounded font-medium text-xs flex items-center justify-center gap-1.5 transition-colors cursor-pointer border ${
                isPreviewing
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                  : "bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border-neutral-700"
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
                  ? "bg-[#2ecc71] text-black"
                  : "bg-[#00e5ff] hover:bg-[#33ebff] text-black"
              }`}
            >
              {submitting ? (
                <span>Persisting CAS Version...</span>
              ) : committedSuccess ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Committed to WorldStore!</span>
                </>
              ) : (
                <span>Commit Version</span>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Building Envelope & Overview */}
      {(entity.type === "building" || entity.id.startsWith("building")) && (
        <div className="p-3 rounded-lg border border-[#f59e0b]/40 bg-[#f59e0b]/10 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-[#f59e0b] uppercase tracking-wide">
            <span className="flex items-center gap-1.5">
              <Building className="w-3.5 h-3.5" />
              <span>Building Envelope</span>
            </span>
            <span className="font-mono text-[10px]">
              {String((entity.custom_properties?.n_storeys as number | undefined) ?? (entity.custom_properties?.storey_ids as string[] | undefined)?.length ?? "1")} Levels
            </span>
          </div>

          <div className="grid grid-cols-2 gap-1.5 text-[11px] text-neutral-300 font-mono">
            <div>
              <span className="text-neutral-500">Envelope: </span>
              <span>{extentDimensions ? `${extentDimensions.x.toFixed(1)}×${extentDimensions.y.toFixed(1)}×${extentDimensions.z.toFixed(1)}m` : "—"}</span>
            </div>
            <div>
              <span className="text-neutral-500">Volume: </span>
              <span>{extentDimensions ? `${extentDimensions.volume.toFixed(1)} m³` : "—"}</span>
            </div>
          </div>

          {Array.isArray(entity.custom_properties?.storey_ids) && (entity.custom_properties.storey_ids as string[]).length > 0 && (
            <div className="pt-1.5 border-t border-[#f59e0b]/20 space-y-1">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Building Storeys / Levels</span>
              <div className="flex flex-wrap gap-1">
                {(entity.custom_properties.storey_ids as string[]).map((sId) => (
                  <button
                    key={sId}
                    type="button"
                    onClick={() => onSelectEntity(sId)}
                    className="px-2 py-0.5 rounded bg-[#f59e0b]/20 text-[#f59e0b] hover:bg-[#f59e0b]/30 text-[10px] font-mono cursor-pointer border border-[#f59e0b]/30"
                  >
                    {sId}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Storey / Level Details */}
      {(entity.type === "storey" || entity.type === "level" || entity.id.startsWith("storey")) && (
        <div className="p-3 rounded-lg border border-[#8b5cf6]/40 bg-[#8b5cf6]/10 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-[#8b5cf6] uppercase tracking-wide">
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5" />
              <span>Building Storey / Level</span>
            </span>
            <span className="font-mono text-[10px]">
              Elev: {Number(entity.custom_properties?.floor_height_m ?? entity.custom_properties?.elevation_m ?? pos?.z ?? pos?.y ?? 0).toFixed(2)} m
            </span>
          </div>

          <div className="space-y-1.5 text-[11px]">
            {Array.isArray(entity.custom_properties?.room_ids) && (entity.custom_properties.room_ids as string[]).length > 0 && (
              <div>
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Rooms</span>
                <div className="flex flex-wrap gap-1 mt-0.5">
                  {(entity.custom_properties.room_ids as string[]).map((rId) => (
                    <button
                      key={rId}
                      type="button"
                      onClick={() => onSelectEntity(rId)}
                      className="px-2 py-0.5 rounded bg-[#3b82f6]/20 text-[#3b82f6] hover:bg-[#3b82f6]/30 text-[10px] font-mono cursor-pointer border border-[#3b82f6]/30"
                    >
                      {rId}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(entity.custom_properties?.corridor_ids) && (entity.custom_properties.corridor_ids as string[]).length > 0 && (
              <div>
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Circulation Corridors</span>
                <div className="flex flex-wrap gap-1 mt-0.5">
                  {(entity.custom_properties.corridor_ids as string[]).map((cId) => (
                    <button
                      key={cId}
                      type="button"
                      onClick={() => onSelectEntity(cId)}
                      className="px-2 py-0.5 rounded bg-[#06b6d4]/20 text-[#06b6d4] hover:bg-[#06b6d4]/30 text-[10px] font-mono cursor-pointer border border-[#06b6d4]/30"
                    >
                      {cId}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {Array.isArray(entity.custom_properties?.stair_ids) && (entity.custom_properties.stair_ids as string[]).length > 0 && (
              <div>
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Vertical Stairs</span>
                <div className="flex flex-wrap gap-1 mt-0.5">
                  {(entity.custom_properties.stair_ids as string[]).map((stId) => (
                    <button
                      key={stId}
                      type="button"
                      onClick={() => onSelectEntity(stId)}
                      className="px-2 py-0.5 rounded bg-[#a855f7]/20 text-[#a855f7] hover:bg-[#a855f7]/30 text-[10px] font-mono cursor-pointer border border-[#a855f7]/30"
                    >
                      {stId}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Structural Plane Details (Wall / Floor / Ceiling) */}
      {(entity.type === "wall" || entity.type === "floor" || entity.type === "ceiling") && (
        <div className="p-3 rounded-lg border border-[#64748b]/40 bg-[#64748b]/10 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-[#94a3b8] uppercase tracking-wide">
            <span className="flex items-center gap-1.5">
              <Box className="w-3.5 h-3.5" />
              <span>Structural {entity.type} Primitive</span>
            </span>
            <span className="font-mono text-[10px]">
              {geometry?.vertex_count ? `${geometry.vertex_count} inliers` : "Plane"}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-1.5 text-[11px] text-neutral-300 font-mono">
            <div>
              <span className="text-neutral-500">Span: </span>
              <span>{extentDimensions ? `${Math.max(extentDimensions.x, extentDimensions.y).toFixed(2)} m` : "—"}</span>
            </div>
            <div>
              <span className="text-neutral-500">Thickness: </span>
              <span>{typeof entity.custom_properties?.thickness_m === "number" ? `${(entity.custom_properties.thickness_m as number).toFixed(3)} m` : "Not available"}</span>
            </div>
          </div>

          {Array.isArray(entity.custom_properties?.window_ids) && (entity.custom_properties.window_ids as string[]).length > 0 && (
            <div className="pt-1.5 border-t border-[#64748b]/20 space-y-1">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">Openings in Surface</span>
              <div className="flex flex-wrap gap-1">
                {(entity.custom_properties.window_ids as string[]).map((wId) => (
                  <button
                    key={wId}
                    type="button"
                    onClick={() => onSelectEntity(wId)}
                    className="px-2 py-0.5 rounded bg-[#10b981]/20 text-[#10b981] hover:bg-[#10b981]/30 text-[10px] font-mono cursor-pointer border border-[#10b981]/30"
                  >
                    {wId}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Opening Inspection Details (for Doors and Windows) */}
      {(entity.type === "door" || entity.type === "window") && (
        <div className="p-3 rounded-lg border border-[#10b981]/40 bg-[#10b981]/10 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-[#10b981] uppercase tracking-wide">
            <span className="flex items-center gap-1.5">
              <Building className="w-3.5 h-3.5" />
              <span>Opening Specification ({entity.type})</span>
            </span>
            <span className="font-mono text-[10px]">
              {extentDimensions ? `${(extentDimensions.x * extentDimensions.y).toFixed(2)} m² area` : "Opening"}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-1.5 text-[11px] text-neutral-300 font-mono">
            <div>
              <span className="text-neutral-500">Width: </span>
              <span>{extentDimensions ? `${extentDimensions.x.toFixed(2)} m` : "—"}</span>
            </div>
            <div>
              <span className="text-neutral-500">Height: </span>
              <span>{extentDimensions ? `${extentDimensions.y.toFixed(2)} m` : "—"}</span>
            </div>
          </div>

          {conf !== null && conf < 0.5 && (
            <div className="flex items-center gap-1.5 p-1.5 rounded bg-amber-500/20 border border-amber-500/40 text-amber-300 text-[10px] font-sans">
              <AlertTriangle className="w-3 h-3 shrink-0" />
              <span>Insufficient evidence to classify opening (depth fitting unconstrained or frame occluded)</span>
            </div>
          )}
        </div>
      )}

      {/* 1. ROOM INSPECTOR: Explicit Room Architecture, Boundaries, Openings & Metrics */}
      {(entity.type === "room" || entity.id.startsWith("room")) && (() => {
        const area = (entity.custom_properties?.floor_area_m2 as number) || (extentDimensions ? extentDimensions.x * extentDimensions.z : null);
        const dims = extentDimensions ? `${extentDimensions.x.toFixed(2)} × ${extentDimensions.z.toFixed(2)} × ${extentDimensions.y.toFixed(2)} m` : "—";
        const height = (entity.custom_properties?.height_m as number) || (extentDimensions ? extentDimensions.y : null);

        const levelId = (entity.parent_id?.startsWith("storey") || entity.parent_id?.startsWith("level"))
          ? entity.parent_id
          : (entity.custom_properties?.storey_id as string) || (entity.custom_properties?.level_id as string) || (entity.custom_properties?.level !== undefined ? `level-${entity.custom_properties.level}` : null)
          || relationships.find(r => (r.kind === "part_of" || r.kind === "contains") && (r.target_id?.startsWith("storey") || r.target_id?.startsWith("level")))?.target_id || null;

        const boundaryWallIds: string[] = Array.from(new Set([
          ...(Array.isArray(entity.custom_properties?.boundary_element_ids) ? (entity.custom_properties.boundary_element_ids as string[]) : []),
          ...(Array.isArray(entity.custom_properties?.wall_ids) ? (entity.custom_properties.wall_ids as string[]) : []),
          ...Object.entries(world?.entities || {})
            .filter(([id, e]) => e.type === "wall" && (e.parent_id === entity.id || relationships.some(r => r.target_id === id)))
            .map(([id]) => id),
        ]));

        const floorId: string | null = (entity.custom_properties?.floor_id as string)
          || relationships.find(r => world?.entities[r.target_id || ""]?.type === "floor")?.target_id
          || Object.entries(world?.entities || {}).find(([id, e]) => e.type === "floor" && e.parent_id === entity.id)?.[0] || null;

        const ceilingId: string | null = (entity.custom_properties?.ceiling_id as string)
          || relationships.find(r => world?.entities[r.target_id || ""]?.type === "ceiling")?.target_id
          || Object.entries(world?.entities || {}).find(([id, e]) => e.type === "ceiling" && e.parent_id === entity.id)?.[0] || null;

        const doorIds: string[] = Array.from(new Set([
          ...(Array.isArray(entity.custom_properties?.door_ids) ? (entity.custom_properties.door_ids as string[]) : []),
          ...relationships
            .filter(r => world?.entities[r.target_id || ""]?.type === "door")
            .map(r => r.target_id as string),
        ]));

        const windowIds: string[] = Array.from(new Set([
          ...(Array.isArray(entity.custom_properties?.window_ids) ? (entity.custom_properties.window_ids as string[]) : []),
          ...relationships
            .filter(r => world?.entities[r.target_id || ""]?.type === "window")
            .map(r => r.target_id as string),
        ]));

        const connectedSpaces: string[] = Array.from(new Set([
          ...(Array.isArray(entity.custom_properties?.connected_corridor_ids) ? (entity.custom_properties.connected_corridor_ids as string[]) : []),
          ...(Array.isArray(entity.custom_properties?.adjacent_room_ids) ? (entity.custom_properties.adjacent_room_ids as string[]) : []),
          ...relationships
            .filter(r => {
              const tgt = world?.entities[r.target_id || ""];
              return tgt && (tgt.type === "room" || tgt.type === "corridor") && r.target_id !== entity.id;
            })
            .map(r => r.target_id as string),
        ]));

        const completeness = entity.custom_properties?.boundary_completeness != null
          ? Number(entity.custom_properties.boundary_completeness)
          : null;

        return (
          <div className="p-3 rounded-lg border border-[#3b82f6]/40 bg-[#3b82f6]/10 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-semibold text-[#3b82f6] uppercase tracking-wide">
              <span className="flex items-center gap-1.5">
                <Building className="w-3.5 h-3.5" />
                <span>Room Inspector</span>
              </span>
              <span className="font-mono text-[10px] text-neutral-300">
                {area ? `${area.toFixed(2)} m²` : "—"}
              </span>
            </div>

            {/* Core Metrics Grid */}
            <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-300 font-mono bg-[#0e1013]/60 p-2 rounded border border-neutral-800">
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Room Area</span>
                <span className="text-white font-semibold">{area ? `${area.toFixed(2)} m²` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Dimensions (L×W×H)</span>
                <span className="text-white font-semibold">{dims}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Clear Height</span>
                <span className="text-white font-semibold">{height ? `${Number(height).toFixed(2)} m` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Storey / Level</span>
                {levelId ? (
                  <button
                    type="button"
                    onClick={() => onSelectEntity(levelId)}
                    className="text-[#8b5cf6] hover:underline flex items-center gap-1 cursor-pointer truncate max-w-[120px]"
                  >
                    <Layers className="w-3 h-3 shrink-0" />
                    <span className="truncate">{levelId}</span>
                  </button>
                ) : (
                  <span className="text-neutral-500">Unassigned</span>
                )}
              </div>
            </div>

            {/* Boundaries & Walls */}
            <div className="space-y-1.5 pt-1 border-t border-[#3b82f6]/20">
              <div className="flex items-center justify-between text-[10px] text-neutral-400 uppercase tracking-wider">
                <span>Boundary Elements & Walls ({boundaryWallIds.length})</span>
              </div>
              {boundaryWallIds.length === 0 ? (
                <p className="text-[10px] text-neutral-500 italic">No boundary walls registered.</p>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {boundaryWallIds.map((wId) => (
                    <button
                      key={wId}
                      type="button"
                      onClick={() => onSelectEntity(wId)}
                      className="px-2 py-0.5 rounded bg-[#f59e0b]/20 text-[#f59e0b] hover:bg-[#f59e0b]/30 text-[10px] font-mono cursor-pointer border border-[#f59e0b]/30 flex items-center gap-1"
                    >
                      <Box className="w-2.5 h-2.5" />
                      <span>{wId}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Floor & Ceiling */}
            <div className="grid grid-cols-2 gap-2 pt-1 border-t border-[#3b82f6]/20 text-[10px]">
              <div>
                <span className="text-neutral-400 uppercase tracking-wider block mb-1">Floor Primitive</span>
                {floorId ? (
                  <button
                    type="button"
                    onClick={() => onSelectEntity(floorId)}
                    className="w-full px-2 py-1 rounded bg-[#3b82f6]/20 text-[#3b82f6] hover:bg-[#3b82f6]/30 font-mono cursor-pointer border border-[#3b82f6]/30 flex items-center justify-between"
                  >
                    <span className="truncate">{floorId}</span>
                    <Maximize2 className="w-2.5 h-2.5" />
                  </button>
                ) : (
                  <span className="text-neutral-500 italic">Unmeasured</span>
                )}
              </div>
              <div>
                <span className="text-neutral-400 uppercase tracking-wider block mb-1">Ceiling Primitive</span>
                {ceilingId ? (
                  <button
                    type="button"
                    onClick={() => onSelectEntity(ceilingId)}
                    className="w-full px-2 py-1 rounded bg-[#94a3b8]/20 text-[#94a3b8] hover:bg-[#94a3b8]/30 font-mono cursor-pointer border border-[#94a3b8]/30 flex items-center justify-between"
                  >
                    <span className="truncate">{ceilingId}</span>
                    <Maximize2 className="w-2.5 h-2.5" />
                  </button>
                ) : (
                  <span className="text-neutral-500 italic">
                    {entity.custom_properties?.ceiling_evidence ? "Measured" : "Procedural Extrusion"}
                  </span>
                )}
              </div>
            </div>

            {/* Doors & Windows */}
            <div className="grid grid-cols-2 gap-2 pt-1 border-t border-[#3b82f6]/20 text-[10px]">
              <div>
                <span className="text-neutral-400 uppercase tracking-wider block mb-1">Doors ({doorIds.length})</span>
                {doorIds.length === 0 ? (
                  <span className="text-neutral-500 italic">None</span>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {doorIds.map((dId) => (
                      <button
                        key={dId}
                        type="button"
                        onClick={() => onSelectEntity(dId)}
                        className="px-1.5 py-0.5 rounded bg-[#10b981]/20 text-[#10b981] hover:bg-[#10b981]/30 font-mono cursor-pointer border border-[#10b981]/30 truncate max-w-full"
                      >
                        {dId}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <div>
                <span className="text-neutral-400 uppercase tracking-wider block mb-1">Windows ({windowIds.length})</span>
                {windowIds.length === 0 ? (
                  <span className="text-neutral-500 italic">None</span>
                ) : (
                  <div className="flex flex-wrap gap-1">
                    {windowIds.map((wId) => (
                      <button
                        key={wId}
                        type="button"
                        onClick={() => onSelectEntity(wId)}
                        className="px-1.5 py-0.5 rounded bg-[#06b6d4]/20 text-[#06b6d4] hover:bg-[#06b6d4]/30 font-mono cursor-pointer border border-[#06b6d4]/30 truncate max-w-full"
                      >
                        {wId}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Connected Spaces (Corridors & Adjacent Rooms) */}
            <div className="space-y-1.5 pt-1 border-t border-[#3b82f6]/20">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">
                Connected Spaces ({connectedSpaces.length})
              </span>
              {connectedSpaces.length === 0 ? (
                <p className="text-[10px] text-neutral-500 italic">No circulation connections recorded.</p>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {connectedSpaces.map((sId) => (
                    <button
                      key={sId}
                      type="button"
                      onClick={() => onSelectEntity(sId)}
                      className="px-2 py-0.5 rounded bg-[#00e5ff]/20 text-[#00e5ff] hover:bg-[#00e5ff]/30 text-[10px] font-mono cursor-pointer border border-[#00e5ff]/30 flex items-center gap-1"
                    >
                      <Workflow className="w-2.5 h-2.5" />
                      <span>{sId}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Uncertainty Model */}
            <div className="p-2 rounded bg-[#0e1013]/60 border border-neutral-800 text-[10px] font-mono space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-neutral-500 uppercase">Boundary Completeness:</span>
                <span className="text-white">
                  {completeness != null ? `${(completeness * 100).toFixed(0)}%` : "Not available"}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-neutral-500 uppercase">Certainty Score:</span>
                <span className="text-[#00e5ff] font-semibold">
                  {conf !== null ? conf.toFixed(3) : "Not available"}
                </span>
              </div>
            </div>
          </div>
        );
      })()}

      {/* 2. CORRIDOR INSPECTOR: Circulation Route, Connected Rooms & Openings */}
      {(entity.type === "corridor" || entity.id.startsWith("corridor")) && (() => {
        const length = (entity.custom_properties?.length_m as number) || (extentDimensions ? Math.max(extentDimensions.x, extentDimensions.z) : null);
        const width = (entity.custom_properties?.width_m as number) || (extentDimensions ? Math.min(extentDimensions.x, extentDimensions.z) : null);
        const height = (entity.custom_properties?.height_m as number) || (extentDimensions ? extentDimensions.y : null);
        const area = (entity.custom_properties?.floor_area_m2 as number) || (length && width ? length * width : null);

        const levelId = (entity.parent_id?.startsWith("storey") || entity.parent_id?.startsWith("level"))
          ? entity.parent_id
          : (entity.custom_properties?.storey_id as string) || (entity.custom_properties?.level_id as string)
          || relationships.find(r => (r.kind === "part_of" || r.kind === "contains") && (r.target_id?.startsWith("storey") || r.target_id?.startsWith("level")))?.target_id || null;

        const connectedRooms: string[] = Array.from(new Set([
          ...(Array.isArray(entity.custom_properties?.connected_room_ids) ? (entity.custom_properties.connected_room_ids as string[]) : []),
          ...relationships
            .filter(r => {
              const tgt = world?.entities[r.target_id || ""];
              return tgt && (tgt.type === "room" || tgt.type === "space") && r.target_id !== entity.id;
            })
            .map(r => r.target_id as string),
        ]));

        const openings: string[] = Array.from(new Set([
          ...relationships
            .filter(r => {
              const tgt = world?.entities[r.target_id || ""];
              return tgt && (tgt.type === "door" || tgt.type === "window");
            })
            .map(r => r.target_id as string),
        ]));

        const intersections: string[] = Array.from(new Set([
          ...relationships
            .filter(r => {
              const tgt = world?.entities[r.target_id || ""];
              return tgt && (tgt.type === "stairs" || tgt.type === "stair" || (tgt.type === "corridor" && r.target_id !== entity.id));
            })
            .map(r => r.target_id as string),
        ]));

        return (
          <div className="p-3 rounded-lg border border-[#06b6d4]/40 bg-[#06b6d4]/10 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-semibold text-[#06b6d4] uppercase tracking-wide">
              <span className="flex items-center gap-1.5">
                <Workflow className="w-3.5 h-3.5" />
                <span>Corridor Inspector</span>
              </span>
              <span className="font-mono text-[10px] text-neutral-300">
                {(entity.custom_properties?.aspect_ratio as number | undefined)?.toFixed(1) ?? (length && width ? (length / width).toFixed(1) : "—")} : 1 Aspect
              </span>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-300 font-mono bg-[#0e1013]/60 p-2 rounded border border-neutral-800">
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Length</span>
                <span className="text-white font-semibold">{length ? `${length.toFixed(2)} m` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Width</span>
                <span className="text-white font-semibold">{width ? `${width.toFixed(2)} m` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Area / Height</span>
                <span className="text-white font-semibold">{area ? `${area.toFixed(1)}m²` : "—"} · {height ? `${Number(height).toFixed(2)}m` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Storey / Level</span>
                {levelId ? (
                  <button
                    type="button"
                    onClick={() => onSelectEntity(levelId)}
                    className="text-[#8b5cf6] hover:underline flex items-center gap-1 cursor-pointer truncate max-w-[120px]"
                  >
                    <Layers className="w-3 h-3 shrink-0" />
                    <span className="truncate">{levelId}</span>
                  </button>
                ) : (
                  <span className="text-neutral-500">Unassigned</span>
                )}
              </div>
            </div>

            {/* Connected Rooms */}
            <div className="space-y-1.5 pt-1 border-t border-[#06b6d4]/20">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">
                Connected Rooms ({connectedRooms.length})
              </span>
              {connectedRooms.length === 0 ? (
                <p className="text-[10px] text-neutral-500 italic">No connected rooms recorded.</p>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {connectedRooms.map((rId) => (
                    <button
                      key={rId}
                      type="button"
                      onClick={() => onSelectEntity(rId)}
                      className="px-2 py-0.5 rounded bg-[#06b6d4]/20 text-[#06b6d4] hover:bg-[#06b6d4]/30 text-[10px] font-mono cursor-pointer border border-[#06b6d4]/30 flex items-center gap-1"
                    >
                      <Building className="w-2.5 h-2.5" />
                      <span>{rId}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Openings & Doorways */}
            <div className="space-y-1.5 pt-1 border-t border-[#06b6d4]/20">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">
                Doorways & Openings ({openings.length})
              </span>
              {openings.length === 0 ? (
                <p className="text-[10px] text-neutral-500 italic">No direct openings attached.</p>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {openings.map((opId) => (
                    <button
                      key={opId}
                      type="button"
                      onClick={() => onSelectEntity(opId)}
                      className="px-2 py-0.5 rounded bg-[#10b981]/20 text-[#10b981] hover:bg-[#10b981]/30 text-[10px] font-mono cursor-pointer border border-[#10b981]/30 flex items-center gap-1"
                    >
                      <span>{opId}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Intersections (Stairs / Cross-Corridors) */}
            <div className="space-y-1.5 pt-1 border-t border-[#06b6d4]/20">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">
                Intersections & Vertical Landings ({intersections.length})
              </span>
              {intersections.length === 0 ? (
                <p className="text-[10px] text-neutral-500 italic">No cross intersections or stairs.</p>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {intersections.map((intId) => (
                    <button
                      key={intId}
                      type="button"
                      onClick={() => onSelectEntity(intId)}
                      className="px-2 py-0.5 rounded bg-[#a855f7]/20 text-[#a855f7] hover:bg-[#a855f7]/30 text-[10px] font-mono cursor-pointer border border-[#a855f7]/30 flex items-center gap-1"
                    >
                      <Layers className="w-2.5 h-2.5" />
                      <span>{intId}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {/* 3. STAIR INSPECTOR: Vertical Circulation & Level Connectors */}
      {(entity.type === "stairs" || entity.type === "stair" || entity.id.startsWith("stair")) && (() => {
        const rise = (entity.custom_properties?.total_rise_m as number) || (extentDimensions ? extentDimensions.y : null);
        const run = (entity.custom_properties?.total_run_m as number) || (extentDimensions ? Math.max(extentDimensions.x, extentDimensions.z) : null);
        const stepCount = entity.custom_properties?.step_count as number | undefined;

        const connectedLevels: string[] = Array.from(new Set([
          ...(Array.isArray(entity.custom_properties?.connected_level_ids) ? (entity.custom_properties.connected_level_ids as string[]) : []),
          ...(Array.isArray(entity.custom_properties?.storey_ids) ? (entity.custom_properties.storey_ids as string[]) : []),
          ...relationships
            .filter(r => {
              const tgt = world?.entities[r.target_id || ""];
              return tgt && (tgt.type === "storey" || tgt.type === "level" || r.target_id?.startsWith("storey") || r.target_id?.startsWith("level"));
            })
            .map(r => r.target_id as string),
        ]));

        const connectedCorridors: string[] = Array.from(new Set([
          ...relationships
            .filter(r => {
              const tgt = world?.entities[r.target_id || ""];
              return tgt && (tgt.type === "corridor" || r.target_id?.startsWith("corridor"));
            })
            .map(r => r.target_id as string),
        ]));

        return (
          <div className="p-3 rounded-lg border border-[#a855f7]/40 bg-[#a855f7]/10 space-y-3">
            <div className="flex items-center justify-between text-[11px] font-semibold text-[#a855f7] uppercase tracking-wide">
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5" />
                <span>Stair Inspector</span>
              </span>
              <span className="font-mono text-[10px] text-neutral-300">
                {stepCount != null ? `${stepCount} Steps` : "Staircase Flight"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-[11px] text-neutral-300 font-mono bg-[#0e1013]/60 p-2 rounded border border-neutral-800">
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Total Rise</span>
                <span className="text-white font-semibold">{rise ? `${rise.toFixed(2)} m` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-500 block text-[9px] uppercase tracking-wider">Total Run</span>
                <span className="text-white font-semibold">{run ? `${run.toFixed(2)} m` : "—"}</span>
              </div>
            </div>

            {/* Connected Levels */}
            <div className="space-y-1.5 pt-1 border-t border-[#a855f7]/20">
              <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">
                Connected Levels ({connectedLevels.length})
              </span>
              {connectedLevels.length === 0 ? (
                <p className="text-[10px] text-neutral-500 italic">No connected levels assigned.</p>
              ) : (
                <div className="flex flex-wrap gap-1">
                  {connectedLevels.map((lvlId) => (
                    <button
                      key={lvlId}
                      type="button"
                      onClick={() => onSelectEntity(lvlId)}
                      className="px-2 py-0.5 rounded bg-[#a855f7]/20 text-[#a855f7] hover:bg-[#a855f7]/30 text-[10px] font-mono cursor-pointer border border-[#a855f7]/30 flex items-center gap-1"
                    >
                      <Layers className="w-2.5 h-2.5" />
                      <span>{lvlId}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Connected Landings / Corridors */}
            {connectedCorridors.length > 0 && (
              <div className="space-y-1.5 pt-1 border-t border-[#a855f7]/20">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider block">
                  Circulation Landings ({connectedCorridors.length})
                </span>
                <div className="flex flex-wrap gap-1">
                  {connectedCorridors.map((cId) => (
                    <button
                      key={cId}
                      type="button"
                      onClick={() => onSelectEntity(cId)}
                      className="px-2 py-0.5 rounded bg-[#06b6d4]/20 text-[#06b6d4] hover:bg-[#06b6d4]/30 text-[10px] font-mono cursor-pointer border border-[#06b6d4]/30 flex items-center gap-1"
                    >
                      <Workflow className="w-2.5 h-2.5" />
                      <span>{cId}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* Geometry Section */}
      {(tab === "all" || tab === "geometry") && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <Box className="w-3.5 h-3.5 text-[#00e5ff]" />
            <span>Geometry & Bounds</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between">
              <span className="text-neutral-400">Geometry ID</span>
              <span className="font-mono text-white truncate max-w-[150px]">
                {geometry?.id || "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Dimensions (L × W × H)</span>
              <span className="font-mono text-white">
                {extentDimensions
                  ? `${extentDimensions.x.toFixed(2)} × ${extentDimensions.y.toFixed(2)} × ${extentDimensions.z.toFixed(2)} m`
                  : "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Inlier / Vertex Count</span>
              <span className="font-mono text-white">
                {geometry?.vertex_count != null
                  ? geometry.vertex_count.toLocaleString()
                  : "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">LOD Representation</span>
              <span className="font-mono text-white">
                {geometry?.lod_level != null ? `LOD ${geometry.lod_level}` : "Unavailable"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Spatial Topology Graph Section */}
      {(tab === "all" || tab === "geometry") && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
              <Workflow className="w-3.5 h-3.5 text-[#00e5ff]" />
              <span>Spatial Topology Graph</span>
            </h4>
            <span className="text-[10px] text-neutral-500 font-mono">
              {relationships.length} connections
            </span>
          </div>

          <div
            className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            {relationships.length === 0 ? (
              <p className="text-neutral-500 italic">No topological relationships recorded in WorldIR.</p>
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
                      className="group p-1.5 rounded bg-[#101217] border border-[#1f222b] hover:border-[#00e5ff] transition-colors flex items-center justify-between font-mono cursor-pointer"
                    >
                      <div className="flex items-center gap-1.5 min-w-0">
                        <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/15 text-amber-400 uppercase font-semibold">
                          {relKind}
                        </span>
                        <span className="text-neutral-200 group-hover:text-white truncate">
                          {targetEnt?.name || targetId || "unknown"}
                        </span>
                      </div>
                      <div className="flex items-center gap-1 text-[10px] text-neutral-500 shrink-0">
                        {targetEnt && <span className="capitalize">{targetEnt.type}</span>}
                        <Maximize2 className="w-3 h-3 text-neutral-500 group-hover:text-[#00e5ff]" />
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
      <div className="p-3 rounded-lg border border-[#1f222b] bg-[#151821] space-y-2.5">
        <div className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400 flex items-center justify-between">
          <span>Provenance Pipeline</span>
          <span className="font-mono text-[#00e5ff] text-[9px]">REAL DATA CHAIN</span>
        </div>
        <div className="flex items-center justify-between text-[10px] font-mono">
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[#00e5ff]" />
            <span className="text-white font-semibold">ENTITY</span>
            <span className="text-neutral-500 max-w-[65px] truncate">{entity.id}</span>
          </div>
          <span className="text-neutral-600">→</span>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[#35d07f]" />
            <span className="text-white font-semibold">EVIDENCE</span>
            <span className="text-neutral-500 max-w-[65px] truncate">
              {provenance?.evidence[0]?.evidence_name || (provenance?.trace_level === "none" ? "None" : "—")}
            </span>
          </div>
          <span className="text-neutral-600">→</span>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[#b28dff]" />
            <span className="text-white font-semibold">SESSION</span>
            <span className="text-neutral-500 max-w-[65px] truncate">
              {provenance?.source_session_ids?.[0] || (provenance?.trace_level === "none" ? "None" : "—")}
            </span>
          </div>
          <span className="text-neutral-600">→</span>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="w-2 h-2 rounded-full bg-[#f5a623]" />
            <span className="text-white font-semibold">RECON</span>
            <span className="text-neutral-500 max-w-[65px] truncate">
              {String(world?.metadata?.reconstruction?.backend ?? "unrecorded")}
            </span>
          </div>
        </div>
      </div>

      {/* Traceable Source Evidence -- real trace, never fabricated */}
      {(tab === "all" || tab === "evidence") && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <Camera className="w-3.5 h-3.5 text-[#35d07f]" />
            <span>Source Evidence</span>
          </h4>
          <div className="space-y-2">
            {provenanceLoading ? (
              <p className="text-neutral-500 italic p-2">Tracing evidence…</p>
            ) : !provenance || provenance.evidence.length === 0 ? (
              <p className="text-neutral-500 italic p-2">
                {provenance?.reason || "No evidence linked to this entity or its capture session."}
              </p>
            ) : (
              <>
                {provenance.trace_level === "session" && (
                  <p className="text-neutral-500 text-[10px] px-0.5">
                    Traced at session level: this entity&apos;s version came from this session&apos;s
                    capture, not a specific frame within it.
                  </p>
                )}
                {provenance.evidence.map((ev) => {
                  const isViewingImage = previewImageId === ev.evidence_id;
                  return (
                    <div
                      key={ev.evidence_id}
                      className="p-2.5 rounded-lg border border-[#1f222b] bg-[#151821] text-[11px] space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono font-semibold text-white truncate">
                          {ev.evidence_name}
                        </span>
                        <span className="font-mono text-[10px] text-[#35d07f] px-1 rounded bg-[#35d07f]/10 shrink-0">
                          {ev.evidence_type}
                        </span>
                      </div>

                      <div className="flex items-center justify-between pt-1 border-t border-neutral-800">
                        <a
                          href={`/evidence/${ev.evidence_id}`}
                          onClick={() => onTraceEvidence?.(ev.evidence_id)}
                          className="text-[10px] text-[#00e5ff] hover:underline flex items-center gap-1 cursor-pointer"
                        >
                          <Maximize2 className="w-3 h-3" />
                          <span>Open Evidence</span>
                        </a>

                        {ev.evidence_type === "photo" && (
                          <button
                            type="button"
                            onClick={() => setPreviewImageId(isViewingImage ? null : ev.evidence_id)}
                            className="text-[10px] text-neutral-400 hover:text-white flex items-center gap-1 cursor-pointer"
                          >
                            <ImageIcon className="w-3 h-3" />
                            <span>{isViewingImage ? "Hide Image" : "View Photo"}</span>
                          </button>
                        )}
                      </div>

                      {isViewingImage && (
                        <div className="pt-2 border-t border-neutral-800 space-y-1">
                          <div className="relative w-full h-36 rounded overflow-hidden bg-black border border-neutral-800">
                            <img
                              src={`/api/worlds/${worldId}/evidence/${ev.evidence_id}/image`}
                              alt={ev.evidence_name}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                e.currentTarget.style.display = "none";
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </>
            )}
          </div>
        </div>
      )}

      {/* Observations: WHERE THIS CAME FROM */}
      {(tab === "all" || tab === "observations") && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <FileSearch className="w-3.5 h-3.5 text-[#ffb84d]" />
            <span>Observations ({observations.length})</span>
          </h4>
          {observations.length === 0 ? (
            <p className="text-neutral-500 italic">No explicit observations attached.</p>
          ) : (
            observations.map((obs) => (
              <div
                key={obs.id}
                className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
                style={{ borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-medium text-[#00e5ff]">
                    {obs.sensor_type}
                  </span>
                  {obs.confidence != null && (
                    <span className="text-neutral-400 font-mono-num">
                      {(obs.confidence * 100).toFixed(0)}% conf
                    </span>
                  )}
                </div>
                {obs.metadata && (
                  <div className="space-y-1 pt-1 border-t border-neutral-800">
                    {Object.entries(obs.metadata).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <span className="text-neutral-400 truncate">{k}</span>
                        <span className="font-mono text-white text-right truncate">
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
      {(tab === "all" || tab === "provenance") && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-[#35d07f]" />
            <span>Provenance & Lineage</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between">
              <span className="text-neutral-400">Derivation</span>
              <span className="font-mono font-medium text-white">
                {entity.provenance || "Not recorded"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Source Session</span>
              <span className="text-neutral-300 font-mono">
                {provenance?.source_session_ids?.join(", ") || "Unassigned"}
              </span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-neutral-400 shrink-0">WorldStore Version</span>
              <span className="text-[#00e5ff] font-mono break-all text-right">
                {provenanceLoading ? "…" : provenance?.version_id ?? "Not compiled"}
              </span>
            </div>
            {entity.semantic_labels && entity.semantic_labels.length > 0 && (
              <div className="flex justify-between">
                <span className="text-neutral-400">Semantic Labels</span>
                <span className="font-mono text-white">
                  {entity.semantic_labels.join(", ")}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Uncertainty */}
      {(tab === "all" || tab === "uncertainty") && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-[#f5a623]" />
            <span>Uncertainty Model</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between">
              <span className="text-neutral-400">Certainty Score</span>
              <span className="font-mono text-white">
                {conf !== null ? conf.toFixed(3) : "Not available"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Inlier Quality</span>
              <span className="text-white">
                {conf !== null ? (conf < 0.3 ? "Noise candidate" : "Robust planar fit") : "Not available"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function WorldOverview({
  world,
  worldId,
  onOpenRoomConstruction,
  onExport,
}: {
  world: WorldIR | null;
  worldId: string;
  onOpenRoomConstruction?: () => void;
  onExport?: (format: "worldir" | "ply" | "cameras" | "report") => void;
}) {
  if (!world) {
    return (
      <div className="p-4 text-center text-neutral-500 space-y-3">
        <p>No 3D WorldIR compiled for this world yet.</p>
        {onOpenRoomConstruction && (
          <button
            type="button"
            onClick={onOpenRoomConstruction}
            className="px-3.5 py-1.5 rounded text-xs bg-[#00e5ff] text-black font-medium hover:bg-[#33ebff] transition-colors cursor-pointer"
          >
            Open Room Construction
          </button>
        )}
      </div>
    );
  }

  const meta = world.metadata || {};
  const nEnt = Object.keys(world.entities || {}).length;

  return (
    <div className="space-y-4">
      {/* World Card */}
      <div
        className="p-3 rounded-lg border bg-[#151821]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="text-[11px] uppercase tracking-wider text-neutral-400">
          World Specification
        </div>
        <div className="text-sm font-semibold text-white mt-0.5 font-mono">
          {world.id || worldId}
        </div>
        <div className="text-xs text-neutral-400 mt-1">
          {world.name || "Canonical WorldIR"}
        </div>
      </div>

      {/* Pipeline Stage Facts */}
      <div className="space-y-2">
        <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-[#00e5ff]" />
          <span>Stage Facts & Diagnostics</span>
        </h4>
        <div
          className="p-3 rounded-lg border bg-[#151821] space-y-2.5 text-[11px]"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex justify-between">
            <span className="text-neutral-400">Reconstruction Backend</span>
            <span className="font-mono text-white">
              {meta.reconstruction?.backend || "Unavailable"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">Registration Status</span>
            <span className="font-mono text-[#2ecc71]">
              {meta.reconstruction?.registration_status || "Unavailable"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">Registered Cameras</span>
            <span className="font-mono text-white">
              {meta.reconstruction?.cameras_registered != null
                ? `${meta.reconstruction.cameras_registered} / ${meta.reconstruction.cameras_input ?? "—"}`
                : "Unavailable"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">Compiled Entities</span>
            <span className="font-mono text-white">{nEnt}</span>
          </div>
        </div>
      </div>

      {/* Metric Scale State */}
      <div className="space-y-2">
        <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
          <Compass className="w-3.5 h-3.5 text-[#35d07f]" />
          <span>Metric Scale Calibration</span>
        </h4>
        <div
          className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
          style={{ borderColor: "var(--border)" }}
        >
          <div className="flex justify-between">
            <span className="text-neutral-400">Scale State</span>
            <span className="font-mono font-medium text-[#00e5ff]">
              {meta.scale?.state || "Unavailable"}
            </span>
          </div>
          <p className="text-neutral-400 text-[11px] leading-relaxed pt-1 border-t border-neutral-800">
            {meta.scale?.note || "Unavailable"}
          </p>
        </div>
      </div>

      {/* Monocular Depth Metricization */}
      {meta.depth && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-[#b28dff]" />
            <span>Monocular Depth Fusion</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[#151821] space-y-2 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="flex justify-between">
              <span className="text-neutral-400">Depth Model</span>
              <span className="font-mono text-white">
                {meta.depth.model || "Unavailable"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Metricized Maps</span>
              <span className="font-mono text-white">
                {meta.depth.metricized != null
                  ? `${meta.depth.metricized} / ${meta.depth.maps ?? "—"}`
                  : "Unavailable"}
              </span>
            </div>
            <p className="text-neutral-400 text-[11px] leading-relaxed pt-1 border-t border-neutral-800">
              {meta.depth.note || "Unavailable"}
            </p>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="pt-2 border-t border-neutral-800 space-y-2">
        {onOpenRoomConstruction && (
          <button
            type="button"
            onClick={onOpenRoomConstruction}
            className="w-full py-2 rounded font-medium text-xs bg-[#151821] hover:bg-[#1a1f2c] border border-[#1f222b] text-neutral-200 hover:text-white transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Workflow className="w-3.5 h-3.5 text-[#00e5ff]" />
            <span>Room Construction Workflow</span>
          </button>
        )}

        {onExport && (
          <button
            type="button"
            onClick={() => onExport("worldir")}
            className="w-full py-2 rounded font-medium text-xs bg-[#151821] hover:bg-[#1a1f2c] border border-[#1f222b] text-neutral-200 hover:text-white transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-[#35d07f]" />
            <span>Export WorldIR (JSON)</span>
          </button>
        )}
      </div>
    </div>
  );
}
