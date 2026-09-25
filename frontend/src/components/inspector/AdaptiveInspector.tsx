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
    changes: { type?: string; confidence?: number; semantic_labels?: string[] },
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
  onFrame: () => void;
  onTraceEvidence?: (evidenceId: string) => void;
  onCommitCorrection?: (
    entityId: string,
    changes: { type?: string; confidence?: number; semantic_labels?: string[] },
    commitMessage: string
  ) => Promise<void>;
  isEditing: boolean;
  setIsEditing: (v: boolean) => void;
  previewImageId: string | null;
  setPreviewImageId: (id: string | null) => void;
}) {
  const conf = typeof entity.confidence === "number" ? entity.confidence : 0.5;
  const hexColor = (TYPE_COLORS[entity.type] || TYPE_COLORS.default)
    .toString(16)
    .padStart(6, "0");

  const pos = entity.transform?.position;
  const observations = entity.observations || [];
  const relationships = entity.relationships || [];

  // Correction Form State
  const [editedType, setEditedType] = useState(entity.type);
  const [editedConfidence, setEditedConfidence] = useState(conf);
  const [editedLabels, setEditedLabels] = useState((entity.semantic_labels || []).join(", "));
  const [commitMessage, setCommitMessage] = useState(`Review & verify ${entity.id}`);
  const [submitting, setSubmitting] = useState(false);
  const [committedSuccess, setCommittedSuccess] = useState(false);

  // Real, entity-specific trace-to-evidence -- refetched whenever the
  // selected entity (or world) changes. Never falls back to fabricated
  // per-entity data; an honest loading/empty/error state throughout.
  // `EntityDetails` is remounted via `key={entity.id}` in the parent, so
  // this state starts fresh (undefined = not yet loaded) on every entity
  // switch -- no manual reset needed, and no synchronous setState in the
  // effect body.
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

  const handleSaveCorrection = async () => {
    if (!onCommitCorrection) return;
    setSubmitting(true);
    try {
      await onCommitCorrection(
        entity.id,
        {
          type: editedType,
          confidence: editedConfidence,
          semantic_labels: editedLabels.split(",").map((s) => s.trim()).filter(Boolean),
        },
        commitMessage
      );
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

  // Derivation status
  const derivationBadge = entity.provenance === "OBSERVED"
    ? { label: "Directly Observed", color: "text-[#35d07f] bg-[#35d07f]/10 border-[#35d07f]/30" }
    : entity.provenance === "PROCEDURAL"
    ? { label: "Procedurally Extruded", color: "text-[#b28dff] bg-[#b28dff]/10 border-[#b28dff]/30" }
    : { label: "Inferred Planar Primitive", color: "text-[#00e5ff] bg-[#00e5ff]/10 border-[#00e5ff]/30" };

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
            <span className="font-mono text-white">{entity.provenance || "INFERRED"}</span>
          </div>
        </div>

        {/* Confidence Meter */}
        <div>
          <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
            <span>Spatial Confidence</span>
            <span className="font-mono-num text-white">
              {(conf * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full h-1.5 rounded-full bg-neutral-800 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${Math.max(5, conf * 100)}%`,
                background: conf > 0.7 ? "#2ecc71" : conf > 0.4 ? "#f5a623" : "#e74c3c",
              }}
            />
          </div>
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
            onClick={() => setIsEditing(!isEditing)}
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

      {/* Review / Correction Form */}
      {isEditing && (
        <div className="p-3 rounded-lg border border-[#00e5ff]/40 bg-[#121622] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#00e5ff] flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              <span>Architectural Correction</span>
            </span>
            <button
              type="button"
              onClick={() => setIsEditing(false)}
              className="text-neutral-400 hover:text-white cursor-pointer"
            >
              <X className="w-3 h-3" />
            </button>
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-neutral-400 uppercase tracking-wider">Semantic Classification</label>
            <select
              value={editedType}
              onChange={(e) => setEditedType(e.target.value)}
              className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white focus:border-[#00e5ff] focus:outline-none cursor-pointer"
            >
              <option value="room">Room (Enclosed Space)</option>
              <option value="wall">Wall (Vertical Partition)</option>
              <option value="door">Door (Passage Opening)</option>
              <option value="window">Window (Glazing Opening)</option>
              <option value="floor">Floor (Walking Surface)</option>
              <option value="ceiling">Ceiling (Upper Boundary)</option>
              <option value="stair">Stair (Circulation Element)</option>
              <option value="corridor">Corridor (Circulation Space)</option>
              <option value="roof">Roof (Exterior Covering)</option>
              <option value="column">Column (Vertical Structural Member)</option>
              <option value="beam">Beam (Horizontal Structural Member)</option>
              <option value="object">Object (Furniture / Equipment)</option>
            </select>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[10px] text-neutral-400 uppercase tracking-wider">
              <span>Adjusted Confidence</span>
              <span className="font-mono text-[#00e5ff]">{(editedConfidence * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={editedConfidence}
              onChange={(e) => setEditedConfidence(parseFloat(e.target.value))}
              className="w-full accent-[#00e5ff] cursor-pointer"
            />
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-neutral-400 uppercase tracking-wider">Semantic Labels</label>
            <input
              type="text"
              value={editedLabels}
              onChange={(e) => setEditedLabels(e.target.value)}
              placeholder="e.g. partition_wall, dry_wall"
              className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none font-mono"
            />
          </div>

          <div className="space-y-1">
            <label className="text-[10px] text-neutral-400 uppercase tracking-wider">Commit Rationale</label>
            <input
              type="text"
              value={commitMessage}
              onChange={(e) => setCommitMessage(e.target.value)}
              placeholder="Rationale for this WorldStore version commit"
              className="w-full h-7 px-2 rounded bg-[#0e1013] border border-neutral-700 text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none font-mono"
            />
          </div>

          <button
            type="button"
            disabled={submitting}
            onClick={handleSaveCorrection}
            className={`w-full py-1.5 rounded font-medium text-xs flex items-center justify-center gap-1.5 transition-colors ${
              committedSuccess
                ? "bg-[#2ecc71] text-black"
                : "bg-[#00e5ff] hover:bg-[#33ebff] text-black cursor-pointer"
            }`}
          >
            {submitting ? (
              <span>Persisting to WorldStore...</span>
            ) : committedSuccess ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span>Version Committed!</span>
              </>
            ) : (
              <span>Commit World Version</span>
            )}
          </button>
        </div>
      )}

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

      {/* Spatial Relationships Section */}
      {(tab === "all" || tab === "geometry") && (
        <div className="space-y-2">
          <h4 className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400 flex items-center gap-1.5">
            <Building className="w-3.5 h-3.5 text-[#ffb84d]" />
            <span>Spatial Relationships ({relationships.length})</span>
          </h4>
          <div
            className="p-3 rounded-lg border bg-[#151821] space-y-1.5 text-[11px]"
            style={{ borderColor: "var(--border)" }}
          >
            {relationships.length === 0 ? (
              <p className="text-neutral-500 italic">No relationships linked</p>
            ) : (
              relationships.map((rel: { predicate?: string; kind?: string; type?: string; target_entity_id?: string; target_id?: string }, idx) => (
                <div key={idx} className="flex justify-between items-center font-mono">
                  <span className="text-[#ffb84d] uppercase text-[10px]">{String(rel.kind || rel.type || "rel")}</span>
                  <span className="text-neutral-300">{String(rel.target_entity_id || rel.target_id || "unknown")}</span>
                </div>
              ))
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
              {String(world?.metadata?.reconstruction?.backend || "Colmap SfM")}
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
                {entity.provenance || "INFERRED"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Source Session</span>
              <span className="text-neutral-300 font-mono">
                {provenance?.source_session_ids?.join(", ") || "Unassigned"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">WorldStore Version</span>
              <span className="text-[#00e5ff] font-mono">
                {world?.version != null ? `v${world.version}.0.0` : "v1.0.0"}
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
                {(conf).toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Inlier Quality</span>
              <span className="text-white">
                {conf < 0.3 ? "Noise candidate" : "Robust planar fit"}
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
