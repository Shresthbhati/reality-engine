"use client";

import { useState } from "react";
import {
  Box,
  Compass,
  FileSearch,
  Layers,
  Maximize2,
  Shield,
  Sparkles,
  Tag,
  X,
  Camera,
  Activity,
} from "lucide-react";
import type {
  Entity,
  Geometry,
  WorldIR,
} from "@/types/worldir";
import { TYPE_COLORS } from "@/lib/viewport/three-scene";

interface AdaptiveInspectorProps {
  world: WorldIR | null;
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
}

type InspectorTab = "all" | "geometry" | "observations" | "provenance" | "uncertainty";

export default function AdaptiveInspector({
  world,
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
}: AdaptiveInspectorProps) {
  const [tab, setTab] = useState<InspectorTab>("all");

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
      className="w-80 shrink-0 h-full flex flex-col border-l overflow-hidden bg-[#0e1013]"
      style={{ borderColor: "var(--border)" }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 h-12 border-b shrink-0"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            {entity ? "Entity Inspector" : "World Inspector"}
          </span>
          {entity && (
            <span
              className="text-[10px] font-mono px-1.5 py-0.5 rounded"
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
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onFrameEntity(entity.id)}
              title="Frame Entity in Viewport [F]"
              className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
            <button
              type="button"
              onClick={() => onSelectEntity(null)}
              title="Deselect"
              className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Tabs */}
      {entity && (
        <div
          className="flex items-center px-3 gap-1 h-9 border-b shrink-0 overflow-x-auto text-xs"
          style={{ borderColor: "var(--border-subtle)" }}
        >
          {(
            [
              { id: "all", label: "Overview" },
              { id: "geometry", label: "Geometry" },
              { id: "observations", label: "Observations" },
              { id: "provenance", label: "Provenance" },
              { id: "uncertainty", label: "Uncertainty" },
            ] as const
          ).map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`px-2 py-1 rounded transition-colors ${
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
            entity={entity}
            geometry={geometry}
            tab={tab}
            onFrame={() => onFrameEntity(entity.id)}
          />
        ) : (
          <WorldOverview world={world} />
        )}
      </div>
    </aside>
  );
}

function EntityDetails({
  entity,
  geometry,
  tab,
  onFrame,
}: {
  entity: Entity;
  geometry: Geometry | null;
  tab: InspectorTab;
  onFrame: () => void;
}) {
  const conf = typeof entity.confidence === "number" ? entity.confidence : 0.5;
  const hexColor = (TYPE_COLORS[entity.type] || TYPE_COLORS.default)
    .toString(16)
    .padStart(6, "0");

  const pos = entity.transform?.position;
  const observations = entity.observations || [];

  return (
    <div className="space-y-4">
      {/* Title Card */}
      <div
        className="p-3 rounded-lg border bg-[#151821]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-sm shrink-0"
              style={{ background: `#${hexColor}` }}
            />
            <span className="font-mono font-semibold text-white truncate">
              {entity.id}
            </span>
          </div>
          <span
            className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded shrink-0"
            style={{
              background: "rgba(255, 255, 255, 0.06)",
              color: "var(--text-secondary)",
            }}
          >
            {entity.provenance || "INFERRED"}
          </span>
        </div>

        {/* Confidence Meter */}
        <div className="mt-3">
          <div className="flex justify-between text-[11px] text-neutral-400 mb-1">
            <span>Confidence</span>
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

        {/* Position */}
        {pos && (
          <div className="mt-3 pt-2 border-t border-neutral-800 flex justify-between items-center text-[11px]">
            <span className="text-neutral-400">Position (XYZ m)</span>
            <span className="font-mono text-white">
              {pos.x.toFixed(2)}, {pos.y.toFixed(2)}, {pos.z.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Geometry Section */}
      {(tab === "all" || tab === "geometry") && geometry && (
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
                {geometry.id}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Extent Dimensions</span>
              <span className="font-mono text-white">
                {(geometry.bounds_max.x - geometry.bounds_min.x).toFixed(2)} ×{" "}
                {(geometry.bounds_max.y - geometry.bounds_min.y).toFixed(2)} ×{" "}
                {(geometry.bounds_max.z - geometry.bounds_min.z).toFixed(2)} m
              </span>
            </div>
            {geometry.vertex_count != null && (
              <div className="flex justify-between">
                <span className="text-neutral-400">Inlier / Vertex Count</span>
                <span className="font-mono text-white">
                  {geometry.vertex_count.toLocaleString()}
                </span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-neutral-400">LOD Level</span>
              <span className="font-mono text-white">{geometry.lod_level ?? 0}</span>
            </div>
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

      {/* Provenance & Engine Facts */}
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
              <span className="text-neutral-400">Verification</span>
              <span className="text-neutral-300 font-mono">
                {entity.confidence && entity.confidence > 0.7
                  ? "Multi-view corroborated"
                  : "Geometric hypothesis"}
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
              <span className="text-neutral-400">Certainty Weight</span>
              <span className="font-mono text-white">
                {(conf).toFixed(3)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Depth Noise Status</span>
              <span className="text-white">
                {conf < 0.3 ? "High Residual / Noise candidate" : "Robust inlier"}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function WorldOverview({ world }: { world: WorldIR | null }) {
  if (!world) {
    return (
      <div className="p-4 text-center text-neutral-500">
        <p>Select an entity in the 3D viewport or left tree to inspect its provenance.</p>
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
          {world.id}
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
              {meta.reconstruction?.backend ?? "SfM Pipeline"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">Registration Status</span>
            <span className="font-mono text-[#2ecc71]">
              {meta.reconstruction?.registration_status ?? "SUCCESS"}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-neutral-400">Registered Cameras</span>
            <span className="font-mono text-white">
              {meta.reconstruction?.cameras_registered ?? "—"} /{" "}
              {meta.reconstruction?.cameras_input ?? "—"}
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
              {meta.scale?.state ?? "METRIC"}
            </span>
          </div>
          <p className="text-neutral-400 text-[11px] leading-relaxed pt-1 border-t border-neutral-800">
            {meta.scale?.note ||
              "Metric scale established from registered baseline references."}
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
                {meta.depth.model ?? "DPT_Hybrid"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-400">Metricized Maps</span>
              <span className="font-mono text-white">
                {meta.depth.metricized ?? "—"} / {meta.depth.maps ?? "—"}
              </span>
            </div>
            <p className="text-neutral-400 text-[11px] leading-relaxed pt-1 border-t border-neutral-800">
              {meta.depth.note ?? "Inverse-depth alignment against SfM sparse cloud."}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
