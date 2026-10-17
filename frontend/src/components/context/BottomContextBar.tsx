"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  ChevronDown,
  ChevronUp,
  Clock,
  Layers,
  Maximize2,
  Sliders,
  Sparkles,
  Terminal,
  Ruler,
  Grid,
} from "lucide-react";
import type {
  Entity,
  WorldIR,
} from "@/types/worldir";
import type { SessionRow } from "@/lib/types";
import { getActivity, isApiError, type ActivityEvent } from "@/lib/activity";
import type { MeasurementResult } from "@/lib/viewport/three-scene";

interface BottomContextBarProps {
  world: WorldIR | null;
  selectedEntity: Entity | null;
  sessions: SessionRow[];
  onFrameSelected: () => void;
  onClearSelection: () => void;
  measurement?: MeasurementResult | null;
  /** The world's current WorldStore version id, or undefined when the world
   * has no committed version. There is no default: "v1.0.0" used to be
   * printed here for worlds whose lineage was empty. */
  activeVersion?: string;
}

type BottomTab = "context" | "timeline" | "processing" | "activity";

export default function BottomContextBar({
  world,
  selectedEntity,
  sessions,
  onFrameSelected,
  onClearSelection,
  measurement,
  activeVersion,
}: BottomContextBarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<BottomTab>("context");
  const [activities, setActivities] = useState<ActivityEvent[]>([]);
  const [activityLoading, setActivityLoading] = useState(true);
  // A failed read leaves `activityError` set so the panel can say the log is
  // unavailable instead of asserting that nothing has been recorded.
  const [activityError, setActivityError] = useState<string | null>(null);

  const meta = world?.metadata || {};

  // A new world restarts the activity read: reset the status during render, not in the effect.
  const [loadedFor, setLoadedFor] = useState(world);
  if (loadedFor !== world) {
    setLoadedFor(world);
    setActivityLoading(true);
    setActivityError(null);
  }

  // Fetch real activity events
  useEffect(() => {
    let mounted = true;
    getActivity()
      .then((events) => {
        if (mounted) {
          setActivities(events.slice(0, 10));
          setActivityLoading(false);
        }
      })
      .catch((err) => {
        if (!mounted) return;
        setActivities([]);
        setActivityError(
          isApiError(err) ? err.message : (err as Error)?.message ?? "unknown error",
        );
        setActivityLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [world]);

  const pos = selectedEntity?.transform?.position;

  return (
    <footer
      className="border-t shrink-0 flex flex-col bg-[var(--bg-surface)] transition-all duration-150 select-none"
      style={{ borderColor: "var(--border)" }}
    >
      {/* Control Bar Header */}
      <div
        className="flex items-center justify-between px-3 h-9 border-b shrink-0 text-xs bg-[var(--bg-surface)]"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        {/* Left: Tab selectors */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => {
              setActiveTab("context");
              setCollapsed(false);
            }}
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors cursor-pointer ${
              activeTab === "context" && !collapsed
                ? "text-[var(--accent)] bg-[rgba(34,184,255,0.12)] font-medium border border-[var(--accent)]/30"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Sliders className="w-3 h-3" />
            <span>Selection & Frame</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab("timeline");
              setCollapsed(false);
            }}
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors cursor-pointer ${
              activeTab === "timeline" && !collapsed
                ? "text-[var(--accent)] bg-[rgba(34,184,255,0.12)] font-medium border border-[var(--accent)]/30"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Clock className="w-3 h-3" />
            <span>Timeline</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab("processing");
              setCollapsed(false);
            }}
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors cursor-pointer ${
              activeTab === "processing" && !collapsed
                ? "text-[var(--accent)] bg-[rgba(34,184,255,0.12)] font-medium border border-[var(--accent)]/30"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>Processing State</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTab("activity");
              setCollapsed(false);
            }}
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors cursor-pointer ${
              activeTab === "activity" && !collapsed
                ? "text-[var(--accent)] bg-[rgba(34,184,255,0.12)] font-medium border border-[var(--accent)]/30"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Activity className="w-3 h-3" />
            <span>Activity</span>
          </button>
        </div>

        {/* Center: Live Measurement readout if active */}
        {measurement && measurement.distance > 0 && (
          <div className="flex items-center gap-2 px-2.5 py-0.5 rounded bg-[var(--accent)]/15 border border-[var(--accent)]/40 text-[var(--accent)] font-mono text-[11px]">
            <Ruler className="w-3 h-3" />
            <span>Distance: <strong>{measurement.distance.toFixed(3)} m</strong></span>
            <span className="text-neutral-500">|</span>
            <span>ΔX: {measurement.delta[0].toFixed(2)}m</span>
            <span>ΔY: {measurement.delta[1].toFixed(2)}m</span>
            <span>ΔZ: {measurement.delta[2].toFixed(2)}m</span>
          </div>
        )}

        {/* Right: Quick collapse toggle & active selection indicator */}
        <div className="flex items-center gap-2">
          {selectedEntity && (
            <div className="flex items-center gap-2 px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--accent)]/40 text-[11px]">
              <span className="text-neutral-400">Selected:</span>
              <span className="font-mono text-white font-semibold">
                {selectedEntity.id}
              </span>
              <button
                type="button"
                onClick={onFrameSelected}
                title="Frame in Viewport [F]"
                className="text-[var(--accent)] hover:text-white p-0.5 cursor-pointer"
              >
                <Maximize2 className="w-3 h-3" />
              </button>
            </div>
          )}

          <div className="flex items-center gap-1.5 text-[11px] font-mono text-neutral-400 px-2 border-l border-neutral-800">
            <span className="text-neutral-500">HEAD:</span>
            <span className="text-white font-medium">{activeVersion ?? "no version"}</span>
          </div>

          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded text-neutral-400 hover:text-white transition-colors cursor-pointer"
            title={collapsed ? "Expand context bar" : "Collapse context bar"}
          >
            {collapsed ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Expandable Context Bar Body */}
      {!collapsed && (
        <div className="p-3 h-28 overflow-y-auto text-xs bg-[var(--bg-surface)]">
          {activeTab === "context" && (
            <div className="grid grid-cols-4 gap-4 h-full">
              {/* Telemetry 1: Active World & Coordinate Frame */}
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)] flex flex-col justify-between">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider font-semibold">
                  Spatial Coordinate Frame
                </span>
                <div className="font-mono text-white font-medium text-xs">
                  {String(world?.coordinate_frame || world?.coordinate_system || "metric_enu")}
                </div>
                <div className="text-[10px] text-neutral-500 font-mono">
                  Scale: {meta.scale?.state || "Calibrated Metric"}
                </div>
              </div>

              {/* Telemetry 2: Selection / Entity Telemetry */}
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)] flex flex-col justify-between">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider font-semibold">
                  Selection Telemetry
                </span>
                {selectedEntity ? (
                  <>
                    <div className="font-mono text-white text-xs truncate">
                      {selectedEntity.id} ({selectedEntity.type})
                    </div>
                    <div className="text-[10px] text-neutral-400 font-mono">
                      {pos ? `XYZ: ${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)} m` : "Pos unavailable"}
                    </div>
                  </>
                ) : (
                  <div className="text-neutral-500 italic text-xs">
                    No entity selected · click 3D mesh or use Nav
                  </div>
                )}
              </div>

              {/* Telemetry 3: Grid & Snap State */}
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)] flex flex-col justify-between">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider font-semibold">
                  Grid & Snap Reference
                </span>
                <div className="font-mono text-[var(--accent)] text-xs">
                  Grid: 1.0 m ENU spacing
                </div>
                <div className="text-[10px] text-neutral-500 font-mono">
                  Snap: Planar Inliers · [G] Toggle
                </div>
              </div>

              {/* Telemetry 4: WorldStore Version Head */}
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)] flex flex-col justify-between">
                <span className="text-[10px] text-neutral-400 uppercase tracking-wider font-semibold">
                  WorldStore Version
                </span>
                <div className="font-mono text-white font-medium text-xs break-all">
                  {activeVersion ?? "—"}
                </div>
                <div className="text-[10px] font-mono" style={{ color: activeVersion ? "var(--success)" : "var(--text-tertiary)" }}>
                  {activeVersion ? "Status: Canonical Head" : "No version committed"}
                </div>
              </div>
            </div>
          )}

          {activeTab === "timeline" && (
            <div className="flex items-center gap-6 h-full px-2">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[var(--accent)]" />
                <div className="text-xs">
                  <div className="text-white font-medium">1. Capture Evidence Ingestion</div>
                  <div className="text-neutral-500 text-[11px] font-mono">
                    {sessions.length} sessions attached
                  </div>
                </div>
              </div>

              <span className="w-8 h-px bg-neutral-800" />

              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[var(--success)]" />
                <div className="text-xs">
                  <div className="text-white font-medium">2. SfM Camera Calibration</div>
                  <div className="text-neutral-500 text-[11px] font-mono">
                    {meta.reconstruction?.cameras_registered != null
                      ? `${meta.reconstruction.cameras_registered} cameras registered`
                      : "Multi-view reconstruction"}
                  </div>
                </div>
              </div>

              <span className="w-8 h-px bg-neutral-800" />

              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-[#b28dff]" />
                <div className="text-xs">
                  <div className="text-white font-medium">3. Structural Plane Promotion</div>
                  <div className="text-neutral-500 text-[11px] font-mono">
                    {Object.keys(world?.entities || {}).length} entities in WorldIR
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "processing" && (
            <div className="grid grid-cols-4 gap-4 h-full">
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)]">
                <div className="text-[10px] text-neutral-400 uppercase">Engine Status</div>
                <div className="text-[var(--success)] font-semibold text-xs mt-0.5 font-mono">
                  {world ? "COMPILED · PERSISTENT" : "STANDBY"}
                </div>
              </div>
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)]">
                <div className="text-[10px] text-neutral-400 uppercase">Reconstruction Backend</div>
                <div className="text-white font-semibold text-xs mt-0.5 font-mono">
                  {meta.reconstruction?.backend || "Unavailable"}
                </div>
              </div>
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)]">
                <div className="text-[10px] text-neutral-400 uppercase">Scale Anchor</div>
                <div className="text-[var(--accent)] font-semibold text-xs mt-0.5 font-mono">
                  {meta.scale?.state || "Unavailable"}
                </div>
              </div>
              <div className="p-2 rounded bg-[var(--bg-elevated)] border border-[var(--border)]">
                <div className="text-[10px] text-neutral-400 uppercase">Depth Model</div>
                <div className="text-white font-semibold text-xs mt-0.5 font-mono">
                  {meta.depth?.model || "Unavailable"}
                </div>
              </div>
            </div>
          )}

          {activeTab === "activity" && (
            <div className="space-y-1.5">
              {activityLoading ? (
                <p className="text-neutral-500 text-xs font-mono">Loading activity log...</p>
              ) : activityError ? (
                // A failed read is not an empty log: say the log is unavailable
                // rather than claiming nothing has been recorded.
                <p className="text-[var(--error)] text-xs font-mono" role="alert">
                  Activity log unavailable: {activityError}
                </p>
              ) : activities.length === 0 ? (
                <p className="text-neutral-500 text-xs font-mono">
                  No activity events recorded for this world yet.
                </p>
              ) : (
                activities.map((a) => (
                  <div key={a.id} className="flex items-center gap-2 text-[11px] font-mono">
                    <span className="text-neutral-500">{a.timestamp ? a.timestamp.slice(11, 19) : "Recent"}</span>
                    <span className="text-[var(--accent)] font-medium">{a.type}</span>
                    <span className="text-neutral-300">{a.label}</span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      )}
    </footer>
  );
}
