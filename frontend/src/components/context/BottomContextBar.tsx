"use client";

import { useState } from "react";
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
} from "lucide-react";
import type {
  Entity,
  PipelineReport,
  WorldIR,
} from "@/types/worldir";
import type { SessionRow } from "@/lib/types";
import type { ActivityEvent } from "@/lib/api/activity";

interface BottomContextBarProps {
  world: WorldIR | null;
  selectedEntity: Entity | null;
  sessions: SessionRow[];
  report: PipelineReport | null;
  activityItems: ActivityEvent[];
  onFrameSelected: () => void;
  onClearSelection: () => void;
}

type BottomTab = "context" | "timeline" | "processing" | "activity";

export default function BottomContextBar({
  world,
  selectedEntity,
  sessions,
  report,
  activityItems,
  onFrameSelected,
  onClearSelection,
}: BottomContextBarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<BottomTab>("context");

  const meta = world?.metadata || {};

  return (
    <footer
      className="border-t shrink-0 flex flex-col bg-[#0e1013] transition-all duration-150"
      style={{ borderColor: "var(--border)" }}
    >
      {/* Control Bar Header */}
      <div
        className="flex items-center justify-between px-3 h-9 border-b shrink-0 text-xs"
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
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors ${
              activeTab === "context" && !collapsed
                ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-medium"
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
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors ${
              activeTab === "timeline" && !collapsed
                ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-medium"
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
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors ${
              activeTab === "processing" && !collapsed
                ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-medium"
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
            className={`flex items-center gap-1.5 px-2.5 h-6 rounded transition-colors ${
              activeTab === "activity" && !collapsed
                ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-medium"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <Activity className="w-3 h-3" />
            <span>Activity</span>
          </button>
        </div>

        {/* Right: Quick collapse toggle & active selection indicator */}
        <div className="flex items-center gap-2">
          {selectedEntity && (
            <div className="flex items-center gap-2 px-2 py-0.5 rounded bg-[#151821] border border-[#00e5ff]/40 text-[11px]">
              <span className="text-neutral-400">Selected:</span>
              <span className="font-mono text-white font-semibold">
                {selectedEntity.id}
              </span>
              <button
                type="button"
                onClick={onFrameSelected}
                title="Frame in Viewport [F]"
                className="text-[#00e5ff] hover:text-white p-0.5"
              >
                <Maximize2 className="w-3 h-3" />
              </button>
            </div>
          )}

          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            title={collapsed ? "Expand panel" : "Collapse panel"}
            className="p-1 rounded text-neutral-400 hover:text-white hover:bg-neutral-800 transition-colors"
          >
            {collapsed ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Expandable Tray Body */}
      {!collapsed && (
        <div className="p-3 h-28 overflow-y-auto text-xs font-mono-num text-neutral-300">
          {activeTab === "context" && (
            <div className="flex items-center justify-between h-full">
              {selectedEntity ? (
                <div className="flex items-center gap-8">
                  <div>
                    <div className="text-[10px] text-neutral-500 uppercase tracking-wider">
                      Entity Identity
                    </div>
                    <div className="text-white font-mono font-semibold text-sm">
                      {selectedEntity.id}
                    </div>
                    <div className="text-neutral-400 text-[11px]">
                      Role: {selectedEntity.type}
                    </div>
                  </div>

                  <div>
                    <div className="text-[10px] text-neutral-500 uppercase tracking-wider">
                      Confidence
                    </div>
                    <div className="text-[#2ecc71] font-mono font-semibold text-sm">
                      {selectedEntity.confidence != null
                        ? `${(selectedEntity.confidence * 100).toFixed(1)}%`
                        : "—"}
                    </div>
                    <div className="text-neutral-400 text-[11px]">
                      {selectedEntity.provenance || "INFERRED"}
                    </div>
                  </div>

                  <div>
                    <div className="text-[10px] text-neutral-500 uppercase tracking-wider">
                      Coordinates (meters)
                    </div>
                    <div className="text-white font-mono text-xs">
                      {selectedEntity.transform?.position
                        ? `X: ${selectedEntity.transform.position.x.toFixed(2)}  Y: ${selectedEntity.transform.position.y.toFixed(2)}  Z: ${selectedEntity.transform.position.z.toFixed(2)}`
                        : "Origin (0,0,0)"}
                    </div>
                  </div>

                  <div>
                    <div className="text-[10px] text-neutral-500 uppercase tracking-wider">
                      Observations
                    </div>
                    <div className="text-white font-mono text-xs">
                      {selectedEntity.observations?.length || 0} sensor attachments
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-3 text-neutral-400">
                  <Terminal className="w-4 h-4 text-neutral-500" />
                  <span>
                    No entity selected. Click any surface box in the 3D viewport or choose from the Entities list.
                  </span>
                </div>
              )}
            </div>
          )}

          {activeTab === "timeline" && (
            <div className="flex items-center gap-6 h-full overflow-x-auto">
              {report ? (
                (() => {
                  const stages = (report.stages ?? {}) as Record<string, Record<string, unknown> | undefined>;
                  const recon = stages.reconstruction;
                  const scale = stages.scale;
                  const compile = stages.compile;
                  const steps: { label: string; detail: string }[] = [];
                  if (recon) {
                    steps.push({
                      label: `Reconstruction (${String(recon.backend ?? "unknown backend")})`,
                      detail: `${recon.cameras_registered ?? 0}/${recon.cameras_input ?? 0} cameras registered · ${recon.points ?? 0} points`,
                    });
                  }
                  if (scale) {
                    steps.push({
                      label: "Scale",
                      detail: `${scale.state ?? "unknown"}${scale.meters_per_unit != null ? ` · ${Number(scale.meters_per_unit).toFixed(4)} m/unit` : ""}`,
                    });
                  }
                  if (compile) {
                    steps.push({
                      label: "WorldIR Compilation",
                      detail: `${compile.entities ?? 0} entities · ${compile.measurements ?? 0} measurements · ${compile.relationships ?? 0} relationships`,
                    });
                  }
                  if (steps.length === 0) {
                    return (
                      <div className="flex items-center gap-3 text-neutral-400">
                        <Clock className="w-4 h-4 text-neutral-500" />
                        <span>Pipeline report has no stage data.</span>
                      </div>
                    );
                  }
                  return steps.map((step, i) => (
                    <div key={step.label} className="flex items-center gap-2">
                      {i > 0 && <span className="w-8 h-px bg-neutral-800 -ml-4 mr-2" />}
                      <span className="w-2.5 h-2.5 rounded-full bg-[#00e5ff]" />
                      <div className="text-xs">
                        <div className="text-white font-medium">{step.label}</div>
                        <div className="text-neutral-500 text-[11px]">{step.detail}</div>
                      </div>
                    </div>
                  ));
                })()
              ) : (
                <div className="flex items-center gap-3 text-neutral-400">
                  <Clock className="w-4 h-4 text-neutral-500" />
                  <span>No pipeline report available — this world has not been compiled yet.</span>
                </div>
              )}
            </div>
          )}

          {activeTab === "processing" && (
            <div className="grid grid-cols-4 gap-4 h-full">
              <div className="p-2 rounded bg-[#151821] border border-[#1f222b]">
                <div className="text-[10px] text-neutral-400 uppercase">Engine Status</div>
                <div className="text-[#2ecc71] font-semibold text-xs mt-0.5">READY · DOCK_ONLINE</div>
              </div>
              <div className="p-2 rounded bg-[#151821] border border-[#1f222b]">
                <div className="text-[10px] text-neutral-400 uppercase">Reconstruction Backend</div>
                <div className="text-white font-semibold text-xs mt-0.5">
                  {meta.reconstruction?.backend ?? "COLMAP / pycolmap"}
                </div>
              </div>
              <div className="p-2 rounded bg-[#151821] border border-[#1f222b]">
                <div className="text-[10px] text-neutral-400 uppercase">Metric Scale Anchor</div>
                <div className="text-[#00e5ff] font-semibold text-xs mt-0.5">
                  {meta.scale?.state ?? "CALIBRATED_METRIC"}
                </div>
              </div>
              <div className="p-2 rounded bg-[#151821] border border-[#1f222b]">
                <div className="text-[10px] text-neutral-400 uppercase">World Version State</div>
                <div className="text-white font-semibold text-xs mt-0.5">v1 (IMMUTABLE)</div>
              </div>
            </div>
          )}

          {activeTab === "activity" && (
            <div className="space-y-1.5">
              {activityItems.length === 0 ? (
                <div className="flex items-center gap-3 text-neutral-400">
                  <Activity className="w-4 h-4 text-neutral-500" />
                  <span>No recent activity.</span>
                </div>
              ) : (
                activityItems.slice(0, 20).map((item) => (
                  <div key={item.id} className="flex items-center gap-2 text-[11px]">
                    <span className="text-neutral-500 font-mono">{item.timestamp}</span>
                    <span className="text-[#00e5ff] font-medium">{item.type.toLowerCase()}</span>
                    <span className="text-neutral-400">{item.label}</span>
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
