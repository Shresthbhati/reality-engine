"use client";

import { useState } from "react";
import {
  Globe,
  Camera,
  ShieldCheck,
  Bookmark,
  GitBranch,
  Box,
  ChevronDown,
  Search,
  Maximize2,
  ExternalLink,
} from "lucide-react";
import type {
  WorldIR,
} from "@/types/worldir";
import type {
  EvidenceRow,
  SessionRow,
  WorldRow,
  WorldVersionRow,
} from "@/lib/types";
import { TYPE_COLORS } from "@/lib/viewport/three-scene";

interface WorldNavPanelProps {
  worlds: WorldRow[];
  activeWorldId: string;
  onSelectWorld: (id: string) => void;
  worldIR: WorldIR | null;
  sessions: SessionRow[];
  evidence: EvidenceRow[];
  versions: WorldVersionRow[];
  selectedEntityId: string | null;
  onSelectEntity: (id: string | null) => void;
  onFrameEntity: (id: string) => void;
  onSelectSession?: (id: string) => void;
}

type NavSection = "entities" | "sessions" | "evidence" | "versions";

export default function WorldNavPanel({
  worlds,
  activeWorldId,
  onSelectWorld,
  worldIR,
  sessions,
  evidence,
  versions,
  selectedEntityId,
  onSelectEntity,
  onFrameEntity,
}: WorldNavPanelProps) {
  const [section, setSection] = useState<NavSection>("entities");
  const [filterText, setFilterText] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("all");

  const entitiesList = Object.values(worldIR?.entities || {});
  const filteredEntities = entitiesList
    .filter((e) => {
      const matchesText =
        !filterText ||
        e.id.toLowerCase().includes(filterText.toLowerCase()) ||
        e.type.toLowerCase().includes(filterText.toLowerCase());
      const matchesType = typeFilter === "all" || e.type === typeFilter;
      return matchesText && matchesType;
    })
    .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));

  const typesCount = entitiesList.reduce<Record<string, number>>((acc, e) => {
    acc[e.type] = (acc[e.type] || 0) + 1;
    return acc;
  }, {});

  const currentWorld = worlds.find((w) => w.id === activeWorldId);

  return (
    <nav
      className="w-64 shrink-0 h-full flex flex-col border-r overflow-hidden bg-[#0e1013]"
      style={{ borderColor: "var(--border)" }}
    >
      {/* World Selector Header */}
      <div
        className="p-3 border-b shrink-0 flex flex-col gap-1.5"
        style={{ borderColor: "var(--border)" }}
      >
        <label className="text-[10px] font-semibold uppercase tracking-wider text-neutral-400">
          Spatial World
        </label>
        <div className="relative">
          <select
            value={activeWorldId}
            onChange={(e) => onSelectWorld(e.target.value)}
            className="w-full h-8 pl-2.5 pr-7 rounded bg-[#151821] border border-[#1f222b] text-xs text-white font-medium appearance-none focus:border-[#00e5ff] focus:outline-none cursor-pointer"
          >
            {worlds.map((w) => (
              <option key={w.id} value={w.id} className="bg-[#151821] text-white">
                {w.name} {w.id === "world-compiled-seed42" ? "★ (Vertical Slice)" : ""}
              </option>
            ))}
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-neutral-400 absolute right-2.5 top-2.5 pointer-events-none" />
        </div>
      </div>

      {/* Navigation Sub-sections */}
      <div
        className="flex items-center px-2 py-1.5 border-b gap-1 shrink-0 overflow-x-auto text-xs"
        style={{ borderColor: "var(--border-subtle)" }}
      >
        <NavTabButton
          active={section === "entities"}
          onClick={() => setSection("entities")}
          icon={Box}
          label="Entities"
          count={entitiesList.length}
        />
        <NavTabButton
          active={section === "sessions"}
          onClick={() => setSection("sessions")}
          icon={Camera}
          label="Sessions"
          count={sessions.length}
        />
        <NavTabButton
          active={section === "evidence"}
          onClick={() => setSection("evidence")}
          icon={ShieldCheck}
          label="Evidence"
          count={evidence.length}
        />
        <NavTabButton
          active={section === "versions"}
          onClick={() => setSection("versions")}
          icon={GitBranch}
          label="Versions"
          count={versions.length}
        />
      </div>

      {/* Section Content */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {section === "entities" && (
          <div className="flex flex-col h-full">
            {/* Search & Filter Bar */}
            <div className="p-2 border-b space-y-1.5" style={{ borderColor: "var(--border-subtle)" }}>
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-neutral-500 absolute left-2.5 top-2" />
                <input
                  type="text"
                  placeholder="Filter entities..."
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                  className="w-full h-7 pl-7 pr-2 rounded bg-[#151821] border border-[#1f222b] text-xs text-white placeholder-neutral-500 focus:border-[#00e5ff] focus:outline-none"
                />
              </div>

              <div className="flex items-center gap-1 overflow-x-auto pb-0.5 text-[11px]">
                <button
                  type="button"
                  onClick={() => setTypeFilter("all")}
                  className={`px-1.5 py-0.5 rounded transition-colors ${
                    typeFilter === "all" ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-semibold" : "text-neutral-400"
                  }`}
                >
                  All ({entitiesList.length})
                </button>
                {Object.entries(typesCount).map(([type, count]) => (
                  <button
                    key={type}
                    type="button"
                    onClick={() => setTypeFilter(type)}
                    className={`px-1.5 py-0.5 rounded transition-colors whitespace-nowrap ${
                      typeFilter === type ? "text-[#00e5ff] bg-[rgba(0,229,255,0.12)] font-semibold" : "text-neutral-400"
                    }`}
                  >
                    {type} ({count})
                  </button>
                ))}
              </div>
            </div>

            {/* Entities List */}
            <div className="flex-1 p-1.5 space-y-0.5 overflow-y-auto">
              {filteredEntities.length === 0 ? (
                <div className="p-4 text-center text-xs text-neutral-500 italic">
                  {entitiesList.length === 0 ? "No entities in world." : "No matching entities."}
                </div>
              ) : (
                filteredEntities.map((e) => {
                  const isSel = e.id === selectedEntityId;
                  const hexColor = (TYPE_COLORS[e.type] || TYPE_COLORS.default).toString(16).padStart(6, "0");
                  return (
                    <div
                      key={e.id}
                      onClick={() => onSelectEntity(e.id)}
                      className={`group flex items-center justify-between px-2.5 py-1.5 rounded text-xs cursor-pointer transition-colors ${
                        isSel ? "bg-[#151821] border border-[#00e5ff]/50" : "hover:bg-[#151821]/70 border border-transparent"
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: `#${hexColor}` }} />
                        <span className={`font-mono truncate ${isSel ? "text-white font-medium" : "text-neutral-300"}`}>
                          {e.id}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0 ml-2">
                        <span className="text-[10px] font-mono-num text-neutral-500">
                          {e.confidence != null ? `${(e.confidence * 100).toFixed(0)}%` : "—"}
                        </span>
                        <button
                          type="button"
                          onClick={(evt) => {
                            evt.stopPropagation();
                            onSelectEntity(e.id);
                            onFrameEntity(e.id);
                          }}
                          title="Frame in 3D"
                          className="opacity-0 group-hover:opacity-100 p-0.5 text-neutral-400 hover:text-[#00e5ff]"
                        >
                          <Maximize2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {section === "sessions" && (
          <div className="p-2 space-y-1.5">
            {sessions.length === 0 ? (
              <p className="p-4 text-center text-xs text-neutral-500">No sessions attached.</p>
            ) : (
              sessions.map((s) => (
                <div
                  key={s.id}
                  className="p-2.5 rounded-md border border-[#1f222b] bg-[#151821] text-xs space-y-1 hover:border-neutral-600 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white truncate">{s.name}</span>
                    <span
                      className={`text-[10px] px-1 rounded font-mono ${
                        s.state === "COMPLETE" ? "text-[#2ecc71] bg-[#2ecc71]/10" : "text-[#f5a623] bg-[#f5a623]/10"
                      }`}
                    >
                      {s.state}
                    </span>
                  </div>
                  <div className="flex justify-between text-[11px] text-neutral-400">
                    <span>{s.evidenceCount} Captures</span>
                    <span>{s.capturedAt ?? "Recent"}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {section === "evidence" && (
          <div className="p-2 space-y-1.5">
            {evidence.length === 0 ? (
              <p className="p-4 text-center text-xs text-neutral-500">No evidence attached.</p>
            ) : (
              evidence.map((ev) => (
                <div
                  key={ev.id}
                  className="p-2 rounded-md border border-[#1f222b] bg-[#151821] text-xs flex items-center justify-between"
                >
                  <div className="flex flex-col min-w-0">
                    <span className="font-medium text-white truncate">{ev.name}</span>
                    <span className="text-[10px] text-neutral-500 font-mono">{ev.type}</span>
                  </div>
                  <span className="text-[10px] px-1.5 py-0.5 rounded text-[#2ecc71] bg-[#2ecc71]/10">
                    {ev.processingState}
                  </span>
                </div>
              ))
            )}
          </div>
        )}

        {section === "versions" && (
          <div className="p-2 space-y-1.5">
            {versions.length === 0 ? (
              <p className="p-4 text-center text-xs text-neutral-500">No versions committed yet.</p>
            ) : (
              versions.map((v) => (
                <div
                  key={v.id}
                  className={`p-2.5 rounded-md border text-xs space-y-1 ${
                    v.isCurrent ? "bg-[#151821] border-[#00e5ff]" : "bg-[#151821] border-[#1f222b]"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white font-mono">{v.label}</span>
                    {v.isCurrent && (
                      <span className="text-[10px] font-mono px-1 rounded text-[#00e5ff] bg-[#00e5ff]/10">
                        ACTIVE
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-neutral-400">{v.changeSummary || "Compiled representation"}</p>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </nav>
  );
}

function NavTabButton({
  active,
  onClick,
  icon: Icon,
  label,
  count,
}: {
  active: boolean;
  onClick: () => void;
  icon: typeof Box;
  label: string;
  count?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1 px-2 py-1 rounded transition-colors whitespace-nowrap ${
        active
          ? "text-[#00e5ff] font-medium bg-[rgba(0,229,255,0.12)]"
          : "text-neutral-400 hover:text-white"
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      <span>{label}</span>
      {count != null && count > 0 && (
        <span className="text-[10px] font-mono text-neutral-500">({count})</span>
      )}
    </button>
  );
}
