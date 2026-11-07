"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  ChevronDown,
  Command,
  Download,
  FolderOpen,
  Hammer,
  ImagePlus,
  Layers,
  PanelBottom,
  PanelLeft,
  PanelRight,
  Search,
  Sparkles,
  Workflow,
} from "lucide-react";
import { cn } from "@/lib/utils";

export interface ActionsMenuHandlers {
  onAddEvidence: () => void;
  onBuild: () => void;
  onGenerated: () => void;
  onCompleteInterior: () => void;
  onConstruction: () => void;
  onQuery: () => void;
  onExport: () => void;
  onBrowse: () => void;
  onCommands: () => void;
  onToggleScene: () => void;
  onToggleInspector: () => void;
  onToggleInsights: () => void;
  onToggleTimeline: () => void;
}

interface Item {
  id: string;
  label: string;
  hint?: string;
  icon: React.ComponentType<{ className?: string }>;
  run: () => void;
  testId?: string;
}

/** The single "Actions" menu that replaces the old toolbar. Every action stays reachable from here. */
export default function ActionsMenu(h: ActionsMenuHandlers) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const sections: { title: string; items: Item[] }[] = [
    {
      title: "Create",
      items: [
        { id: "add-evidence", label: "Add evidence", hint: "photographs", icon: ImagePlus, run: h.onAddEvidence },
        { id: "build", label: "Build a room, corridor or city", icon: Hammer, run: h.onBuild, testId: "open-build" },
        { id: "complete", label: "Complete interior", hint: "generated", icon: Sparkles, run: h.onCompleteInterior, testId: "open-complete" },
      ],
    },
    {
      title: "Edit",
      items: [
        { id: "generated", label: "Generated objects", icon: Layers, run: h.onGenerated, testId: "open-generated" },
        { id: "construction", label: "Room construction pipeline", icon: Workflow, run: h.onConstruction },
      ],
    },
    {
      title: "Explore",
      items: [
        { id: "query", label: "Spatial query", icon: Search, run: h.onQuery },
        { id: "export", label: "Export", icon: Download, run: h.onExport, testId: "open-export" },
        { id: "browse", label: "Worlds, sessions, evidence", icon: FolderOpen, run: h.onBrowse },
        { id: "commands", label: "Command palette", hint: "Ctrl K", icon: Command, run: h.onCommands },
      ],
    },
    {
      title: "Panels",
      items: [
        { id: "scene", label: "Scene panel", hint: "[", icon: PanelLeft, run: h.onToggleScene },
        { id: "inspector", label: "Inspector", hint: "]", icon: PanelRight, run: h.onToggleInspector },
        { id: "insights", label: "Reconstruction insights", hint: "\\", icon: PanelBottom, run: h.onToggleInsights },
        { id: "timeline", label: "Timeline and telemetry", icon: Activity, run: h.onToggleTimeline },
      ],
    },
  ];

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        data-testid="open-actions"
        className={cn(
          "flex h-7 items-center gap-1 rounded border px-2.5 text-xs font-medium cursor-pointer",
          open
            ? "border-[var(--accent-border)] bg-[var(--accent-subtle)] text-[var(--accent)]"
            : "border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]",
        )}
      >
        Actions
        <ChevronDown className="h-3.5 w-3.5" />
      </button>
      {open && (
        <div role="menu" aria-label="Actions" className="absolute right-0 top-8 z-50 w-64 rounded border border-[var(--border)] bg-[var(--bg-surface)] py-1 shadow-lg">
          {sections.map((s, i) => (
            <div key={s.title} role="none" className={cn(i > 0 && "mt-1 border-t border-[var(--border-subtle)] pt-1")}>
              <div className="px-3 py-1 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-tertiary)]">{s.title}</div>
              {s.items.map(({ id, label, hint, icon: Icon, run, testId }) => (
                <button
                  key={id}
                  type="button"
                  role="menuitem"
                  data-testid={testId}
                  onClick={() => {
                    setOpen(false);
                    run();
                  }}
                  className="flex h-8 w-full items-center gap-2 px-3 text-left text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] cursor-pointer"
                >
                  <Icon className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
                  <span className="flex-1 truncate">{label}</span>
                  {hint && <span className="font-mono text-[10px] text-[var(--text-tertiary)]">{hint}</span>}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
