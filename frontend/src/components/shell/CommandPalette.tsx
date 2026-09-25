"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { useRouter } from "next/navigation";
import { useWorlds } from "@/lib/api";
import {
  Search,
  Camera,
  Globe,
  ShieldCheck,
  Workflow,
  FileText,
  Zap,
  Box,
  Eye,
  Maximize2,
  Sparkles,
  MapPin,
  Map as MapIcon,
  Compass,
  Download,
  Filter,
  GitBranch,
  Layers,
  Sliders,
  PanelLeft,
  PanelRight,
} from "lucide-react";
import { searchAll, type SearchResult, type SearchResultType } from "@/lib/search";

const TYPE_LABEL: Record<SearchResultType, string> = {
  SESSION: "Session",
  WORLD: "World",
  EVIDENCE: "Evidence",
  ANALYSIS: "Analysis",
  RESULT: "Result",
  REPORT: "Report",
};

interface Command {
  label: string;
  href?: string;
  icon: any;
  action?: () => void;
  category?: string;
}

const COMMANDS: Command[] = [
  // Viewport & Framing
  {
    label: "Frame Selection / Reset View [F]",
    icon: Maximize2,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "f" })),
  },
  {
    label: "Toggle 3D Viewport / Geospatial Map [M]",
    icon: MapIcon,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "m" })),
  },
  {
    label: "Toggle Navigation Panel [ [ ]",
    icon: PanelLeft,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "[" })),
  },
  {
    label: "Toggle Adaptive Inspector [ ] ]",
    icon: PanelRight,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "]" })),
  },
  // Layers
  {
    label: "Toggle Points Cloud Layer [1]",
    icon: Eye,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "1" })),
  },
  {
    label: "Toggle Structural Entities Layer [2]",
    icon: Box,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "2" })),
  },
  {
    label: "Toggle Camera Frustums [3]",
    icon: Camera,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "3" })),
  },
  {
    label: "Toggle Depth Noise Filter [O]",
    icon: Layers,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "o" })),
  },
  {
    label: "Toggle Uncertainty Heatmap [U]",
    icon: Sparkles,
    action: () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "u" })),
  },
  // Spatial Queries
  {
    label: "Spatial Query: Filter Floor Planes",
    icon: Filter,
    action: () => window.dispatchEvent(new CustomEvent("apply-spatial-query", { detail: { type: "floor" } })),
  },
  {
    label: "Spatial Query: Filter Vertical Walls",
    icon: Filter,
    action: () => window.dispatchEvent(new CustomEvent("apply-spatial-query", { detail: { type: "wall" } })),
  },
  {
    label: "Spatial Query: Filter Objects & Furniture",
    icon: Filter,
    action: () => window.dispatchEvent(new CustomEvent("apply-spatial-query", { detail: { type: "object" } })),
  },
  {
    label: "Spatial Query: High Confidence (≥ 80%)",
    icon: Filter,
    action: () => window.dispatchEvent(new CustomEvent("apply-spatial-query", { detail: { minConfidence: 0.8 } })),
  },
  {
    label: "Spatial Query: Flagged for Review (< 50%)",
    icon: Filter,
    action: () => window.dispatchEvent(new CustomEvent("apply-spatial-query", { detail: { type: "all", minConfidence: 0 } })),
  },
  {
    label: "Reset Active Spatial Query",
    icon: Sliders,
    action: () => window.dispatchEvent(new CustomEvent("reset-spatial-query")),
  },
  // Workflows & Versioning
  {
    label: "Compare WorldStore Versions (Diff)",
    icon: GitBranch,
    action: () => window.dispatchEvent(new CustomEvent("open-version-diff")),
  },
  {
    label: "Open Room Construction Pipeline",
    icon: Workflow,
    action: () => window.dispatchEvent(new CustomEvent("open-room-construction")),
  },
  {
    label: "Open Spatial Query",
    icon: Search,
    action: () => window.dispatchEvent(new CustomEvent("open-spatial-query")),
  },
  // Exports
  {
    label: "Export Canonical WorldIR (JSON)",
    icon: Download,
    action: () => window.dispatchEvent(new CustomEvent("trigger-export", { detail: { format: "worldir" } })),
  },
  {
    label: "Export Points Cloud (PLY)",
    icon: Download,
    action: () => window.dispatchEvent(new CustomEvent("trigger-export", { detail: { format: "ply" } })),
  },
  {
    label: "Export Camera Trajectories (JSON)",
    icon: Download,
    action: () => window.dispatchEvent(new CustomEvent("trigger-export", { detail: { format: "cameras" } })),
  },
  {
    label: "Export Pipeline Report (JSON)",
    icon: Download,
    action: () => window.dispatchEvent(new CustomEvent("trigger-export", { detail: { format: "report" } })),
  },
  // Worlds & Navigation
  { label: "Open Spatial Studio", href: "/", icon: Globe },
  { label: "Create Capture Session", href: "/sessions/new", icon: Camera },
  { label: "Create Spatial World", href: "/worlds/new", icon: Globe },
  { label: "Attach Evidence", href: "/evidence", icon: ShieldCheck },
  { label: "Open Reports", href: "/reports", icon: FileText },
];


export default function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const [searchState, setSearchState] = useState<{ q: string; rows: SearchResult[] }>({ q: "", rows: [] });
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<Element | null>(null);
  const isOpenRef = useRef(open);

  // Capture the trigger element on open, restore focus to it on close
  // (Escape, backdrop click, item selection, or the parent flipping
  // `open` to false for any other reason all funnel through this).
  useEffect(() => {
    isOpenRef.current = open;
    if (open) {
      previousFocusRef.current = document.activeElement;
      // Don't setQuery here - let the parent control query via key if needed
      requestAnimationFrame(() => inputRef.current?.focus());
    } else {
      const el = previousFocusRef.current;
      if (el instanceof HTMLElement && document.contains(el)) {
        el.focus();
      }
      previousFocusRef.current = null;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- cleanup on close
      setQuery("");
      setHighlightedIndex(0);
    }
  }, [open]);

  // Reset query + keyboard highlight when the palette opens — the react.dev
  // "adjust state during render" pattern (guarded, so nothing stateful runs
  // inside an effect body).
  const [prevOpen, setPrevOpen] = useState(open);
  if (open !== prevOpen) {
    setPrevOpen(open);
    if (open) {
      setQuery("");
      setHighlightedIndex(0);
    }
  }

  // Reset the keyboard highlight whenever the query changes.
  const [prevQuery, setPrevQuery] = useState(query);
  if (query !== prevQuery) {
    setPrevQuery(query);
    setHighlightedIndex(0);
  }

  // `searchAll` fetches real collections over the API, so it resolves into
  // state — only promise callbacks touch state, never the effect body. Rows
  // are tagged with the query they answer; a superseded query's late response
  // is discarded via `cancelled`.
  useEffect(() => {
    if (query.trim() === "") return;
    let cancelled = false;
    searchAll(query)
      .then((rows) => {
        if (!cancelled) setSearchState({ q: query, rows });
      })
      .catch(() => {
        if (!cancelled) setSearchState({ q: query, rows: [] });
      });
    return () => {
      cancelled = true;
    };
  }, [query]);

  const { data: rawWorlds } = useWorlds();
  const worlds = rawWorlds ?? [];

  // Dynamically include registered worlds in default command list
  const dynamicCommands = useMemo(() => {
    const list = [...COMMANDS];
    for (const w of worlds) {
      list.push({
        label: `Open World: ${w.name || w.id}`,
        href: `/worlds/${w.id}`,
        icon: Globe,
      });
    }
    return list;
  }, [worlds]);

  // Only rows that answer the current query are shown; anything stale (or
  // from a previous palette session) reads as an empty result set.
  const results = searchState.q === query ? searchState.rows : [];
  const flatList = query.trim() === "" ? dynamicCommands : results;

  const go = useCallback(
    (target: string | Command) => {
      onClose();
      if (typeof target === "string") {
        router.push(target);
      } else if (target?.action) {
        target.action();
      } else if (target?.href) {
        router.push(target.href);
      }
    },
    [onClose, router]
  );

  useEffect(() => {
    if (!open) return;
    const handler = (e: globalThis.KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "ArrowDown") {
        if (flatList.length === 0) return;
        e.preventDefault();
        setHighlightedIndex((i) => (i + 1) % flatList.length);
        return;
      }
      if (e.key === "ArrowUp") {
        if (flatList.length === 0) return;
        e.preventDefault();
        setHighlightedIndex((i) => (i - 1 + flatList.length) % flatList.length);
        return;
      }
      if (e.key === "Enter") {
        const item = flatList[highlightedIndex];
        if (item) {
          e.preventDefault();
          go(item as any);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, flatList, highlightedIndex, go]);


  const handleDialogKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "Tab") return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusable = Array.from(dialog.querySelectorAll<HTMLElement>("input, button, [href]"));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    const atBoundary = e.shiftKey
      ? active === first || !dialog.contains(active)
      : active === last || !dialog.contains(active);
    if (atBoundary) {
      e.preventDefault();
      (e.shiftKey ? last : first).focus();
    }
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
      style={{ background: "rgba(6,7,9,0.6)" }}
      onClick={onClose}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Search and commands"
        className="w-full max-w-lg rounded-lg overflow-hidden shadow-2xl"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleDialogKeyDown}
      >
        <div className="flex items-center gap-2 px-3 h-12 border-b" style={{ borderColor: "var(--border)" }}>
          <Search className="w-4 h-4 shrink-0" style={{ color: "var(--text-tertiary)" }} />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search Reality Engine…"
            aria-label="Search Reality Engine"
            className="flex-1 bg-transparent outline-none text-sm"
            style={{ color: "var(--text-primary)" }}
          />
          <kbd
            className="text-xs font-mono-num px-1.5 py-0.5 rounded shrink-0"
            style={{ background: "var(--bg-base)", border: "1px solid var(--border)", color: "var(--text-tertiary)" }}
          >
            Esc
          </kbd>
        </div>

        <div className="max-h-96 overflow-y-auto py-1.5">
          {query.trim() === "" ? (
            <>
              <div className="px-3 pt-1 pb-1.5 text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-tertiary)" }}>
                Commands
              </div>
              {COMMANDS.map((c, index) => (
                <button
                  key={c.label}
                  type="button"
                  onClick={() => go(c)}
                  className="w-full flex items-center gap-2.5 px-3 h-9 text-sm text-left transition-colors"
                  style={{
                    color: "var(--text-primary)",
                    background: index === highlightedIndex ? "var(--bg-hover)" : "transparent",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <c.icon className="w-4 h-4" style={{ color: "var(--text-tertiary)" }} />
                  {c.label}
                </button>
              ))}
            </>
          ) : results.length === 0 ? (
            <p className="px-3 py-6 text-sm text-center" style={{ color: "var(--text-tertiary)" }}>
              No results for &quot;{query}&quot;.
            </p>
          ) : (
            results.map((r, index) => (
              <button
                key={`${r.type}-${r.id}`}
                type="button"
                onClick={() => go(r.href)}
                className="w-full flex flex-col items-start gap-0.5 px-3 py-2 text-left transition-colors"
                style={{ background: index === highlightedIndex ? "var(--bg-hover)" : "transparent" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--accent)" }}>
                  {TYPE_LABEL[r.type]}
                </span>
                <span className="text-sm" style={{ color: "var(--text-primary)" }}>
                  {r.title}
                </span>
                {r.subtitle && (
                  <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
                    {r.subtitle}
                  </span>
                )}
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
