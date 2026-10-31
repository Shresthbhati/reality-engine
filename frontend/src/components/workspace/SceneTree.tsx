"use client";

import React, { useMemo, useState } from "react";
import {
  Armchair,
  BrickWall,
  Building2,
  ChevronDown,
  ChevronRight,
  DoorOpen,
  Eye,
  EyeOff,
  Layers,
  Lightbulb,
  PanelTop,
  Search,
  Shapes,
  Square,
  SquareDashed,
  X,
  Box,
  FolderTree,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  ancestorKeys,
  defaultExpandedKeys,
  filterTree,
  visibilityOf,
  type SceneNode,
  type SceneNodeKind,
} from "@/lib/workspace/scene-tree-model";

const ICONS: Record<SceneNodeKind, React.ComponentType<{ className?: string }>> = {
  building: Building2,
  level: Layers,
  room: Box,
  floor: Square,
  ceiling: SquareDashed,
  wall: BrickWall,
  window: PanelTop,
  door: DoorOpen,
  furniture: Armchair,
  lighting: Lightbulb,
  other: Shapes,
  group: FolderTree,
  unassigned: FolderTree,
};

/** Group rows take the icon of what they contain. */
function iconFor(node: SceneNode): React.ComponentType<{ className?: string }> {
  if (node.kind === "group") return ICONS[(node.children[0]?.kind as SceneNodeKind) ?? "group"];
  return ICONS[node.kind];
}

interface SceneTreeProps {
  tree: SceneNode[];
  selectedEntityId: string | null;
  hiddenIds: ReadonlySet<string>;
  onSelect: (entityId: string) => void;
  onToggleVisibility: (leafIds: string[]) => void;
  /** Opens the worlds / sessions / evidence / versions browser. */
  onBrowse?: () => void;
  /** Present when the tree is shown as a drawer. */
  onClose?: () => void;
  loading?: boolean;
}

export default function SceneTree({ tree, selectedEntityId, hiddenIds, onSelect, onToggleVisibility, onBrowse, onClose, loading }: SceneTreeProps) {
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});

  const defaults = useMemo(() => defaultExpandedKeys(tree), [tree]);
  const selectedAncestors = useMemo(() => new Set(ancestorKeys(tree, selectedEntityId)), [tree, selectedEntityId]);
  const searching = query.trim().length > 0;
  const visible = useMemo(() => filterTree(tree, query), [tree, query]);

  const isOpen = (key: string) => searching || (overrides[key] ?? (defaults.has(key) || selectedAncestors.has(key)));
  const toggleOpen = (key: string) => setOverrides((o) => ({ ...o, [key]: !isOpen(key) }));

  const renderNode = (node: SceneNode, depth: number): React.ReactNode => {
    const expandable = node.children.length > 0;
    const open = expandable && isOpen(node.key);
    const vis = visibilityOf(node.leafIds, hiddenIds);
    const hidden = vis === "none";
    const selected = node.entityId !== null && node.entityId === selectedEntityId;
    const Icon = iconFor(node);
    const eyeLabel = `${hidden ? "Show" : "Hide"} ${node.label}`;

    return (
      <li key={node.key} role="none">
        <div
          role="treeitem"
          aria-level={depth + 1}
          aria-expanded={expandable ? open : undefined}
          aria-selected={selected}
          data-testid={node.entityId ? `scene-row-${node.entityId}` : `scene-group-${node.key}`}
          className={cn(
            "group flex h-7 items-center gap-1 border-l-2 pr-1 text-xs transition-colors",
            selected
              ? "border-[var(--accent)] bg-[var(--accent-subtle)] text-[var(--text-primary)]"
              : "border-transparent text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]",
            hidden && "opacity-60",
          )}
          style={{ paddingLeft: 4 + depth * 14 }}
        >
          {expandable ? (
            <button
              type="button"
              onClick={() => toggleOpen(node.key)}
              aria-label={`${open ? "Collapse" : "Expand"} ${node.label}`}
              className="flex h-5 w-5 shrink-0 items-center justify-center rounded text-[var(--text-tertiary)] hover:text-[var(--text-primary)] cursor-pointer"
            >
              {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
            </button>
          ) : (
            <span className="h-5 w-5 shrink-0" aria-hidden />
          )}
          <button
            type="button"
            onClick={() => (node.entityId ? onSelect(node.entityId) : toggleOpen(node.key))}
            onDoubleClick={() => expandable && toggleOpen(node.key)}
            className="flex min-w-0 flex-1 items-center gap-1.5 text-left cursor-pointer"
          >
            <Icon className={cn("h-3.5 w-3.5 shrink-0", selected ? "text-[var(--accent)]" : "text-[var(--text-tertiary)]")} />
            <span className="max-w-[75%] shrink-0 truncate">{node.label}</span>
            {node.hint && <span className="truncate font-mono text-[10px] text-[var(--text-tertiary)]">{node.hint}</span>}
          </button>
          <button
            type="button"
            onClick={() => onToggleVisibility(node.leafIds)}
            aria-label={eyeLabel}
            aria-pressed={hidden}
            title={eyeLabel}
            data-testid={`scene-eye-${node.entityId ?? node.key}`}
            className={cn(
              "flex h-5 w-5 shrink-0 items-center justify-center rounded cursor-pointer",
              hidden ? "text-[var(--text-tertiary)]" : vis === "some" ? "text-[var(--text-secondary)]" : "text-[var(--text-tertiary)] hover:text-[var(--text-primary)]",
            )}
          >
            {hidden ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className={cn("h-3.5 w-3.5", vis === "some" && "opacity-60")} />}
          </button>
        </div>
        {open && (
          <ul role="group" className="m-0 list-none p-0">
            {node.children.map((c) => renderNode(c, depth + 1))}
          </ul>
        )}
      </li>
    );
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-[var(--bg-sidebar)]" data-testid="scene-tree">
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-[var(--border-subtle)] px-3">
        <h2 className="text-xs font-semibold tracking-wide text-[var(--text-primary)]">Scene</h2>
        <div className="flex items-center gap-0.5">
          <button
            type="button"
            onClick={() => {
              setSearchOpen((v) => !v);
              if (searchOpen) setQuery("");
            }}
            aria-label="Search scene objects"
            aria-pressed={searchOpen}
            className={cn("flex h-6 w-6 items-center justify-center rounded cursor-pointer", searchOpen ? "text-[var(--accent)]" : "text-[var(--text-tertiary)] hover:text-[var(--text-primary)]")}
          >
            <Search className="h-3.5 w-3.5" />
          </button>
          {onClose && (
            <button type="button" onClick={onClose} aria-label="Close scene panel" className="flex h-6 w-6 items-center justify-center rounded text-[var(--text-tertiary)] hover:text-[var(--text-primary)] cursor-pointer">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="shrink-0 px-2 py-2">
        <label className="flex h-7 items-center gap-1.5 rounded border border-[var(--border)] bg-[var(--bg-base)] px-2 text-[var(--text-tertiary)] focus-within:border-[var(--accent-border)]">
          <Search className="h-3 w-3 shrink-0" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search objects..."
            aria-label="Search objects"
            autoFocus={searchOpen}
            className="min-w-0 flex-1 bg-transparent text-xs text-[var(--text-primary)] outline-none placeholder:text-[var(--text-tertiary)]"
          />
          {query && (
            <button type="button" onClick={() => setQuery("")} aria-label="Clear search" className="cursor-pointer text-[var(--text-tertiary)] hover:text-[var(--text-primary)]">
              <X className="h-3 w-3" />
            </button>
          )}
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto pb-2">
        {visible.length > 0 ? (
          <ul role="tree" aria-label="Scene" className="m-0 list-none p-0">
            {visible.map((n) => renderNode(n, 0))}
          </ul>
        ) : (
          <p className="px-3 py-4 text-xs text-[var(--text-tertiary)]" data-testid="scene-empty">
            {loading ? "Loading scene..." : searching ? `No objects match "${query.trim()}".` : "This model has no objects yet."}
          </p>
        )}
      </div>

      {onBrowse && (
        <div className="shrink-0 border-t border-[var(--border-subtle)] p-2">
          <button
            type="button"
            onClick={onBrowse}
            data-testid="open-browse"
            className="flex h-7 w-full items-center justify-center rounded border border-[var(--border)] text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] cursor-pointer"
          >
            Worlds, sessions, evidence...
          </button>
        </div>
      )}
    </div>
  );
}
