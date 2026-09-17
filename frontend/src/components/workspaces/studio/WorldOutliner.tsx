'use client';

/**
 * WorldOutliner — Simplified categorical scene entity tree.
 * - Starts in a clean, compact hierarchy (Structures >, Objects >, Rooms >, Evidence >)
 * - Selecting a category expands it progressively
 * - Type icons, confidence indicators, hover visibility/lock toggles
 * - Panel collapse toggle for maximizing 3D viewport
 */

import React, { useState, useMemo, useEffect, useRef } from 'react';
import { useREStore } from '@/store/re-store';
import type { Entity, EntityId, EntityType } from '@/types/reality-engine';
import {
  ChevronRight,
  ChevronDown,
  Search,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  X,
  PanelLeftClose,
  Folder,
  FolderOpen,
  Layers,
  Focus,
  Eye as EyeIcon,
  ShieldCheck,
  Ruler,
  ListTree,
  Boxes,
} from 'lucide-react';

// ── Entity type icons ──────────────────────────────────────────────────────────

const TYPE_ICONS: Record<EntityType, string> = {
  WORLD: '🌍',
  SITE: '📍',
  BUILDING: '🏗',
  FACADE: '🏛',
  COMPONENT: '🏛',
  COLUMN: '🏛',
  CAPITAL: '⚜️',
  ORNAMENT: '🌿',
  RELIEF: '🪨',
  OBJECT: '📦',
  FLOOR: '🏠',
  ROOM: '📐',
  WALL: '🧱',
  TERRAIN: '🏔',
  ROAD: '🛣',
  VEGETATION: '🌿',
  INFRASTRUCTURE: '⚙️',
  CAMERA: '📷',
  POINT_CLOUD: '☁️',
  MESH: '📦',
  SPLAT: '✨',
  TRAJECTORY: '〰️',
  OBSERVATION: '👁',
  FEATURE: '⬡',
  MATCH: '🔗',
  GENERIC: '◻️',
};

// ── Confidence dot ─────────────────────────────────────────────────────────────

function ConfidenceDot({ confidence }: { confidence: number }) {
  const color =
    confidence >= 0.85 ? '#22c55e' : confidence >= 0.65 ? '#eab308' : '#ef4444';
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
      style={{ background: color }}
      title={`Confidence: ${(confidence * 100).toFixed(0)}%`}
    />
  );
}

// ── Category Group Definition ──────────────────────────────────────────────────

type OutlinerCategory = 'STRUCTURES' | 'OBJECTS' | 'ROOMS' | 'EVIDENCE';

const CATEGORY_MAP: Record<EntityType, OutlinerCategory> = {
  WORLD: 'STRUCTURES',
  SITE: 'STRUCTURES',
  BUILDING: 'STRUCTURES',
  FACADE: 'STRUCTURES',
  COMPONENT: 'OBJECTS',
  COLUMN: 'STRUCTURES',
  CAPITAL: 'OBJECTS',
  ORNAMENT: 'OBJECTS',
  RELIEF: 'OBJECTS',
  OBJECT: 'OBJECTS',
  FLOOR: 'STRUCTURES',
  WALL: 'STRUCTURES',
  TERRAIN: 'STRUCTURES',
  ROAD: 'STRUCTURES',
  INFRASTRUCTURE: 'STRUCTURES',
  ROOM: 'ROOMS',
  MESH: 'OBJECTS',
  GENERIC: 'OBJECTS',
  VEGETATION: 'OBJECTS',
  CAMERA: 'EVIDENCE',
  POINT_CLOUD: 'EVIDENCE',
  SPLAT: 'EVIDENCE',
  TRAJECTORY: 'EVIDENCE',
  OBSERVATION: 'EVIDENCE',
  FEATURE: 'EVIDENCE',
  MATCH: 'EVIDENCE',
};

interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  entityId: EntityId;
}

export function WorldOutliner() {
  const {
    entities,
    selection,
    selectEntity,
    hoverEntity,
    toggleEntityVisibility,
    toggleEntityLock,
    isolateEntity,
    showAllEntities,
    toggleOutliner,
    setActiveWorkspace,
    setActiveMeasurementTool,
  } = useREStore();

  const [viewMode, setViewMode] = useState<'TREE' | 'GROUPS'>('TREE');
  const [filterText, setFilterText] = useState('');
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({
    'ent-world': true,
    'ent-site': true,
    'ent-structure-main': true,
  });
  
  // Categories start COLLAPSED by default (as required by master prompt!)
  const [expandedCategories, setExpandedCategories] = useState<Record<OutlinerCategory, boolean>>({
    STRUCTURES: true, // only structures initial preview
    OBJECTS: false,
    ROOMS: false,
    EVIDENCE: false,
  });

  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const contextMenuRef = useRef<HTMLDivElement>(null);

  // Close context menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenu(null);
      }
    };
    window.addEventListener('click', handleClickOutside);
    return () => window.removeEventListener('click', handleClickOutside);
  }, []);

  const selectedIds = useMemo(
    () => new Set(selection.selectedEntityIds),
    [selection.selectedEntityIds]
  );

  const toggleNodeExpand = (id: string) => {
    setExpandedNodes((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleCategory = (cat: OutlinerCategory) => {
    setExpandedCategories((prev) => ({
      ...prev,
      [cat]: !prev[cat],
    }));
  };

  // Build tree roots
  const rootEntities = useMemo(() => {
    const roots: Entity[] = [];
    entities.forEach((entity) => {
      if (!entity.parentId || !entities.has(entity.parentId)) {
        roots.push(entity);
      }
    });
    return roots;
  }, [entities]);

  // Group entities by category
  const groupedEntities = useMemo(() => {
    const groups: Record<OutlinerCategory, Entity[]> = {
      STRUCTURES: [],
      OBJECTS: [],
      ROOMS: [],
      EVIDENCE: [],
    };

    const q = filterText.toLowerCase().trim();

    entities.forEach((entity) => {
      if (q && !entity.name.toLowerCase().includes(q) && !entity.type.toLowerCase().includes(q)) {
        return;
      }
      const cat = CATEGORY_MAP[entity.type] || 'OBJECTS';
      groups[cat].push(entity);
    });

    return groups;
  }, [entities, filterText]);

  const handleContextMenu = (e: React.MouseEvent, entityId: EntityId) => {
    e.preventDefault();
    e.stopPropagation();
    selectEntity(entityId);
    setContextMenu({
      visible: true,
      x: Math.min(e.clientX, window.innerWidth - 180),
      y: Math.min(e.clientY, window.innerHeight - 220),
      entityId,
    });
  };

  const renderTreeRow = (entity: Entity, depth: number = 0): React.ReactNode => {
    const q = filterText.toLowerCase().trim();
    const hasChildren = entity.childIds && entity.childIds.length > 0;
    const isExpanded = expandedNodes[entity.id] || Boolean(q);
    const isSelected = selectedIds.has(entity.id);
    const isHidden = entity.visibility === 'HIDDEN';
    const isLocked = entity.lock === 'LOCKED';

    // If search active, only show if match or children match
    if (q && !entity.name.toLowerCase().includes(q) && !entity.type.toLowerCase().includes(q)) {
      const hasMatchingChild = entity.childIds?.some((cId) => {
        const c = entities.get(cId);
        return c && (c.name.toLowerCase().includes(q) || c.type.toLowerCase().includes(q));
      });
      if (!hasMatchingChild) return null;
    }

    return (
      <React.Fragment key={entity.id}>
        <div
          onClick={() => selectEntity(entity.id)}
          onContextMenu={(e) => handleContextMenu(e, entity.id)}
          onMouseEnter={() => hoverEntity(entity.id)}
          onMouseLeave={() => hoverEntity(undefined)}
          style={{ paddingLeft: `${8 + depth * 14}px` }}
          className={`group flex items-center justify-between pr-2 py-1 cursor-pointer text-xs font-mono transition-colors select-none ${
            isSelected
              ? 'bg-[#3d8ef7]/15 text-[#6eb0ff] border-l-2 border-[#3d8ef7]'
              : 'hover:bg-[#141720] text-[#c4c7d4]'
          }`}
        >
          <div className="flex items-center gap-1.5 min-w-0 flex-1">
            {hasChildren ? (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleNodeExpand(entity.id);
                }}
                className="p-0.5 text-[#54596b] hover:text-[#9296a6]"
              >
                {isExpanded ? (
                  <ChevronDown className="w-3 h-3 text-[#3d8ef7]" />
                ) : (
                  <ChevronRight className="w-3 h-3" />
                )}
              </button>
            ) : (
              <span className="w-4" />
            )}

            <span className="text-xs shrink-0">{TYPE_ICONS[entity.type] || '◻️'}</span>
            <span className={`truncate text-[11px] ${isHidden ? 'opacity-40 line-through' : ''}`}>
              {entity.name}
            </span>
          </div>

          <div className="flex items-center gap-1.5 shrink-0 ml-1">
            <ConfidenceDot confidence={entity.provenance.confidence} />

            <div className="flex items-center gap-0.5">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleEntityVisibility(entity.id);
                }}
                title={isHidden ? 'Show Entity' : 'Hide Entity'}
                className="p-0.5 text-[#9296a6] hover:text-[#ededf2] transition-colors"
              >
                {isHidden ? (
                  <EyeOff className="w-3 h-3 text-[#e74c3c]" />
                ) : (
                  <Eye className="w-3 h-3 text-[#54596b] group-hover:text-[#9296a6]" />
                )}
              </button>

              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  toggleEntityLock(entity.id);
                }}
                title={isLocked ? 'Unlock Entity' : 'Lock Entity'}
                className="p-0.5 transition-colors"
              >
                {isLocked ? (
                  <Lock className="w-3 h-3 text-amber-400" />
                ) : (
                  <Unlock className="w-3 h-3 text-[#303545] group-hover:text-[#54596b]" />
                )}
              </button>
            </div>
          </div>
        </div>

        {hasChildren && isExpanded && (
          <div>
            {entity.childIds!.map((childId) => {
              const child = entities.get(childId);
              return child ? renderTreeRow(child, depth + 1) : null;
            })}
          </div>
        )}
      </React.Fragment>
    );
  };

  const categories: Array<{ id: OutlinerCategory; label: string; icon: string }> = [
    { id: 'STRUCTURES', label: 'Structures & Sites', icon: '🏗' },
    { id: 'OBJECTS', label: 'Architectural Objects', icon: '📦' },
    { id: 'ROOMS', label: 'Interiors & Rooms', icon: '📐' },
    { id: 'EVIDENCE', label: 'Evidence & Sensors', icon: '📷' },
  ];

  return (
    <div className="flex flex-col h-full bg-[#101217] text-[#ededf2] select-none border-r border-[#1f222b] relative">
      {/* ── Top Header with Mode Switcher & Collapse Button ────────── */}
      <div className="flex items-center justify-between px-2.5 h-8 border-b border-[#1f222b] bg-[#0c0d11]">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-mono font-bold tracking-widest text-[#54596b] uppercase">
            Outliner
          </span>
          <span className="text-[9px] font-mono px-1 py-0.2 rounded bg-[#1f222b] text-[#9296a6] num-tabular">
            {entities.size}
          </span>
        </div>

        <div className="flex items-center gap-1">
          {/* Mode Switcher */}
          <div className="flex items-center bg-[#151720] rounded p-0.5 border border-[#1f222b]">
            <button
              type="button"
              onClick={() => setViewMode('TREE')}
              title="Spatial Tree Hierarchy (World -> Site -> Building -> Facade -> Detail)"
              className={`p-1 rounded text-[10px] transition-colors ${
                viewMode === 'TREE'
                  ? 'bg-[#3d8ef7] text-black font-bold'
                  : 'text-[#9296a6] hover:text-white'
              }`}
            >
              <ListTree className="w-3 h-3" />
            </button>
            <button
              type="button"
              onClick={() => setViewMode('GROUPS')}
              title="Categorical Groups"
              className={`p-1 rounded text-[10px] transition-colors ${
                viewMode === 'GROUPS'
                  ? 'bg-[#3d8ef7] text-black font-bold'
                  : 'text-[#9296a6] hover:text-white'
              }`}
            >
              <Boxes className="w-3 h-3" />
            </button>
          </div>

          <button
            type="button"
            onClick={toggleOutliner}
            title="Collapse Outliner (⌘B)"
            className="p-1 rounded text-[#9296a6] hover:text-[#ededf2] hover:bg-[#1a1d26] transition-colors"
          >
            <PanelLeftClose className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Filter Bar ────────────────────────────────────────────── */}
      <div className="relative flex items-center px-2 py-1.5 border-b border-[#1f222b] bg-[#0f1014]">
        <Search className="absolute left-3.5 w-3 h-3 text-[#54596b] pointer-events-none" />
        <input
          type="text"
          placeholder="Filter scene nodes..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          className="w-full bg-[#15171e] border border-[#1f222b] rounded text-[11px] text-[#ededf2] placeholder-[#54596b] pl-6 pr-6 py-1 outline-none focus:border-[#3d8ef7]/60 font-mono transition-colors"
        />
        {filterText && (
          <button
            type="button"
            onClick={() => setFilterText('')}
            className="absolute right-3.5 text-[#54596b] hover:text-[#9296a6]"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      {/* ── Entity Hierarchy or Categorical View ────────────────────── */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-1 space-y-1">
        {viewMode === 'TREE' ? (
          <div className="py-1">
            {rootEntities.map((root) => renderTreeRow(root, 0))}
          </div>
        ) : (
          categories.map((cat) => {
            const items = groupedEntities[cat.id];
            const isExpanded = expandedCategories[cat.id] || Boolean(filterText);

            return (
              <div key={cat.id} className="rounded border border-[#1f222b]/50 overflow-hidden">
                {/* Category Bar */}
                <button
                  type="button"
                  onClick={() => toggleCategory(cat.id)}
                  className="w-full flex items-center justify-between px-2 py-1.5 bg-[#14161f] hover:bg-[#181b26] transition-colors text-left font-mono"
                >
                  <div className="flex items-center gap-1.5">
                    {isExpanded ? (
                      <ChevronDown className="w-3.5 h-3.5 text-[#3d8ef7]" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 text-[#54596b]" />
                    )}
                    {isExpanded ? (
                      <FolderOpen className="w-3.5 h-3.5 text-[#3d8ef7]" />
                    ) : (
                      <Folder className="w-3.5 h-3.5 text-[#9296a6]" />
                    )}
                    <span className="text-[11px] font-semibold text-[#f0f1f6]">
                      {cat.label}
                    </span>
                  </div>
                  <span className="text-[9px] font-mono text-[#54596b] num-tabular">
                    {items.length}
                  </span>
                </button>

                {/* Items in Category */}
                {isExpanded && (
                  <div className="divide-y divide-[#181b24] bg-[#0c0d11]">
                    {items.length === 0 ? (
                      <div className="px-6 py-2 text-[10px] font-mono text-[#54596b] italic">
                        No matching elements
                      </div>
                    ) : (
                      items.map((entity) => {
                        const isSelected = selectedIds.has(entity.id);
                        const isHidden = entity.visibility === 'HIDDEN';
                        const isLocked = entity.lock === 'LOCKED';

                        return (
                          <div
                            key={entity.id}
                            onClick={() => selectEntity(entity.id)}
                            onContextMenu={(e) => handleContextMenu(e, entity.id)}
                            onMouseEnter={() => hoverEntity(entity.id)}
                            onMouseLeave={() => hoverEntity(undefined)}
                            className={`group flex items-center justify-between px-3 py-1.5 cursor-pointer text-xs font-mono transition-colors ${
                              isSelected
                                ? 'bg-[#3d8ef7]/15 text-[#6eb0ff] border-l-2 border-[#3d8ef7]'
                                : 'hover:bg-[#141720] text-[#c4c7d4]'
                            }`}
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="text-xs shrink-0">
                                {TYPE_ICONS[entity.type] || '◻️'}
                              </span>
                              <span className={`truncate text-[11px] ${isHidden ? 'opacity-40 line-through' : ''}`}>
                                {entity.name}
                              </span>
                            </div>

                            <div className="flex items-center gap-1.5 shrink-0 ml-2">
                              <ConfidenceDot confidence={entity.provenance.confidence} />
                              
                              {/* Hover Action Buttons */}
                              <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 transition-opacity">
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleEntityVisibility(entity.id);
                                  }}
                                  className="p-0.5 text-[#9296a6] hover:text-[#ededf2]"
                                >
                                  {isHidden ? (
                                    <EyeOff className="w-3 h-3 text-[#e74c3c]" />
                                  ) : (
                                    <Eye className="w-3 h-3" />
                                  )}
                                </button>
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    toggleEntityLock(entity.id);
                                  }}
                                  className="p-0.5"
                                >
                                  {isLocked ? (
                                    <Lock className="w-2.5 h-2.5 text-amber-400" />
                                  ) : (
                                    <Unlock className="w-2.5 h-2.5 text-[#54596b]" />
                                  )}
                                </button>
                              </div>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* ── Context Menu ───────────────────────────────────────────── */}
      {contextMenu && (
        <div
          ref={contextMenuRef}
          style={{ top: `${contextMenu.y}px`, left: `${contextMenu.x}px` }}
          className="fixed z-50 w-48 bg-[#141620] border border-[#2b3040] rounded-lg shadow-2xl py-1 text-[11px] font-mono select-none animate-in fade-in zoom-in-95 duration-100"
        >
          <div className="px-2.5 py-1 text-[9px] text-[#54596b] uppercase tracking-wider border-b border-[#1f222b]">
            Entity Actions
          </div>
          <button
            type="button"
            onClick={() => {
              selectEntity(contextMenu.entityId);
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#ededf2]"
          >
            <Focus className="w-3 h-3 text-[#3d8ef7]" />
            <span>Focus Selection (F)</span>
          </button>
          <button
            type="button"
            onClick={() => {
              isolateEntity(contextMenu.entityId);
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#ededf2]"
          >
            <Layers className="w-3 h-3 text-[#a855f7]" />
            <span>Isolate Selection (I)</span>
          </button>
          <button
            type="button"
            onClick={() => {
              showAllEntities();
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#ededf2]"
          >
            <EyeIcon className="w-3 h-3 text-[#2ecc71]" />
            <span>Show All Entities</span>
          </button>
          <button
            type="button"
            onClick={() => {
              toggleEntityVisibility(contextMenu.entityId);
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#ededf2]"
          >
            <EyeOff className="w-3 h-3 text-[#f59e0b]" />
            <span>Toggle Visibility (H)</span>
          </button>
          <button
            type="button"
            onClick={() => {
              toggleEntityLock(contextMenu.entityId);
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#ededf2]"
          >
            <Lock className="w-3 h-3 text-amber-400" />
            <span>Toggle Lock</span>
          </button>
          <div className="my-1 border-t border-[#1f222b]" />
          <button
            type="button"
            onClick={() => {
              setActiveWorkspace('evidence');
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#3d8ef7]"
          >
            <ShieldCheck className="w-3 h-3 text-[#3d8ef7]" />
            <span>Trace Evidence Lineage</span>
          </button>
          <button
            type="button"
            onClick={() => {
              setActiveMeasurementTool('POINT_TO_POINT');
              setContextMenu(null);
            }}
            className="w-full text-left px-2.5 py-1.5 hover:bg-[#1f2330] flex items-center gap-2 text-[#ededf2]"
          >
            <Ruler className="w-3 h-3 text-[#3d8ef7]" />
            <span>Measure Entity</span>
          </button>
        </div>
      )}
    </div>
  );
}
