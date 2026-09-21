'use client';

/**
 * WorldDiff — World Version Comparison & Diff Visualization.
 * - Compare any two WorldIR versions (base vs head)
 * - Entity-level diff: Added / Modified / Removed / Moved
 * - Property-level diff with uncertainty propagation
 * - Visual overlay in 3D viewport (color-coded changes)
 * - Provenance traceability for every change
 */

import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useREStore, Entity, EntityId, WorldVersionItem } from '@/store/re-store';
import { fetchWorldDiff, type BackendWorldDiffPayload } from '@/lib/api';
import {
  ChevronRight,
  ChevronDown,
  Plus,
  Minus,
  ArrowRight,
  GitCompare,
  Eye,
  EyeOff,
  Filter,
  X,
  Copy,
  Download,
  RotateCcw,
  History,
  AlertCircle,
  Loader2,
} from 'lucide-react';

export type DiffKind = 'ADDED' | 'REMOVED' | 'MODIFIED' | 'MOVED' | 'UNCHANGED';

export interface EntityDiff {
  entityId: EntityId;
  kind: DiffKind;
  name: string;
  type: string;
  baseEntity?: Partial<Entity>;
  headEntity?: Partial<Entity>;
  propertyDiffs: PropertyDiff[];
  spatialShift?: { distance: number; uncertainty: number };
  provenanceChange?: string;
}

export interface PropertyDiff {
  path: string;
  kind: DiffKind;
  baseValue?: unknown;
  headValue?: unknown;
  uncertaintyChange?: number;
  confidenceChange?: number;
}

interface WorldDiffProps {
  className?: string;
  onClose?: () => void;
  baseVersionId?: string;
  headVersionId?: string;
}

const DIFF_COLORS: Record<DiffKind, { bg: string; border: string; text: string; icon: string }> = {
  ADDED: { bg: '#22c55e15', border: '#22c55e', text: '#22c55e', icon: '+' },
  REMOVED: { bg: '#ef444415', border: '#ef4444', text: '#ef4444', icon: '−' },
  MODIFIED: { bg: '#3d8ef715', border: '#3d8ef7', text: '#3d8ef7', icon: '≈' },
  MOVED: { bg: '#a855f715', border: '#a855f7', text: '#a855f7', icon: '⇄' },
  UNCHANGED: { bg: '#1f222b', border: '#1f222b', text: '#54596b', icon: '=' },
};

export function WorldDiff({
  className = '',
  onClose,
  baseVersionId: propBaseVersionId,
  headVersionId: propHeadVersionId,
}: WorldDiffProps) {
  const {
    world,
    entities,
    selectedVersionId,
    worldVersions,
    refreshWorldVersions,
    setSelectedVersionId,
    setActiveWorkspace,
    addNotification,
  } = useREStore();

  const versions: WorldVersionItem[] = useMemo(() => {
    if (worldVersions && worldVersions.length > 0) {
      return worldVersions.map((wv) => ({
        id: wv.version_id,
        label: wv.name || `World ${wv.world_id}`,
        tag: wv.parent ? `PARENT_${wv.parent.slice(0, 8)}` : 'BASELINE',
        isCurrent: wv.version_id === (selectedVersionId || worldVersions[worldVersions.length - 1]?.version_id),
        date: wv.created_at ? new Date(wv.created_at * 1000).toISOString().split('T')[0] : 'stored',
      }));
    }
    return [];
  }, [worldVersions, selectedVersionId]);

  const [baseVersionId, setBaseVersionId] = useState(propBaseVersionId || '');
  const [headVersionId, setHeadVersionId] = useState(propHeadVersionId || selectedVersionId || '');

  const effectiveBaseVersionId = baseVersionId || (worldVersions.length >= 2 ? worldVersions[worldVersions.length - 2].version_id : worldVersions[0]?.version_id || '');
  const effectiveHeadVersionId = headVersionId || selectedVersionId || (worldVersions.length >= 1 ? worldVersions[worldVersions.length - 1].version_id : '');

  useEffect(() => {
    refreshWorldVersions();
  }, [refreshWorldVersions]);

  const [filterKind, setFilterKind] = useState<DiffKind | 'ALL'>('ALL');
  const [expandedDiffs, setExpandedDiffs] = useState<Set<EntityId>>(new Set());
  const [showOnlyChanged, setShowOnlyChanged] = useState(true);
  const [viewMode, setViewMode] = useState<'LIST' | 'SPATIAL'>('LIST');

  const [backendDiff, setBackendDiff] = useState<BackendWorldDiffPayload | null>(null);
  const [isLoadingDiff, setIsLoadingDiff] = useState(false);
  const [diffError, setDiffError] = useState<string | null>(null);

  useEffect(() => {
    if (!effectiveBaseVersionId || !effectiveHeadVersionId) {
      return;
    }
    let active = true;
    Promise.resolve().then(() => {
      if (active) {
        setIsLoadingDiff(true);
        setDiffError(null);
      }
    });

    fetchWorldDiff(effectiveBaseVersionId, effectiveHeadVersionId).then((res) => {
      if (!active) return;
      setIsLoadingDiff(false);
      if (res.available && res.diff) {
        setBackendDiff(res.diff);
        setDiffError(null);
      } else {
        setBackendDiff(null);
        setDiffError(res.error || 'Failed to compute diff between versions');
      }
    });

    return () => {
      active = false;
    };
  }, [effectiveBaseVersionId, effectiveHeadVersionId]);

  const diffs = useMemo((): EntityDiff[] => {
    if (!backendDiff) return [];
    const result: EntityDiff[] = [];

    for (const ed of backendDiff.entities || []) {
      const kind: DiffKind =
        ed.kind === 'added' ? 'ADDED' :
        ed.kind === 'removed' ? 'REMOVED' :
        ed.kind === 'modified' ? 'MODIFIED' : 'UNCHANGED';

      const existing = entities.get(ed.entity_id);
      const name = existing?.name || ed.entity_id;
      const type = existing?.type || 'ENTITY';

      const propertyDiffs: PropertyDiff[] = (ed.changes || []).map((c) => ({
        path: c.field,
        kind: 'MODIFIED',
        baseValue: c.old,
        headValue: c.new,
      }));

      result.push({
        entityId: ed.entity_id,
        kind,
        name,
        type,
        baseEntity: existing ? { ...existing } : undefined,
        headEntity: existing ? { ...existing } : undefined,
        propertyDiffs,
        provenanceChange: `Change detected in field(s): ${ed.changes.map(c => c.field).join(', ') || 'state'}`,
      });
    }

    for (const gd of backendDiff.geometries || []) {
      const kind: DiffKind =
        gd.kind === 'added' ? 'ADDED' :
        gd.kind === 'removed' ? 'REMOVED' :
        gd.kind === 'modified' ? 'MODIFIED' : 'UNCHANGED';

      const propertyDiffs: PropertyDiff[] = (gd.changes || []).map((c) => ({
        path: c.field,
        kind: 'MODIFIED',
        baseValue: c.old,
        headValue: c.new,
      }));

      result.push({
        entityId: gd.geometry_id,
        kind,
        name: `Geometry ${gd.geometry_id.slice(0, 8)}`,
        type: 'GEOMETRY',
        propertyDiffs,
        provenanceChange: `Geometry geometry_id=${gd.geometry_id}`,
      });
    }

    return result;
  }, [backendDiff, entities]);

  const filteredDiffs = useMemo(() => {
    return diffs.filter(d => filterKind === 'ALL' || d.kind === filterKind);
  }, [diffs, filterKind]);

  const stats = useMemo(() => {
    if (backendDiff?.summary) {
      const s = backendDiff.summary;
      return {
        ADDED: s.entities_added + s.geometries_added,
        REMOVED: s.entities_removed + s.geometries_removed,
        MODIFIED: s.entities_modified + s.geometries_modified,
        MOVED: 0,
        UNCHANGED: 0,
      };
    }
    const counts: Record<DiffKind, number> = { ADDED: 0, REMOVED: 0, MODIFIED: 0, MOVED: 0, UNCHANGED: 0 };
    diffs.forEach(d => counts[d.kind]++);
    return counts;
  }, [backendDiff, diffs]);

  const toggleExpand = useCallback((id: EntityId) => {
    setExpandedDiffs(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const copyDiffSummary = useCallback(() => {
    const summary = `World Diff: ${baseVersionId} → ${headVersionId}\n` +
      `Added: ${stats.ADDED} | Modified: ${stats.MODIFIED} | Removed: ${stats.REMOVED} | Moved: ${stats.MOVED}\n` +
      filteredDiffs.map(d => `${DIFF_COLORS[d.kind].icon} ${d.name} (${d.type})`).join('\n');
    navigator.clipboard.writeText(summary);
    addNotification({ type: 'success', title: 'Copied', message: 'Diff summary copied to clipboard' });
  }, [baseVersionId, headVersionId, stats, filteredDiffs, addNotification]);

  return (
    <div className={`flex flex-col h-full bg-[#0c0d11] text-[#ededf2] ${className}`}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-3 h-8 border-b border-[#1f222b] bg-[#0f1014]">
        <div className="flex items-center gap-2">
          <GitCompare className="w-4 h-4 text-[#3d8ef7]" />
          <span className="text-[10px] font-mono font-bold tracking-wider text-[#54596b] uppercase">World Diff</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={copyDiffSummary}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#9296a6] hover:text-[#ededf2] bg-[#14161f] border border-[#1f222b] transition-colors"
            title="Copy diff summary"
          >
            <Copy className="w-3 h-3" />
            <span>Copy</span>
          </button>
          <button
            type="button"
            onClick={onClose}
            className="p-1 text-[#9296a6] hover:text-[#ededf2] rounded transition-colors"
            title="Close"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Version Selectors ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217] space-y-2">
        <div className="grid grid-cols-2 gap-2">
          {/* Base Version */}
          <div>
            <label className="text-[9px] font-mono text-[#54596b] uppercase tracking-wider block mb-1">
              Base Version (From)
            </label>
            <select
              value={effectiveBaseVersionId}
              onChange={e => setBaseVersionId(e.target.value)}
              className="w-full px-2 py-1.5 rounded bg-[#14161f] border border-[#1f222b] text-[11px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
            >
              {versions.length === 0 ? (
                <option value="">No stored versions</option>
              ) : (
                versions.map(v => (
                  <option key={v.id} value={v.id}>
                    {v.id} — {v.label} ({v.tag})
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Head Version */}
          <div>
            <label className="text-[9px] font-mono text-[#54596b] uppercase tracking-wider block mb-1">
              Head Version (To)
            </label>
            <select
              value={effectiveHeadVersionId}
              onChange={e => setHeadVersionId(e.target.value)}
              className="w-full px-2 py-1.5 rounded bg-[#14161f] border border-[#1f222b] text-[11px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
            >
              {versions.length === 0 ? (
                <option value="">No stored versions</option>
              ) : (
                versions.map(v => (
                  <option key={v.id} value={v.id}>
                    {v.id} — {v.label} ({v.tag})
                  </option>
                ))
              )}
            </select>
          </div>
        </div>

        {/* Swap & Filters */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => {
              setBaseVersionId(effectiveHeadVersionId);
              setHeadVersionId(effectiveBaseVersionId);
            }}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#9296a6] hover:text-[#ededf2] bg-[#14161f] border border-[#1f222b] transition-colors"
            title="Swap base/head"
          >
            <RotateCcw className="w-3 h-3" />
            <span>Swap</span>
          </button>

          <div className="flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-[#54596b]" />
            <select
              value={filterKind}
              onChange={e => setFilterKind(e.target.value as DiffKind | 'ALL')}
              className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
            >
              <option value="ALL">All Changes ({diffs.length})</option>
              <option value="ADDED">Added ({stats.ADDED})</option>
              <option value="MODIFIED">Modified ({stats.MODIFIED})</option>
              <option value="REMOVED">Removed ({stats.REMOVED})</option>
              <option value="MOVED">Moved ({stats.MOVED})</option>
            </select>
            <label className="flex items-center gap-1 text-[10px] text-[#9296a6] cursor-pointer">
              <input
                type="checkbox"
                checked={showOnlyChanged}
                onChange={e => setShowOnlyChanged(e.target.checked)}
                className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7] focus:ring-[#3d8ef7]"
              />
              <span>Only Changed</span>
            </label>
            <select
              value={viewMode}
              onChange={e => setViewMode(e.target.value as 'LIST' | 'SPATIAL')}
              className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
            >
              <option value="LIST">List View</option>
              <option value="SPATIAL">Spatial Overlay</option>
            </select>
          </div>
        </div>
      </div>

      {/* ── Stats Summary ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217]">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#22c55e]/15 border border-[#22c55e]/40">
            <span className="text-xs font-bold text-[#22c55e">+{stats.ADDED}</span>
            <span className="text-[10px] text-[#9296a6]">Added</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#3d8ef7]/15 border border-[#3d8ef7]/40">
            <span className="text-xs font-bold text-[#3d8ef7">≈{stats.MODIFIED}</span>
            <span className="text-[10px] text-[#9296a6]">Modified</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#ef4444]/15 border border-[#ef4444]/40">
            <span className="text-xs font-bold text-[#ef4444]">−{stats.REMOVED}</span>
            <span className="text-[10px] text-[#9296a6]">Removed</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#a855f7]/15 border border-[#a855f7]/40">
            <span className="text-xs font-bold text-[#a855f7]">⇄{stats.MOVED}</span>
            <span className="text-[10px] text-[#9296a6]">Moved</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#1f222b] border border-[#1f222b] ml-auto">
            <span className="text-xs font-bold text-[#54596b]">{filteredDiffs.length}</span>
            <span className="text-[10px] text-[#9296a6]">Showing</span>
          </div>
        </div>
      </div>

      {/* ── Diff List ── */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoadingDiff ? (
          <div className="flex flex-col items-center justify-center h-full text-[#3d8ef7]">
            <Loader2 className="w-8 h-8 animate-spin mb-2" />
            <span className="text-xs font-mono">Computing structural diff...</span>
          </div>
        ) : diffError ? (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center text-[#ef4444]">
            <AlertCircle className="w-10 h-10 mb-2 text-[#ef4444]" />
            <p className="text-xs font-mono font-bold uppercase tracking-wider">DIFF: UNAVAILABLE</p>
            <p className="text-[11px] font-mono mt-1 text-[#f87171] max-w-sm">{diffError}</p>
            <p className="text-[10px] text-[#54596b] mt-3">Select two stored WorldStore versions to compare.</p>
          </div>
        ) : versions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center text-[#54596b]">
            <History className="w-10 h-10 mb-2 text-[#1f222b]" />
            <p className="text-xs font-mono font-bold uppercase tracking-wider">NO STORED VERSIONS</p>
            <p className="text-[11px] mt-1 text-[#54596b] max-w-sm">No versions found in WorldStore. Save multiple versions to visualize diffs.</p>
          </div>
        ) : filteredDiffs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-[#54596b]">
            <GitCompare className="w-12 h-12 text-[#1f222b] mb-3" />
            <p className="text-sm font-mono">No differences found</p>
            <p className="text-[10px] mt-1">Versions are identical or filter hides all changes</p>
          </div>
        ) : (
          filteredDiffs.map(diff => {
            const colors = DIFF_COLORS[diff.kind];
            const isExpanded = expandedDiffs.has(diff.entityId);
            const entity = diff.headEntity || diff.baseEntity;

            return (
              <div
                key={diff.entityId}
                className={`border rounded-lg overflow-hidden ${isExpanded ? 'bg-[#14161f]' : 'bg-[#101217]'} border-[${colors.border}]/30`}
              >
                {/* Diff Row Header */}
                <button
                  type="button"
                  onClick={() => toggleExpand(diff.entityId)}
                  className="w-full flex items-center gap-2 px-3 py-2 hover:bg-[#141720] transition-colors"
                >
                  <span
                    className="w-5 h-5 flex items-center justify-center rounded text-[10px] font-bold flex-shrink-0"
                    style={{ background: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}
                  >
                    {colors.icon}
                  </span>
                  <span className="text-xs font-mono font-semibold text-[#f0f1f6] truncate">{diff.name}</span>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#1f222b] text-[#9296a6]">{diff.type}</span>

                  <div className="flex-1" />

                  {diff.spatialShift && (
                    <span className="flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-mono text-[#a855f7] bg-[#a855f7]/10 border border-[#a855f7]/30">
                      <ArrowRight className="w-2.5 h-2.5" />
                      <span>Shift: {diff.spatialShift.distance.toFixed(3)}m ± {diff.spatialShift.uncertainty.toFixed(4)}m</span>
                    </span>
                  )}

                  {diff.provenanceChange && (
                    <span className="text-[9px] text-[#54596b] max-w-[200px] truncate flex-1 pr-2">{diff.provenanceChange}</span>
                  )}

                  <ChevronDown
                    className={`w-3 h-3 text-[#54596b] transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                  />
                </button>

                {/* Expanded Property Diffs */}
                {isExpanded && (
                  <div className="border-t border-[#1f222b] bg-[#0c0d11] p-3 font-mono text-xs space-y-2">
                    {diff.propertyDiffs.length === 0 ? (
                      <div className="text-[10px] text-[#54596b] italic">No property changes (entity-level only)</div>
                    ) : (
                      diff.propertyDiffs.map((prop, i) => {
                        const pColors = DIFF_COLORS[prop.kind];
                        return (
                          <div key={i} className="p-2 rounded bg-[#14161f] border border-[#1f222b]/50">
                            <div className="flex items-center justify-between mb-1">
                              <div className="flex items-center gap-1.5">
                                <span
                                  className="w-4 h-4 flex items-center justify-center rounded text-[9px] font-bold"
                                  style={{ background: pColors.bg, color: pColors.text, border: `1px solid ${pColors.border}` }}
                                >
                                  {pColors.icon}
                                </span>
                                <span className="text-[#c4c7d4]">{prop.path}</span>
                                <span className="text-[9px] text-[#54596b]">({prop.kind})</span>
                              </div>
                              <div className="flex items-center gap-3 text-[9px]">
                                {prop.baseValue !== undefined && (
                                  <span className="text-[#ef4444]">− {JSON.stringify(prop.baseValue)}</span>
                                )}
                                {prop.headValue !== undefined && (
                                  <span className="text-[#22c55e]">+ {JSON.stringify(prop.headValue)}</span>
                                )}
                                {prop.uncertaintyChange !== undefined && (
                                  <span className="text-[#a855f7] flex items-center gap-0.5">
                                    <span>σ</span>
                                    {prop.uncertaintyChange > 0 ? '+' : ''}{prop.uncertaintyChange.toFixed(4)}m
                                  </span>
                                )}
                                {prop.confidenceChange !== undefined && (
                                  <span className="text-[#3d8ef7] flex items-center gap-0.5">
                                    <span>conf</span>
                                    {prop.confidenceChange > 0 ? '+' : ''}{(prop.confidenceChange * 100).toFixed(1)}%
                                  </span>
                                )}
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
    </div>
  );
}