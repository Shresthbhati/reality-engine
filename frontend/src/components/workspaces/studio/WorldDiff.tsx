'use client';

/**
 * WorldDiff — World Version Comparison & Diff Visualization.
 * - Compare any two WorldIR versions (base vs head)
 * - Entity-level diff: Added / Modified / Removed / Moved
 * - Property-level diff with uncertainty propagation
 * - Visual overlay in 3D viewport (color-coded changes)
 * - Provenance traceability for every change
 */

import React, { useState, useMemo, useCallback } from 'react';
import { useREStore, Entity, EntityId, WorldVersionItem } from '@/store/re-store';
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
    setSelectedVersionId,
    setActiveWorkspace,
    addNotification,
  } = useREStore();

  const [baseVersionId, setBaseVersionId] = useState(propBaseVersionId || 'v1');
  const [headVersionId, setHeadVersionId] = useState(propHeadVersionId || selectedVersionId || 'v3');
  const [filterKind, setFilterKind] = useState<DiffKind | 'ALL'>('ALL');
  const [expandedDiffs, setExpandedDiffs] = useState<Set<EntityId>>(new Set());
  const [showOnlyChanged, setShowOnlyChanged] = useState(true);
  const [viewMode, setViewMode] = useState<'LIST' | 'SPATIAL'>('LIST');

  const versions: WorldVersionItem[] = useMemo(() => [
    { id: 'v1', label: 'Initial Reconstruction', tag: 'BASELINE', isCurrent: false, date: '2026-09-14' },
    { id: 'v2', label: 'Facade Pass Integration', tag: 'FACADE_PASS', isCurrent: false, date: '2026-09-15' },
    { id: 'v3', label: 'Detail + Micro-Detail Pass', tag: 'MICRO_DETAIL', isCurrent: true, date: '2026-09-16' },
    { id: 'v4', label: 'LiDAR Fusion', tag: 'LIDAR_FUSION', isCurrent: false, date: '2026-09-17' },
  ], []);

  // Generate mock diff data (in production, this comes from `reality diff` CLI or WorldStore)
  const diffs = useMemo((): EntityDiff[] => {
    const mockDiffs: EntityDiff[] = [
      {
        entityId: 'ent-micro-weathering',
        kind: 'ADDED',
        name: 'Surface Weathering & Fine Marble Veins',
        type: 'OBJECT',
        headEntity: entities.get('ent-micro-weathering'),
        propertyDiffs: [],
        provenanceChange: 'MICRO_DETAIL_PASS: Sub-pixel Disparity Fusion from oblique macro captures',
      },
      {
        entityId: 'ent-ornament-floral',
        kind: 'MODIFIED',
        name: 'Carved Leaf & Floral Relief',
        type: 'OBJECT',
        baseEntity: { ...entities.get('ent-ornament-floral')!, provenance: { ...entities.get('ent-ornament-floral')!.provenance, confidence: 0.92 } },
        headEntity: entities.get('ent-ornament-floral'),
        propertyDiffs: [
          { path: 'provenance.confidence', kind: 'MODIFIED', baseValue: 0.92, headValue: 0.96, confidenceChange: 0.04 },
          { path: 'provenance.uncertainty', kind: 'MODIFIED', baseValue: 0.003, headValue: 0.001, uncertaintyChange: -0.002 },
          { path: 'metadata.relief_depth_mm', kind: 'MODIFIED', baseValue: 42, headValue: 45, uncertaintyChange: 0.001 },
        ],
        spatialShift: { distance: 0.002, uncertainty: 0.0005 },
        provenanceChange: 'DETAIL_PASS → MICRO_DETAIL_PASS: Sub-pixel refinement reduced uncertainty by 67%',
      },
      {
        entityId: 'ent-capital-ionic-04',
        kind: 'MODIFIED',
        name: 'Carved Ionic Capital & Volutes',
        type: 'OBJECT',
        baseEntity: { ...entities.get('ent-capital-ionic-04')!, provenance: { ...entities.get('ent-capital-ionic-04')!.provenance, confidence: 0.94 } },
        headEntity: entities.get('ent-capital-ionic-04'),
        propertyDiffs: [
          { path: 'provenance.confidence', kind: 'MODIFIED', baseValue: 0.94, headValue: 0.97, confidenceChange: 0.03 },
          { path: 'provenance.uncertainty', kind: 'MODIFIED', baseValue: 0.005, headValue: 0.002, uncertaintyChange: -0.003 },
        ],
        spatialShift: { distance: 0.003, uncertainty: 0.001 },
        provenanceChange: 'DETAIL_PASS → MICRO_DETAIL_PASS: Dense macro captures improved geometry',
      },
      {
        entityId: 'ent-column-ionic-04',
        kind: 'MODIFIED',
        name: 'Ionic Fluted Column #4',
        type: 'WALL',
        baseEntity: { ...entities.get('ent-column-ionic-04')!, provenance: { ...entities.get('ent-column-ionic-04')!.provenance, confidence: 0.96 } },
        headEntity: entities.get('ent-column-ionic-04'),
        propertyDiffs: [
          { path: 'provenance.confidence', kind: 'MODIFIED', baseValue: 0.96, headValue: 0.99, confidenceChange: 0.03 },
          { path: 'metadata.flute_count', kind: 'MODIFIED', baseValue: 22, headValue: 24, confidenceChange: 0.02 },
        ],
        spatialShift: { distance: 0.001, uncertainty: 0.0008 },
        provenanceChange: 'FACADE_PASS → DETAIL_PASS: Additional oblique views resolved flute geometry',
      },
      {
        entityId: 'ent-dome-central',
        kind: 'MODIFIED',
        name: 'Central Queen\'s Dome and Angel of Victory',
        type: 'WALL',
        baseEntity: { ...entities.get('ent-dome-central')!, provenance: { ...entities.get('ent-dome-central')!.provenance, confidence: 0.94 } },
        headEntity: entities.get('ent-dome-central'),
        propertyDiffs: [
          { path: 'provenance.confidence', kind: 'MODIFIED', baseValue: 0.94, headValue: 0.97, confidenceChange: 0.03 },
          { path: 'provenance.uncertainty', kind: 'MODIFIED', baseValue: 0.025, headValue: 0.018, uncertaintyChange: -0.007 },
        ],
        spatialShift: { distance: 0.005, uncertainty: 0.002 },
        provenanceChange: 'DETAIL_PASS: Dense stereo on dome curvature reduced apex uncertainty',
      },
      {
        entityId: 'ent-facade-south',
        kind: 'ADDED',
        name: 'South Facade & Terrace',
        type: 'WALL',
        headEntity: {
          id: 'ent-facade-south',
          name: 'South Facade & Terrace',
          type: 'WALL',
          provenance: { state: 'RECONSTRUCTED', confidence: 0.78, algorithm: 'Dense Multi-view Stereo', uncertainty: 0.038 },
          sessionIds: ['sess-003'],
        },
        propertyDiffs: [],
        provenanceChange: 'LIDAR_FUSION: Terrestrial LiDAR scans added south facade coverage',
      },
      {
        entityId: 'ent-structure-east',
        kind: 'REMOVED',
        name: 'East Facade Glazed Marble (Low Confidence)',
        type: 'WALL',
        baseEntity: {
          id: 'ent-structure-east',
          name: 'East Facade Glazed Marble',
          type: 'WALL',
          provenance: { state: 'RECONSTRUCTED', confidence: 0.42, algorithm: 'Dense Multi-view Stereo', uncertainty: 0.12 },
          sessionIds: ['sess-002'],
        },
        propertyDiffs: [],
        provenanceChange: 'VALIDATION_GATE: Insufficient feature matches (inlier ratio 0.08) — entity rejected',
      },
    ];
    return mockDiffs;
  }, [entities]);

  const filteredDiffs = useMemo(() => {
    return diffs.filter(d => filterKind === 'ALL' || d.kind === filterKind);
  }, [diffs, filterKind]);

  const stats = useMemo(() => {
    const counts: Record<DiffKind, number> = { ADDED: 0, REMOVED: 0, MODIFIED: 0, MOVED: 0, UNCHANGED: 0 };
    diffs.forEach(d => counts[d.kind]++);
    return counts;
  }, [diffs]);

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
              value={baseVersionId}
              onChange={e => setBaseVersionId(e.target.value)}
              className="w-full px-2 py-1.5 rounded bg-[#14161f] border border-[#1f222b] text-[11px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
            >
              {versions.map(v => (
                <option key={v.id} value={v.id}>
                  {v.label} ({v.tag}) — {v.date}
                </option>
              ))}
            </select>
          </div>

          {/* Head Version */}
          <div>
            <label className="text-[9px] font-mono text-[#54596b] uppercase tracking-wider block mb-1">
              Head Version (To)
            </label>
            <select
              value={headVersionId}
              onChange={e => setHeadVersionId(e.target.value)}
              className="w-full px-2 py-1.5 rounded bg-[#14161f] border border-[#1f222b] text-[11px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
            >
              {versions.map(v => (
                <option key={v.id} value={v.id}>
                  {v.label} ({v.tag}) — {v.date}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Swap & Filters */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => {
              setBaseVersionId(headVersionId);
              setHeadVersionId(baseVersionId);
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
        {filteredDiffs.length === 0 ? (
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