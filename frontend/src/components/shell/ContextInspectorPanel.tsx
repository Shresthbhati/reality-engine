'use client';

/**
 * Context Inspector — every value from the SELECTED entity's real
 * backend record; nothing invented.
 *
 * The previous version fell back to a hardcoded "Central Street Light
 * #14" with fabricated mesh counts, invented EXIF-style provenance and
 * a made-up temporal history whenever no entity was selected — a fake
 * dashboard. Now:
 *  - no selection  -> explicit "no entity selected" state;
 *  - real entity   -> sections render only what the backend actually
 *    carries (bounds from geometry, provenance state/confidence/
 *    algorithm, observation count, relationships from WorldIR);
 *  - a field the backend does not measure (material, height, per-LOD
 *    resolution, per-observation residuals) is OMITTED, not decorated.
 */

import React, { useState } from 'react';
import { useREStore } from '@/store/re-store';
import {
  PanelRightClose,
  Box,
  Layers,
  Sparkles,
  Camera,
  Activity,
  Network,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import type { Entity } from '@/types/reality-engine';

interface EntityRelationship {
  kind: string;
  target_id: string;
  confidence: number;
}

function relationshipsOf(entity: Entity): EntityRelationship[] {
  const rels = entity.metadata.relationships as EntityRelationship[] | undefined;
  return Array.isArray(rels) ? rels : [];
}

function boundsOf(entity: Entity): {
  min: [number, number, number];
  max: [number, number, number];
  extent: [number, number, number];
} | null {
  const b = entity.metadata.bounds as
    | { min: [number, number, number]; max: [number, number, number]; extent?: [number, number, number] }
    | undefined;
  if (!b || !b.min || !b.max) return null;
  return {
    min: b.min,
    max: b.max,
    extent: b.extent ?? [
      b.max[0] - b.min[0],
      b.max[1] - b.min[1],
      b.max[2] - b.min[2],
    ],
  };
}

export default function ContextInspectorPanel() {
  const selection = useREStore((s) => s.selection);
  const entities = useREStore((s) => s.entities);
  const toggleInspector = useREStore((s) => s.toggleInspector);
  const spatialScale = useREStore((s) => s.spatialScale);

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    geometry: true,
    semantics: true,
    evidence: true,
    provenance: true,
    uncertainty: true,
    relationships: true,
  });

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const selectedEntityId = selection.selectedEntityIds[0] ?? selection.focusedEntityId;
  const entity: Entity | undefined = selectedEntityId
    ? entities.get(selectedEntityId)
    : undefined;

  return (
    <aside
      className="flex flex-col w-80 h-full bg-[#101217] border-l border-[#1f222b] select-none text-[#ededf2] font-sans shrink-0 overflow-hidden"
      aria-label="Context Inspector Panel"
    >
      {/* ── Header ── */}
      <div className="h-10 px-3 border-b border-[#1f222b] flex items-center justify-between shrink-0 bg-[#0d0e12]">
        <div className="flex items-center gap-1.5 truncate">
          <span className="text-[11px] font-mono font-bold tracking-wider text-[#9296a6] uppercase">
            Context Inspector
          </span>
          <span className="text-[9px] font-mono px-1 rounded bg-[#161a24] text-[#00e5ff] border border-[#00e5ff]/30">
            {spatialScale}
          </span>
        </div>

        <button
          type="button"
          onClick={toggleInspector}
          className="p-1 rounded text-[#54596b] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
          title="Collapse Panel (⌘I)"
        >
          <PanelRightClose className="w-3.5 h-3.5" />
        </button>
      </div>

      {!entity ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 px-6 text-center">
          <Box className="w-6 h-6 text-[#3d3d55]" />
          <div className="text-[11px] font-semibold uppercase tracking-wider text-[#9296a6]">
            NO ENTITY SELECTED
          </div>
          <div className="text-[10px] font-mono text-[#54596b]">
            Select an entity in the Outliner or Viewport. The inspector
            shows only what the backend has actually measured for it.
          </div>
        </div>
      ) : (
        <>
          {/* ── Entity Title / Badge ── */}
          <div className="p-3 border-b border-[#1f222b] bg-[#0f1116] shrink-0">
            <div className="flex items-center justify-between gap-1">
              <div className="flex items-center gap-1.5 truncate">
                <div className="w-2 h-2 rounded-full bg-[#00e5ff] shadow-[0_0_6px_#00e5ff]" />
                <h3 className="text-xs font-bold text-[#f0f1f6] truncate font-sans">
                  {entity.name}
                </h3>
              </div>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-white/5 text-[#9296a6]">
                {entity.type}
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1 text-[10px] font-mono text-[#54596b]">
              <span className="truncate max-w-[150px]">ID: {entity.id}</span>
              <span>•</span>
              {/* Provenance state comes from the backend, not a hardcoded VERIFIED. */}
              <span
                className={
                  entity.provenance.state === 'RECONSTRUCTED' || entity.provenance.state === 'OBSERVED'
                    ? 'text-[#2ecc71] font-semibold'
                    : 'text-[#f5a623] font-semibold'
                }
              >
                {entity.provenance.state}
              </span>
            </div>
          </div>

          {/* ── Scrollable Inspector Sections ── */}
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3 font-mono text-xs">
            {/* 1. GEOMETRY — bounds from the backend's geometry payload. */}
            {boundsOf(entity) && (
              <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
                <button
                  type="button"
                  onClick={() => toggleSection('geometry')}
                  className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
                >
                  <span className="flex items-center gap-1.5">
                    <Box className="w-3.5 h-3.5 text-[#38bdf8]" />
                    <span className="text-[#f0f1f6]">GEOMETRY</span>
                  </span>
                  {openSections.geometry ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                </button>
                {openSections.geometry && (
                  <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Representations:</span>
                      <span className="text-[#f0f1f6]">{entity.representations.join(', ')}</span>
                    </div>
                    <div className="text-[#54596b] pt-1">Bounds (world units):</div>
                    <div className="grid grid-cols-2 gap-x-2 font-num">
                      <span>min</span>
                      <span className="text-[#f0f1f6] text-right truncate">
                        {boundsOf(entity)!.min.map((v) => v.toFixed(2)).join(', ')}
                      </span>
                      <span>max</span>
                      <span className="text-[#f0f1f6] text-right truncate">
                        {boundsOf(entity)!.max.map((v) => v.toFixed(2)).join(', ')}
                      </span>
                      <span>extent</span>
                      <span className="text-[#00e5ff] text-right truncate">
                        {boundsOf(entity)!.extent.map((v) => v.toFixed(2)).join(' × ')}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 2. SEMANTICS — the entity's real class and tags. */}
            <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection('semantics')}
                className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
              >
                <span className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-[#a855f7]" />
                  <span className="text-[#f0f1f6]">SEMANTICS</span>
                </span>
                {openSections.semantics ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>

              {openSections.semantics && (
                <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
                  <div className="flex justify-between">
                    <span className="text-[#54596b]">Class:</span>
                    <span className="text-[#f0f1f6] font-semibold">{entity.type}</span>
                  </div>
                  {entity.tags.length > 0 && (
                    <div className="flex justify-between gap-2">
                      <span className="text-[#54596b] shrink-0">Tags:</span>
                      <span className="text-[#f0f1f6] text-right">{entity.tags.join(', ')}</span>
                    </div>
                  )}
                  {entity.childIds.length > 0 && (
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Children:</span>
                      <span className="text-[#f0f1f6] font-semibold">{entity.childIds.length}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 3. EVIDENCE — real observation count; detail in the Evidence workspace. */}
            <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection('evidence')}
                className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
              >
                <span className="flex items-center gap-1.5">
                  <Camera className="w-3.5 h-3.5 text-[#2ecc71]" />
                  <span className="text-[#f0f1f6]">EVIDENCE</span>
                </span>
                {openSections.evidence ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>

              {openSections.evidence && (
                <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
                  <div className="flex justify-between">
                    <span className="text-[#54596b]">Observations:</span>
                    <span className="text-[#f0f1f6] font-bold font-num">{entity.observationCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#54596b]">Sessions:</span>
                    <span className="text-[#f0f1f6]">{entity.sessionIds.join(', ')}</span>
                  </div>
                </div>
              )}
            </div>

            {/* 4. PROVENANCE — state, confidence, algorithm as recorded. */}
            <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection('provenance')}
                className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
              >
                <span className="flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-[#ec4899]" />
                  <span className="text-[#f0f1f6]">PROVENANCE</span>
                </span>
                {openSections.provenance ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>

              {openSections.provenance && (
                <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
                  <div className="flex justify-between">
                    <span className="text-[#54596b]">State:</span>
                    <span className="text-[#f0f1f6] font-semibold">{entity.provenance.state}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#54596b]">Algorithm:</span>
                    <span className="text-[#f0f1f6] truncate max-w-[150px]">
                      {entity.provenance.algorithm ?? 'not recorded'}
                    </span>
                  </div>
                  {entity.provenance.coordinateFrame && (
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Frame:</span>
                      <span className="text-[#f0f1f6]">{entity.provenance.coordinateFrame}</span>
                    </div>
                  )}
                  {entity.provenance.version && (
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Version:</span>
                      <span className="text-[#f0f1f6]">{entity.provenance.version}</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 5. CONFIDENCE / UNCERTAINTY — only when the backend measured them. */}
            <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
              <button
                type="button"
                onClick={() => toggleSection('uncertainty')}
                className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
              >
                <span className="flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-[#f5a623]" />
                  <span className="text-[#f0f1f6]">CONFIDENCE</span>
                </span>
                {openSections.uncertainty ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>

              {openSections.uncertainty && (
                <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
                  <div className="flex justify-between">
                    <span className="text-[#54596b]">Confidence:</span>
                    <span className="text-[#2ecc71] font-bold">
                      {(entity.provenance.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {typeof entity.provenance.uncertainty === 'number' && (
                    <div className="flex justify-between">
                      <span className="text-[#54596b]">Uncertainty:</span>
                      <span className="text-[#f5a623] font-bold">± {entity.provenance.uncertainty} m</span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* 6. RELATIONSHIPS — straight from WorldIR relationships. */}
            {relationshipsOf(entity).length > 0 && (
              <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
                <button
                  type="button"
                  onClick={() => toggleSection('relationships')}
                  className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
                >
                  <span className="flex items-center gap-1.5">
                    <Network className="w-3.5 h-3.5 text-[#00e5ff]" />
                    <span className="text-[#f0f1f6]">RELATIONSHIPS</span>
                  </span>
                  {openSections.relationships ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                </button>

                {openSections.relationships && (
                  <div className="p-2.5 pt-0 space-y-1.5 text-[9px] text-[#9296a6] border-t border-white/5 mt-1">
                    {relationshipsOf(entity).map((rel, i) => (
                      <div key={`${rel.kind}-${rel.target_id}-${i}`} className="flex justify-between items-center p-1 rounded hover:bg-white/5">
                        <span className="text-[#54596b]">{rel.kind}:</span>
                        <span className="text-[#f0f1f6] font-semibold truncate max-w-[140px]">{rel.target_id}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
