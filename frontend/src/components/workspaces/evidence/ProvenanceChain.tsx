'use client';

/**
 * Provenance lineage chain, built from REAL store state.
 *
 * The previous version rendered a hardcoded MOCK_PROVENANCE_NODES list
 * (fake DJI/iPhone frames, invented confidences) regardless of what the
 * backend held. Now every node is derived from the selected entity:
 * the entity itself -> its provenance record -> each contributing
 * observation (frame id + sensor type + measured confidence straight
 * from the ReconstructionContract observations).
 *
 * Honesty rules:
 *  - No selection or an entity with no provenance -> explicit empty
 *    state, never a demo chain.
 *  - Backend-unavailable is surfaced, not papered over.
 *  - Confidence/uncertainty shown are the entity's own recorded values;
 *    nothing is recomputed or decorated.
 */

import React from 'react';
import { GitCommit, Box, ShieldCheck, Database, CircleAlert } from 'lucide-react';
import { useREStore } from '@/store/re-store';
import type { Entity } from '@/types/reality-engine';

export interface ProvenanceNode {
  id: string;
  type: string;
  title: string;
  count: string;
  detail: string;
  confidence: number;
  backend: string;
  color: string;
}

/** Build the real lineage chain for one entity (entity -> provenance -> observations). */
export function buildProvenanceNodes(entity: Entity | null): ProvenanceNode[] {
  if (!entity) return [];
  const nodes: ProvenanceNode[] = [];

  // Node 1: the entity itself, with its measured confidence.
  nodes.push({
    id: `entity-${entity.id}`,
    type: entity.type,
    title: entity.name,
    count: `${entity.childIds.length} children · ${entity.observationCount} observations`,
    detail: `provenance ${entity.provenance.state}`,
    confidence: entity.provenance.confidence,
    backend: entity.provenance.algorithm ?? 'unknown algorithm',
    color: '#3d8ef7',
  });

  // Node 2: provenance record (algorithm + coordinate frame + version).
  nodes.push({
    id: `prov-${entity.id}`,
    type: 'PROVENANCE',
    title: 'PROVENANCE RECORD',
    count: entity.provenance.state,
    detail: [
      entity.provenance.coordinateFrame ? `frame: ${entity.provenance.coordinateFrame}` : null,
      entity.provenance.version ? `version: ${entity.provenance.version}` : null,
      entity.provenance.timestamp ? `recorded: ${entity.provenance.timestamp}` : null,
    ]
      .filter(Boolean)
      .join(' · ') || 'no further provenance detail recorded',
    confidence: entity.provenance.confidence,
    backend: 'WorldIR provenance',
    color: '#5edaff',
  });

  // Node 3+: one node per contributing observation (bounded — an entity
  // can carry hundreds; the chain shows the first 8 and says so).
  const obs = (entity.metadata.observations as Array<{
    id: string;
    sensor_type: string;
    frame_id: string;
    confidence: number;
  }> | undefined) ?? [];
  const shown = obs.slice(0, 8);
  for (const o of shown) {
    nodes.push({
      id: `obs-${o.id}`,
      type: 'OBSERVATION',
      title: o.frame_id || o.id,
      count: o.sensor_type,
      detail: `observation ${o.id}`,
      confidence: o.confidence,
      backend: 'ReconstructionContract observation',
      color: '#c77df5',
    });
  }
  if (obs.length > shown.length) {
    nodes.push({
      id: 'obs-more',
      type: 'OBSERVATION',
      title: `+${obs.length - shown.length} more observations`,
      count: `${obs.length} total`,
      detail: 'bounded display; full list in the observation table below',
      confidence: entity.provenance.confidence,
      backend: 'ReconstructionContract observation',
      color: '#c77df5',
    });
  }
  return nodes;
}

interface ProvenanceChainProps {
  selectedNodeId: string;
  onSelectNode: (node: ProvenanceNode) => void;
}

export function ProvenanceChain({ selectedNodeId, onSelectNode }: ProvenanceChainProps) {
  const entities = useREStore((s) => s.entities);
  const selectedIds = useREStore((s) => s.selection.selectedEntityIds);
  const backendConnected = useREStore((s) => s.backendConnected);
  const backendError = useREStore((s) => s.backendError);

  const entity: Entity | null = selectedIds.length
    ? entities.get(selectedIds[0]) ?? null
    : null;
  const nodes = buildProvenanceNodes(entity);

  if (!backendConnected) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 bg-[#121215] text-[#e8e8f0] px-6 text-center">
        <CircleAlert className="w-6 h-6 text-[#f5a623]" />
        <div className="text-[11px] font-semibold uppercase tracking-wider">BACKEND UNAVAILABLE</div>
        <div className="text-[10px] font-mono text-[#9898b0] max-w-[280px]">
          {backendError ?? 'The Reality Engine backend is not reachable.'} No provenance can be shown because there is no data to trace — none is invented.
        </div>
      </div>
    );
  }

  if (!entity) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 bg-[#121215] text-[#e8e8f0] px-6 text-center">
        <Box className="w-6 h-6 text-[#3d3d55]" />
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[#9898b0]">NO ENTITY SELECTED</div>
        <div className="text-[10px] font-mono text-[#5c5c78] max-w-[280px]">
          Select an entity in the Outliner or Viewport to trace its provenance: entity → provenance record → contributing observations.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#121215] text-[#e8e8f0] select-none text-xs">
      {/* Header */}
      <div className="px-3 h-[36px] bg-[#17171c] border-b border-[#1e1e28] flex items-center justify-between">
        <span className="font-semibold text-[11px] uppercase tracking-wider text-[#9898b0] flex items-center gap-1.5">
          <GitCommit className="w-3.5 h-3.5 text-[#3d8ef7]" />
          Provenance Lineage
        </span>
        <span className="text-[10px] font-mono text-[#9898b0]">
          {nodes.length} nodes
        </span>
      </div>

      {/* Vertical chain */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center space-y-2">
        {nodes.map((node, idx) => {
          const isSelected = selectedNodeId === node.id;
          const isLast = idx === nodes.length - 1;

          return (
            <React.Fragment key={node.id}>
              <div
                onClick={() => onSelectNode(node)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') onSelectNode(node);
                }}
                aria-pressed={isSelected}
                className={`w-full max-w-[340px] p-2.5 rounded border transition-all cursor-pointer ${
                  isSelected
                    ? 'bg-[#1c1c23] border-[#3d8ef7] shadow-[0_0_12px_rgba(61,142,247,0.25)]'
                    : 'bg-[#17171c] border-[#272733] hover:border-[#3d8ef7]/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1.5">
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ background: node.color }}
                    />
                    <span className="text-[9px] font-mono uppercase tracking-wider text-[#5c5c78]">{node.type}</span>
                  </div>
                  <span className="text-[9px] font-mono text-[#9898b0]">{node.count}</span>
                </div>
                <div className="text-[11px] font-semibold truncate">{node.title}</div>
                <div className="text-[9px] font-mono text-[#5c5c78] truncate">{node.detail}</div>
                <div className="flex items-center justify-between mt-1.5">
                  <span className="text-[9px] font-mono text-[#9898b0] flex items-center gap-1">
                    <ShieldCheck className="w-3 h-3" /> {node.backend}
                  </span>
                  <span className="text-[9px] font-mono text-green-400">
                    {(node.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              {!isLast && <div className="w-px h-3 bg-[#3d3d55]" aria-hidden="true" />}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
