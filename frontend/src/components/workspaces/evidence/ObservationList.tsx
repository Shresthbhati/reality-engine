'use client';

/**
 * Contributing-observations table for the SELECTED entity, fed from
 * real backend state (ReconstructionContract observations carried on
 * each entity by the bridge).
 *
 * The previous version rendered a hardcoded MOCK_OBSERVATIONS list with
 * invented residuals ("0.42 px") and fake sessions. Now every row is a
 * real observation: frame id, sensor type, measured confidence. There
 * is no residual column because the backend does not currently expose a
 * per-observation residual — showing one would be fabrication; the
 * timestamp is the observation's own recorded value.
 *
 * Honesty rules:
 *  - No selection -> explicit empty state.
 *  - Entity with zero observations -> "no observations recorded",
 *    which is data, not an error.
 *  - Backend offline -> UNAVAILABLE, never demo rows.
 */

import React, { useState } from 'react';
import { ShieldCheck, Search, CircleAlert, Box } from 'lucide-react';
import { useREStore } from '@/store/re-store';
import type { Entity } from '@/types/reality-engine';

interface ObservationRow {
  id: string;
  type: string;
  sourceFile: string;
  timestamp: string;
  confidence: number;
}

function rowsFromEntity(entity: Entity): ObservationRow[] {
  const obs = (entity.metadata.observations as Array<{
    id: string;
    sensor_type: string;
    timestamp?: number;
    frame_id: string;
    confidence: number;
  }> | undefined) ?? [];
  return obs.map((o) => ({
    id: o.id,
    type: o.sensor_type,
    sourceFile: o.frame_id || '—',
    timestamp:
      typeof o.timestamp === 'number' && Number.isFinite(o.timestamp)
        ? new Date(o.timestamp * 1000).toISOString().replace('T', ' ').slice(0, 19)
        : '—',
    confidence: o.confidence,
  }));
}

export function ObservationList() {
  const [filter, setFilter] = useState('');
  const entities = useREStore((s) => s.entities);
  const selectedIds = useREStore((s) => s.selection.selectedEntityIds);
  const backendConnected = useREStore((s) => s.backendConnected);
  const backendError = useREStore((s) => s.backendError);

  const entity: Entity | null = selectedIds.length
    ? entities.get(selectedIds[0]) ?? null
    : null;

  if (!backendConnected) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 bg-[#121215] text-[#e8e8f0] px-6 text-center">
        <CircleAlert className="w-6 h-6 text-[#f5a623]" />
        <div className="text-[11px] font-semibold uppercase tracking-wider">BACKEND UNAVAILABLE</div>
        <div className="text-[10px] font-mono text-[#9898b0] max-w-[320px]">
          {backendError ?? 'The Reality Engine backend is not reachable.'} Observations come from real reconstruction runs; none can be listed right now.
        </div>
      </div>
    );
  }

  if (!entity) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 bg-[#121215] text-[#e8e8f0] px-6 text-center">
        <Box className="w-6 h-6 text-[#3d3d55]" />
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[#9898b0]">NO ENTITY SELECTED</div>
        <div className="text-[10px] font-mono text-[#5c5c78] max-w-[320px]">
          Select an entity to inspect the observations that support it.
        </div>
      </div>
    );
  }

  const allRows = rowsFromEntity(entity);
  const filtered = allRows.filter(
    (o) =>
      o.id.toLowerCase().includes(filter.toLowerCase()) ||
      o.sourceFile.toLowerCase().includes(filter.toLowerCase()) ||
      o.type.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="flex flex-col h-full bg-[#121215] text-[#e8e8f0] select-none text-xs">
      {/* Header with Search and Filter */}
      <div className="px-3 h-[32px] bg-[#17171c] border-b border-[#1e1e28] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-[10px] uppercase tracking-wider text-[#9898b0] flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-[#34c76f]" />
            Contributing Observations ({filtered.length}
            {filtered.length !== allRows.length ? ` of ${allRows.length}` : ''})
          </span>
          <span className="text-[9px] font-mono text-[#5c5c78] truncate max-w-[220px]">
            {entity.id}
          </span>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 px-2 py-0.5 bg-[#0d0d0f] border border-[#272733] rounded">
            <Search className="w-3 h-3 text-[#5c5c78]" />
            <input
              type="text"
              placeholder="Filter observations..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              aria-label="Filter observations"
              className="bg-transparent text-[10px] font-mono text-[#e8e8f0] focus:outline-none w-36"
            />
          </div>
        </div>
      </div>

      {allRows.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 px-6 text-center">
          <Box className="w-5 h-5 text-[#3d3d55]" />
          <div className="text-[10px] font-mono text-[#9898b0]">
            No observations recorded for this entity. The backend carries none — nothing is synthesized here.
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto">
          <table className="w-full border-collapse font-mono text-[11px] text-left">
            <thead>
              <tr className="bg-[#17171c] border-b border-[#1e1e28] text-[9px] text-[#5c5c78] uppercase tracking-wider">
                <th className="py-1.5 px-3">Observation ID</th>
                <th className="py-1.5 px-3">Type</th>
                <th className="py-1.5 px-3">Source Frame</th>
                <th className="py-1.5 px-3">Timestamp (UTC)</th>
                <th className="py-1.5 px-3">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((row) => (
                <tr
                  key={row.id}
                  className="border-b border-[#1e1e28]/50 hover:bg-[#1c1c23] transition-colors"
                >
                  <td className="py-1.5 px-3 text-[#3d8ef7] font-semibold">{row.id}</td>
                  <td className="py-1.5 px-3 text-[#9898b0]">{row.type}</td>
                  <td className="py-1.5 px-3 text-[#e8e8f0]">{row.sourceFile}</td>
                  <td className="py-1.5 px-3 text-[#5c5c78]">{row.timestamp}</td>
                  <td className="py-1.5 px-3 text-green-400">{Math.round(row.confidence * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
