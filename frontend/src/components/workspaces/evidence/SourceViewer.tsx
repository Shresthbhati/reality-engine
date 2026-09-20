'use client';

/**
 * Evidence verification viewer for the selected provenance node.
 *
 * Shows what the backend actually holds about the selected observation
 * / provenance node: its recorded identity, sensor type, confidence.
 * The previous version simulated a camera keyframe with invented EXIF
 * ("1\" CMOS 20MP • ISO 100 • 1/800s • f/2.8") and a fabricated
 * "Cryptographically Signed SHA-256" badge. Both are gone: when the
 * backend exposes image artifacts (data_uri on observations) they can
 * be rendered; until then this panel states exactly what is and is not
 * available rather than drawing a plausible-looking fake.
 */

import React from 'react';
import { Camera, ShieldCheck, CircleAlert, Box } from 'lucide-react';
import type { ProvenanceNode } from './ProvenanceChain';
import type { Entity } from '@/types/reality-engine';

interface SourceViewerProps {
  node: ProvenanceNode | null;
  entity: Entity | null;
}

export function SourceViewer({ node, entity }: SourceViewerProps) {
  if (!node) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 bg-[#0d0d0f] text-[#e8e8f0] px-6 text-center">
        <Box className="w-6 h-6 text-[#3d3d55]" />
        <div className="text-[11px] font-semibold uppercase tracking-wider text-[#9898b0]">
          {entity ? 'NO NODE SELECTED' : 'NO ENTITY SELECTED'}
        </div>
        <div className="text-[10px] font-mono text-[#5c5c78] max-w-[320px]">
          Select a node in the lineage chain to inspect the evidence behind it.
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-[#0d0d0f] text-[#e8e8f0] select-none">
      {/* Header */}
      <div className="px-4 h-[36px] bg-[#121215] border-b border-[#1e1e28] flex items-center justify-between text-xs">
        <div className="flex items-center gap-2 min-w-0">
          <Camera className="w-4 h-4 text-[#3d8ef7] shrink-0" />
          <span className="font-semibold text-[11px] uppercase tracking-wider text-[#e8e8f0] truncate">
            Evidence Inspector — [{node.type}] {node.title}
          </span>
        </div>
        <span className="text-[10px] font-mono text-[#9898b0] shrink-0 flex items-center gap-1">
          <ShieldCheck className="w-3 h-3" /> {node.backend}
        </span>
      </div>

      {/* Measured record of the selected node */}
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded border border-[#272733] bg-[#121215] p-4">
            <div className="text-[9px] uppercase tracking-wider text-[#5c5c78] mb-2">Record</div>
            <dl className="grid grid-cols-[110px_1fr] gap-y-1.5 font-mono text-[11px]">
              <dt className="text-[#5c5c78]">id</dt>
              <dd className="text-[#e8e8f0] truncate">{node.id}</dd>
              <dt className="text-[#5c5c78]">type</dt>
              <dd className="text-[#9898b0]">{node.type}</dd>
              <dt className="text-[#5c5c78]">title</dt>
              <dd className="text-[#e8e8f0] truncate">{node.title}</dd>
              <dt className="text-[#5c5c78]">count</dt>
              <dd className="text-[#9898b0]">{node.count}</dd>
              <dt className="text-[#5c5c78]">confidence</dt>
              <dd className="text-green-400">{(node.confidence * 100).toFixed(1)}%</dd>
              <dt className="text-[#5c5c78]">source</dt>
              <dd className="text-[#9898b0] truncate">{node.backend}</dd>
            </dl>
          </div>

          <div className="rounded border border-[#272733] bg-[#121215] p-4">
            <div className="text-[9px] uppercase tracking-wider text-[#5c5c78] mb-2">Source image</div>
            <div className="flex flex-col items-center justify-center gap-2 h-[120px] border border-dashed border-[#272733] rounded">
              <CircleAlert className="w-5 h-5 text-[#f5a623]" />
              <div className="text-[10px] font-mono text-[#9898b0] text-center px-4">
                IMAGE ARTIFACT: UNAVAILABLE
                <br />
                <span className="text-[#5c5c78]">
                  The backend does not currently expose source imagery for
                  observations. When it does (observation data_uri), the
                  frame renders here — no simulated keyframe is shown.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
