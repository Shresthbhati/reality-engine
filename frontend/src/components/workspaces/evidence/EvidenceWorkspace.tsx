'use client';

import React, { useState } from 'react';
import { Group, Panel, Separator } from 'react-resizable-panels';
import { ProvenanceChain, MOCK_PROVENANCE_NODES, type ProvenanceNode } from './ProvenanceChain';
import { SourceViewer } from './SourceViewer';
import { ObservationList } from './ObservationList';

export function EvidenceWorkspace() {
  const [selectedNode, setSelectedNode] = useState<ProvenanceNode>(MOCK_PROVENANCE_NODES[0]);

  return (
    <div className="flex flex-col w-full h-full overflow-hidden bg-[#0d0d0f] text-[#e8e8f0]">
      {/* ── Top Header Context Bar ───────────────────────────────── */}
      <div className="flex items-center justify-between px-4 h-[38px] bg-[#121215] border-b border-[#1e1e28] select-none">
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-semibold tracking-wider text-[#e8e8f0] uppercase flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[#34c76f]" />
            EVIDENCE INVESTIGATION CONSOLE
          </span>
          <span className="text-[10px] text-[#9898b0] font-mono px-2 py-0.5 bg-[#17171c] rounded border border-[#272733]">
            Target Entity: ent-building-a / Wall_042
          </span>
          <span className="text-[10px] text-green-400 font-mono bg-green-500/10 px-2 py-0.5 rounded border border-green-500/20">
            Overall Confidence: 94.2%
          </span>
          <span className="text-[10px] text-[#f5a623] font-mono">
            Spatial Uncertainty: ± 2.7 cm
          </span>
        </div>

        <div className="text-[10px] text-[#5c5c78] font-mono">
          Answering: &quot;Why does Reality Engine believe this wall exists?&quot;
        </div>
      </div>

      {/* ── Main Resizable Layout: DAG & Viewer on top, Observation Table on bottom ── */}
      <div className="flex-1 min-h-0 w-full">
        <Group orientation="vertical">
          {/* Top Panel: Lineage DAG (left) + Verification Viewer (right) */}
          <Panel defaultSize={68} minSize={40}>
            <Group orientation="horizontal">
              {/* Left: Lineage DAG */}
              <Panel defaultSize={32} minSize={24} maxSize={45}>
                <div className="h-full w-full bg-[#121215] border-r border-[#1e1e28] overflow-hidden">
                  <ProvenanceChain
                    selectedNodeId={selectedNode.id}
                    onSelectNode={setSelectedNode}
                  />
                </div>
              </Panel>

              <Separator className="w-[3px] bg-[#1e1e28] hover:bg-[#3d8ef7] transition-colors cursor-col-resize active:bg-[#3d8ef7]" />

              {/* Right: Source Sensor Keyframe & Extrinsics Inspector */}
              <Panel defaultSize={68} minSize={40}>
                <div className="h-full w-full bg-[#0d0d0f] overflow-hidden">
                  <SourceViewer node={selectedNode} />
                </div>
              </Panel>
            </Group>
          </Panel>

          <Separator className="h-[3px] bg-[#1e1e28] hover:bg-[#3d8ef7] transition-colors cursor-row-resize active:bg-[#3d8ef7]" />

          {/* Bottom Panel: Contributing Observations Table */}
          <Panel defaultSize={32} minSize={18} maxSize={50}>
            <div className="h-full w-full bg-[#121215] overflow-hidden">
              <ObservationList />
            </div>
          </Panel>
        </Group>
      </div>
    </div>
  );
}

export default EvidenceWorkspace;
