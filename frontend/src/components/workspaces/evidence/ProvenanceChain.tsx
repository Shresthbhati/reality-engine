'use client';

import React, { useState } from 'react';
import { Camera, Layers, GitCommit, Crosshair, Box, ShieldCheck, Cpu } from 'lucide-react';

export interface ProvenanceNode {
  id: string;
  type: string;
  title: string;
  count: string;
  detail: string;
  confidence: number;
  uncertainty?: string;
  backend: string;
  color: string;
}

interface ProvenanceChainProps {
  selectedNodeId: string;
  onSelectNode: (node: ProvenanceNode) => void;
}

export const MOCK_PROVENANCE_NODES: ProvenanceNode[] = [
  {
    id: 'node-images',
    type: 'IMAGE',
    title: 'RAW IMAGERY SENSORS',
    count: '14 Frames',
    detail: 'DJI FC3582 (12) + iPhone 15 Pro (2)',
    confidence: 1.0,
    backend: 'EXIF / GNSS Log',
    color: '#3d8ef7',
  },
  {
    id: 'node-features',
    type: 'FEATURE',
    title: 'KEYPOINT DETECTION',
    count: '84,120 Points',
    detail: 'DoG Scale-Space Extrema & SuperPoint',
    confidence: 0.98,
    backend: 'OpenCV / PyTorch',
    color: '#5edaff',
  },
  {
    id: 'node-matches',
    type: 'MATCH',
    title: 'TWO-VIEW CORRESPONDENCE',
    count: '61,400 Matches',
    detail: 'Mutual nearest neighbor with epipolar filter',
    confidence: 0.95,
    backend: 'LightGlue / COLMAP',
    color: '#c77df5',
  },
  {
    id: 'node-poses',
    type: 'CAMERA_POSE',
    title: 'BUNDLE ADJUSTMENT',
    count: '14 Extrinsics',
    detail: 'SE(3) poses registered in Local ENU',
    confidence: 0.96,
    uncertainty: '± 1.2 cm',
    backend: 'COLMAP Ceres Solver',
    color: '#34c76f',
  },
  {
    id: 'node-depth',
    type: 'DEPTH',
    title: 'MULTI-VIEW STEREO',
    count: '3 Dense Sweeps',
    detail: 'PatchMatch stereo depth estimation',
    confidence: 0.92,
    uncertainty: '± 2.1 cm',
    backend: 'OpenMVS Depth Estimator',
    color: '#f5a623',
  },
  {
    id: 'node-points',
    type: 'POINT',
    title: 'FUSED POINT CLOUD',
    count: '294,345 Points',
    detail: 'Outlier rejection via visibility voting',
    confidence: 0.94,
    uncertainty: '± 1.8 cm',
    backend: 'Open3D Voxel Grid',
    color: '#5edaff',
  },
  {
    id: 'node-entity',
    type: 'ENTITY',
    title: 'WALL_042 (Building A)',
    count: 'WorldIR Entity',
    detail: 'Planar segmentation + Semantic classification',
    confidence: 0.94,
    uncertainty: '± 2.7 cm',
    backend: 'RTE Perception / SAM2',
    color: '#3d8ef7',
  },
];

export function ProvenanceChain({ selectedNodeId, onSelectNode }: ProvenanceChainProps) {
  return (
    <div className="flex flex-col h-full bg-[#121215] text-[#e8e8f0] select-none text-xs">
      {/* Header */}
      <div className="px-3 h-[36px] bg-[#17171c] border-b border-[#1e1e28] flex items-center justify-between">
        <span className="font-semibold text-[11px] uppercase tracking-wider text-[#9898b0] flex items-center gap-1.5">
          <GitCommit className="w-3.5 h-3.5 text-[#3d8ef7]" />
          Provenance Lineage DAG
        </span>
        <span className="text-[10px] font-mono text-[#34c76f] bg-green-500/10 px-1.5 py-0.5 rounded border border-green-500/20">
          Trace Depth: 7
        </span>
      </div>

      {/* Vertical DAG Chain */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center space-y-2">
        {MOCK_PROVENANCE_NODES.map((node, idx) => {
          const isSelected = selectedNodeId === node.id;
          const isLast = idx === MOCK_PROVENANCE_NODES.length - 1;

          return (
            <React.Fragment key={node.id}>
              <div
                onClick={() => onSelectNode(node)}
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
                    <span className="text-[10px] font-mono text-[#9898b0] uppercase">
                      {node.type}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#3d8ef7] font-semibold">
                    {node.count}
                  </span>
                </div>

                <div className="text-[12px] font-medium text-[#e8e8f0] mb-0.5">{node.title}</div>
                <div className="text-[10px] text-[#5c5c78] truncate mb-2">{node.detail}</div>

                <div className="flex items-center justify-between text-[9px] font-mono pt-1.5 border-t border-[#272733]">
                  <span className="text-[#9898b0]">Backend: {node.backend}</span>
                  <span className="text-green-400 font-semibold">
                    {Math.round(node.confidence * 100)}% Conf
                  </span>
                </div>
              </div>

              {!isLast && (
                <div className="flex flex-col items-center my-0.5">
                  <div className="w-[2px] h-3 bg-[#272733]" />
                  <div className="w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[4px] border-t-[#3d8ef7]" />
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
