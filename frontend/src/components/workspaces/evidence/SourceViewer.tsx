'use client';

import React, { useState } from 'react';
import { Camera, Maximize2, ShieldCheck, Crosshair, Cpu, Database, Eye } from 'lucide-react';
import type { ProvenanceNode } from './ProvenanceChain';

interface SourceViewerProps {
  node: ProvenanceNode;
}

export function SourceViewer({ node }: SourceViewerProps) {
  const [showKeypoints, setShowKeypoints] = useState(true);

  return (
    <div className="flex flex-col h-full bg-[#0d0d0f] text-[#e8e8f0] select-none">
      {/* Header */}
      <div className="px-4 h-[36px] bg-[#121215] border-b border-[#1e1e28] flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-[#3d8ef7]" />
          <span className="font-semibold text-[11px] uppercase tracking-wider text-[#e8e8f0]">
            Evidence Verification Inspector — [{node.type}]
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={() => setShowKeypoints(!showKeypoints)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors ${
              showKeypoints 
                ? 'bg-[#3d8ef7]/20 border-[#3d8ef7] text-[#3d8ef7]' 
                : 'bg-[#17171c] border-[#272733] text-[#9898b0]'
            }`}
          >
            {showKeypoints ? 'Keypoints: ON' : 'Keypoints: OFF'}
          </button>
          <span className="text-[10px] font-mono text-green-400 bg-green-500/10 px-2 py-0.5 rounded border border-green-500/20">
            Cryptographically Signed SHA-256
          </span>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-3 gap-4 overflow-y-auto">
        {/* Left 2 Cols: Sensor Image / Keypoint Canvas View */}
        <div className="lg:col-span-2 flex flex-col gap-3">
          <div className="relative aspect-video bg-[#121215] rounded border border-[#272733] overflow-hidden flex items-center justify-center group">
            {/* Grid background representing sensor plane */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#181822_1px,transparent_1px),linear-gradient(to_bottom,#181822_1px,transparent_1px)] bg-[size:24px_24px]" />

            {/* Fake camera image simulation */}
            <div className="relative z-10 flex flex-col items-center gap-2 text-center p-6">
              <Camera className="w-12 h-12 text-[#34344a] group-hover:text-[#3d8ef7] transition-colors" />
              <div className="text-sm font-semibold font-mono text-[#e8e8f0]">
                KEYFRAME_DJI_0492.DNG
              </div>
              <div className="text-xs text-[#5c5c78] font-mono">
                Sensor: 1&quot; CMOS 20MP (5472 x 3648) • ISO 100 • 1/800s • f/2.8
              </div>
            </div>

            {/* Simulated Feature Keypoints Overlay */}
            {showKeypoints && (
              <div className="absolute inset-0 pointer-events-none">
                {[
                  { x: 30, y: 40 }, { x: 35, y: 42 }, { x: 48, y: 55 }, 
                  { x: 62, y: 35 }, { x: 70, y: 60 }, { x: 25, y: 70 },
                  { x: 55, y: 72 }, { x: 42, y: 28 }, { x: 68, y: 22 }
                ].map((pt, i) => (
                  <div
                    key={i}
                    className="absolute w-2 h-2 rounded-full border border-[#3d8ef7] bg-[#5edaff]/40 -translate-x-1/2 -translate-y-1/2 shadow-[0_0_6px_#3d8ef7]"
                    style={{ left: `${pt.x}%`, top: `${pt.y}%` }}
                  />
                ))}
              </div>
            )}

            {/* In-view badges */}
            <div className="absolute top-2 left-2 bg-[#0d0d0f]/80 backdrop-blur-sm px-2 py-1 rounded text-[10px] font-mono text-[#9898b0] border border-[#272733]">
              Timestamp: 2026-09-14 08:34:12.822 UTC
            </div>
            <div className="absolute bottom-2 right-2 bg-[#0d0d0f]/80 backdrop-blur-sm px-2 py-1 rounded text-[10px] font-mono text-[#34c76f] border border-[#272733]">
              Reprojection Error: 0.48 px
            </div>
          </div>

          {/* Epipolar Ray Convergence Metric */}
          <div className="bg-[#121215] p-3 rounded border border-[#272733] flex items-center justify-between text-xs font-mono">
            <div>
              <span className="text-[#5c5c78]">RAY INTERSECTION ANGLE:</span>
              <span className="text-[#3d8ef7] ml-2 font-semibold">38.4° (Optimal Baseline)</span>
            </div>
            <div>
              <span className="text-[#5c5c78]">TRIANGULATION RESIDUAL:</span>
              <span className="text-green-400 ml-2 font-semibold">0.014 m</span>
            </div>
          </div>
        </div>

        {/* Right Col: Sensor Metadata & Transformation Matrix */}
        <div className="flex flex-col gap-3">
          {/* Node specifics */}
          <div className="bg-[#121215] p-3 rounded border border-[#272733] space-y-2">
            <div className="text-[10px] text-[#5c5c78] font-mono uppercase tracking-wider">
              Selected Evidence Node
            </div>
            <div className="text-sm font-semibold text-[#e8e8f0]">{node.title}</div>
            <div className="text-xs text-[#9898b0]">{node.detail}</div>

            <div className="pt-2 border-t border-[#1e1e28] space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-[#5c5c78]">Algorithm:</span>
                <span className="text-[#e8e8f0]">{node.backend}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#5c5c78]">Confidence:</span>
                <span className="text-green-400">{Math.round(node.confidence * 100)}%</span>
              </div>
              {node.uncertainty && (
                <div className="flex justify-between">
                  <span className="text-[#5c5c78]">Uncertainty:</span>
                  <span className="text-[#f5a623]">{node.uncertainty}</span>
                </div>
              )}
            </div>
          </div>

          {/* SE(3) Extrinsics Pose Matrix */}
          <div className="bg-[#121215] p-3 rounded border border-[#272733] space-y-2">
            <div className="text-[10px] text-[#5c5c78] font-mono uppercase tracking-wider flex items-center justify-between">
              <span>Extrinsic Pose Matrix [R | t]</span>
              <span className="text-[#3d8ef7]">SE(3)</span>
            </div>

            <div className="bg-[#0d0d0f] p-2 rounded border border-[#1e1e28] font-mono text-[10px] text-[#9898b0] space-y-0.5">
              <div>[  0.9842, -0.0412,  0.1721,  142.82 ]</div>
              <div>[  0.0381,  0.9991,  0.0210,   84.15 ]</div>
              <div>[ -0.1728, -0.0141,  0.9848,   34.50 ]</div>
              <div>[  0.0000,  0.0000,  0.0000,    1.00 ]</div>
            </div>
          </div>

          {/* Geospatial Fix */}
          <div className="bg-[#121215] p-3 rounded border border-[#272733] space-y-1 text-[11px] font-mono">
            <div className="text-[10px] text-[#5c5c78] uppercase">RTK GNSS Georeference</div>
            <div className="text-[#e8e8f0]">Lat: 22.572646° N</div>
            <div className="text-[#e8e8f0]">Lon: 88.363892° E</div>
            <div className="text-[#3d8ef7]">Ellipsoidal Height: 12.4m ± 0.018m</div>
          </div>

          <div className="text-[9px] text-[#5c5c78] font-mono italic">
            {"// WIRE: GET /api/evidence/{nodeId}/source"}
          </div>
        </div>
      </div>
    </div>
  );
}
