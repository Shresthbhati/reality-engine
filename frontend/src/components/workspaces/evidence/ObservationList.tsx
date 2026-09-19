'use client';

import React, { useState } from 'react';
import { ShieldCheck, ArrowUpDown, Filter, Search } from 'lucide-react';

interface ObservationRow {
  id: string;
  type: string;
  sessionId: string;
  sourceFile: string;
  timestamp: string;
  confidence: number;
  residual: string;
  backend: string;
}

const MOCK_OBSERVATIONS: ObservationRow[] = [
  { id: 'obs-0841', type: 'KEYPOINT_RAY', sessionId: 'sess-001', sourceFile: 'DJI_0491.JPG', timestamp: '08:34:12.100', confidence: 0.98, residual: '0.42 px', backend: 'COLMAP' },
  { id: 'obs-0842', type: 'KEYPOINT_RAY', sessionId: 'sess-001', sourceFile: 'DJI_0492.JPG', timestamp: '08:34:14.300', confidence: 0.97, residual: '0.48 px', backend: 'COLMAP' },
  { id: 'obs-0843', type: 'KEYPOINT_RAY', sessionId: 'sess-001', sourceFile: 'DJI_0493.JPG', timestamp: '08:34:16.500', confidence: 0.94, residual: '0.62 px', backend: 'COLMAP' },
  { id: 'obs-0844', type: 'DEPTH_VOXEL', sessionId: 'sess-001', sourceFile: 'OpenMVS_Sweep01', timestamp: '08:35:00.000', confidence: 0.91, residual: '0.012 m', backend: 'OpenMVS' },
  { id: 'obs-0845', type: 'KEYPOINT_RAY', sessionId: 'sess-002', sourceFile: 'IMG_8120.HEIC', timestamp: '08:41:02.120', confidence: 0.89, residual: '1.10 px', backend: 'COLMAP' },
  { id: 'obs-0846', type: 'KEYPOINT_RAY', sessionId: 'sess-002', sourceFile: 'IMG_8121.HEIC', timestamp: '08:41:04.400', confidence: 0.86, residual: '1.24 px', backend: 'COLMAP' },
  { id: 'obs-0847', type: 'LIDAR_POINT', sessionId: 'sess-003', sourceFile: 'Scan_Sector4.las', timestamp: '09:10:22.000', confidence: 0.99, residual: '0.004 m', backend: 'RTE LiDAR' },
  { id: 'obs-0848', type: 'SEMANTIC_MASK', sessionId: 'sess-001', sourceFile: 'SAM2_Mask_042', timestamp: '09:12:00.000', confidence: 0.95, residual: '0.92 IoU', backend: 'SAM2' },
];

export function ObservationList() {
  const [filter, setFilter] = useState('');

  const filtered = MOCK_OBSERVATIONS.filter((o) => 
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
            Contributing Observations ({MOCK_OBSERVATIONS.length})
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
              className="bg-transparent text-[10px] font-mono text-[#e8e8f0] focus:outline-none w-36"
            />
          </div>
          <span className="text-[9px] font-mono text-[#5c5c78]">
            {"// WIRE: GET /api/entities/{id}/observations"}
          </span>
        </div>
      </div>

      {/* Observation Table */}
      <div className="flex-1 overflow-y-auto">
        <table className="w-full border-collapse font-mono text-[11px] text-left">
          <thead>
            <tr className="bg-[#17171c] border-b border-[#1e1e28] text-[9px] text-[#5c5c78] uppercase tracking-wider">
              <th className="py-1.5 px-3">Observation ID</th>
              <th className="py-1.5 px-3">Type</th>
              <th className="py-1.5 px-3">Source Artifact</th>
              <th className="py-1.5 px-3">Timestamp (UTC)</th>
              <th className="py-1.5 px-3">Confidence</th>
              <th className="py-1.5 px-3">Residual / Error</th>
              <th className="py-1.5 px-3">Solver</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => (
              <tr 
                key={row.id} 
                className="border-b border-[#1e1e28]/50 hover:bg-[#1c1c23] transition-colors cursor-pointer"
              >
                <td className="py-1.5 px-3 text-[#3d8ef7] font-semibold">{row.id}</td>
                <td className="py-1.5 px-3 text-[#9898b0]">{row.type}</td>
                <td className="py-1.5 px-3 text-[#e8e8f0]">{row.sourceFile}</td>
                <td className="py-1.5 px-3 text-[#5c5c78]">{row.timestamp}</td>
                <td className="py-1.5 px-3 text-green-400">{Math.round(row.confidence * 100)}%</td>
                <td className="py-1.5 px-3 text-[#f5a623]">{row.residual}</td>
                <td className="py-1.5 px-3 text-[#9898b0]">{row.backend}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
