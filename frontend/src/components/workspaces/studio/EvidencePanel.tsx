'use client';

import React, { useState } from 'react';
import { useREStore } from '@/store/re-store';
import {
  ChevronUp,
  ChevronDown,
  ExternalLink,
  ShieldCheck,
  Camera,
  Layers,
  Cpu,
  Compass,
  Zap,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Terminal,
  Activity,
  HardDrive,
  RefreshCw,
} from 'lucide-react';

export default function EvidencePanel() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const {
    entities,
    selection,
    builds,
    computeMetrics,
    logs,
    studioBottomTab,
    setStudioBottomTab,
    toggleBottomDrawer,
    setActiveWorkspace,
  } = useREStore();

  const selectedId = selection.selectedEntityIds[0] || 'ent-structure-main';
  const entity = entities.get(selectedId) || entities.get('ent-structure-main');
  const currentBuild = builds[0];

  const provenanceSteps = [
    { name: 'RAW SENSORS', count: '7,661 frames', sub: 'Drone + Phone', icon: Camera, color: '#3d8ef7' },
    { name: 'TIME & CALIB', count: '1.8 ms drift', sub: 'PTP & Brown-Conrady', icon: Clock, color: '#5edaff' },
    { name: 'FEATURES', count: '684,200 pts', sub: 'SIFT / SuperPoint', icon: Layers, color: '#5edaff' },
    { name: 'MATCHES', count: '412,000 inliers', sub: 'LightGlue epipolar', icon: Compass, color: '#c77df5' },
    { name: 'CAMERA POSES', count: '7,590 registered', sub: 'COLMAP bundle adj', icon: Cpu, color: '#2ecc71' },
    { name: 'DENSE POINT CLOUD', count: '4,360,000 pts', sub: 'OpenMVS PatchMatch', icon: Layers, color: '#f39c12' },
    { name: 'VOLUMETRIC FUSION', count: '2,150,000 tri', sub: 'Signed Distance Field', icon: Layers, color: '#f39c12' },
    { name: '3D PERCEPTION', count: '184 segments', sub: 'SAM2 3D Masks', icon: ShieldCheck, color: '#2ecc71' },
    { name: entity?.name || 'VICTORIA MEMORIAL', count: 'WorldIR Committed', sub: 'Persistent Graph', icon: ShieldCheck, color: '#3d8ef7', isTarget: true },
  ];

  const getHeaderTitle = () => {
    switch (studioBottomTab) {
      case 'pipeline':
        return 'RECONSTRUCTION PIPELINE — 10 STAGES (100% COMPLETE)';
      case 'evidence':
        return `EVIDENCE LINEAGE — TARGET: ${(entity?.name ?? 'VICTORIA MEMORIAL').toUpperCase()}`;
      case 'jobs':
        return 'ACTIVE COMPUTE JOBS (3 BACKGROUND PROCESSES)';
      case 'diagnostics':
        return 'GPU COMPUTE LOAD & VRAM ALLOCATION';
      case 'console':
        return 'ENGINE REAL-TIME EVENT STREAM';
      default:
        return 'DOCK PANEL';
    }
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#0c0d11] select-none">
      {/* ── Streamlined Contextual Action Header (28px) ─────────────── */}
      <div className="flex items-center justify-between px-3 h-7 bg-[#0f1014] border-b border-[#1f222b] shrink-0 text-xs font-mono">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#3d8ef7]" />
          <span className="text-[10px] font-bold text-[#f0f1f6] tracking-wider">
            {getHeaderTitle()}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {studioBottomTab === 'pipeline' && (
            <button
              type="button"
              onClick={() => setActiveWorkspace('build')}
              className="flex items-center gap-1 text-[10px] font-mono text-[#3d8ef7] hover:text-[#5ca2f9] bg-[#3d8ef7]/10 hover:bg-[#3d8ef7]/20 px-2 py-0.5 rounded border border-[#3d8ef7]/30 transition-colors"
            >
              <span>Open Build DAG</span>
              <ExternalLink className="w-2.5 h-2.5" />
            </button>
          )}

          {studioBottomTab === 'evidence' && (
            <button
              type="button"
              onClick={() => setActiveWorkspace('evidence')}
              className="flex items-center gap-1 text-[10px] font-mono text-[#2ecc71] hover:text-[#52e891] bg-[#2ecc71]/10 hover:bg-[#2ecc71]/20 px-2 py-0.5 rounded border border-[#2ecc71]/30 transition-colors"
            >
              <span>Evidence Console</span>
              <ExternalLink className="w-2.5 h-2.5" />
            </button>
          )}

          <button
            type="button"
            onClick={() => setActiveWorkspace('loader')}
            className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6] hover:text-[#ededf2] px-2 py-0.5 rounded border border-[#222633] transition-colors"
          >
            <span>Loader Ingest</span>
            <ExternalLink className="w-2.5 h-2.5" />
          </button>

          <button
            type="button"
            onClick={() => toggleBottomDrawer()}
            title="Close Drawer (⌘J)"
            className="text-[#9296a6] hover:text-[#ededf2] p-0.5"
          >
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Content Viewport ────────────────────────────────────────── */}
      <div className="flex-1 p-3 overflow-y-auto bg-[#090a0d]">
          {/* Tab 1: Reconstruction Pipeline Stages */}
          {studioBottomTab === 'pipeline' && currentBuild && (
            <div className="flex items-center gap-2 overflow-x-auto pb-2 h-full">
              {currentBuild.stages.map((stg, idx) => (
                <div
                  key={stg.stage}
                  className="min-w-[145px] h-full p-2.5 rounded-lg bg-[#12141a] border border-[#1f222b] flex flex-col justify-between font-mono shrink-0"
                >
                  <div>
                    <div className="flex items-center justify-between text-[9px] text-[#54596b] uppercase mb-1">
                      <span>Stage 0{idx + 1}</span>
                      <span className="text-[#2ecc71] font-semibold">100%</span>
                    </div>
                    <span className="text-xs font-bold text-[#f0f1f6] block truncate">
                      {stg.stage}
                    </span>
                    <span className="text-[10px] text-[#3d8ef7] block truncate mt-0.5">
                      {stg.backend}
                    </span>
                  </div>

                  <div className="pt-2 border-t border-[#1a1d24] text-[9px] text-[#9296a6]">
                    <div className="flex justify-between">
                      <span>DURATION</span>
                      <span className="text-[#ededf2]">{((stg.durationMs ?? 0) / 1000).toFixed(1)}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span>GPU LOAD</span>
                      <span className="text-[#ededf2]">{stg.gpuUsage}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Tab 2: Evidence Traceability */}
          {studioBottomTab === 'evidence' && (
            <div className="flex items-center gap-2 overflow-x-auto pb-2 h-full">
              {provenanceSteps.map((step, idx) => {
                const Icon = step.icon;
                return (
                  <React.Fragment key={step.name}>
                    <div
                      className={`min-w-[135px] h-full p-2.5 rounded-lg border font-mono flex flex-col justify-between shrink-0 ${
                        step.isTarget
                          ? 'bg-[#3d8ef7]/10 border-[#3d8ef7] shadow-lg shadow-blue-500/10'
                          : 'bg-[#12141a] border-[#1f222b]'
                      }`}
                    >
                      <div>
                        <div className="flex items-center justify-between text-[9px] text-[#54596b] uppercase mb-1">
                          <span>Step 0{idx + 1}</span>
                          <Icon className="w-3.5 h-3.5" style={{ color: step.color }} />
                        </div>
                        <span className="text-xs font-bold text-[#f0f1f6] block truncate">
                          {step.name}
                        </span>
                        <span className="text-[10px] text-[#3d8ef7] block mt-0.5">
                          {step.count}
                        </span>
                      </div>
                      <span className="text-[9px] text-[#54596b] truncate pt-1 border-t border-[#1a1d24]">
                        {step.sub}
                      </span>
                    </div>
                    {idx < provenanceSteps.length - 1 && (
                      <span className="text-[#2d323f] font-bold text-xs shrink-0">→</span>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          )}

          {/* Tab 3: Active Jobs Queue */}
          {studioBottomTab === 'jobs' && (
            <div className="space-y-2 font-mono text-xs">
              <div className="p-2.5 rounded-lg bg-[#12141a] border border-[#1f222b] flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-xs font-bold text-[#f0f1f6]">
                      JOB-COLMAP-04: Multi-View Epipolar Matcher
                    </span>
                    <span className="px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-400 text-[9px]">
                      COMPLETE
                    </span>
                  </div>
                  <span className="text-[10px] text-[#9296a6]">
                    Processed 7,661 camera stations across 2 sessions • 0.48 px residual error
                  </span>
                </div>
                <span className="text-xs text-[#2ecc71] font-bold">100%</span>
              </div>

              <div className="p-2.5 rounded-lg bg-[#12141a] border border-[#1f222b] flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-xs font-bold text-[#f0f1f6]">
                      JOB-SAM2-08: High-Resolution 3D Segment Anything
                    </span>
                    <span className="px-1.5 py-0.2 rounded bg-blue-500/15 text-blue-400 text-[9px] flex items-center gap-1">
                      <RefreshCw className="w-2 h-2 animate-spin" /> RUNNING
                    </span>
                  </div>
                  <span className="text-[10px] text-[#9296a6]">
                    Semantic prompt clustering on dome rotunda & marble colonnade
                  </span>
                </div>
                <span className="text-xs text-[#3d8ef7] font-bold">78%</span>
              </div>
            </div>
          )}

          {/* Tab 4: Diagnostics & Hardware */}
          {studioBottomTab === 'diagnostics' && (
            <div className="grid grid-cols-4 gap-3 font-mono text-xs">
              <div className="p-3 rounded-lg bg-[#12141a] border border-[#1f222b]">
                <span className="text-[10px] text-[#54596b] uppercase block mb-1">
                  GPU VRAM (NVIDIA CUDA)
                </span>
                <span className="text-lg font-bold text-[#3d8ef7] block num-tabular">
                  {computeMetrics.gpuVramUsed} / {computeMetrics.gpuVramTotal} MB
                </span>
                <span className="text-[10px] text-[#9296a6]">Usage: {computeMetrics.gpuUsage}%</span>
              </div>

              <div className="p-3 rounded-lg bg-[#12141a] border border-[#1f222b]">
                <span className="text-[10px] text-[#54596b] uppercase block mb-1">
                  System Memory (RAM)
                </span>
                <span className="text-lg font-bold text-[#2ecc71] block num-tabular">
                  {(computeMetrics.ramUsed / 1024).toFixed(1)} / {(computeMetrics.ramTotal / 1024).toFixed(1)} GB
                </span>
                <span className="text-[10px] text-[#9296a6]">Allocated to Sparse Cache</span>
              </div>

              <div className="p-3 rounded-lg bg-[#12141a] border border-[#1f222b]">
                <span className="text-[10px] text-[#54596b] uppercase block mb-1">
                  Host CPU Utilization
                </span>
                <span className="text-lg font-bold text-[#f0f1f6] block num-tabular">
                  {computeMetrics.cpuUsage}%
                </span>
                <span className="text-[10px] text-[#9296a6]">16 Worker Threads Active</span>
              </div>

              <div className="p-3 rounded-lg bg-[#12141a] border border-[#1f222b]">
                <span className="text-[10px] text-[#54596b] uppercase block mb-1">
                  Ray-Tracing Accelerators
                </span>
                <span className="text-lg font-bold text-emerald-400 block">
                  RTX OptiX Active
                </span>
                <span className="text-[10px] text-[#9296a6]">BVH Spatial Indexing</span>
              </div>
            </div>
          )}

          {/* Tab 5: Engine Console */}
          {studioBottomTab === 'console' && (
            <div className="space-y-1 font-mono text-[11px]">
              {logs.map((log) => (
                <div key={log.id} className="flex items-center gap-2 py-0.5 text-[#ededf2]">
                  <span className="text-[#54596b] num-tabular">{log.timestamp}</span>
                  <span
                    className={`px-1 rounded text-[9px] font-bold ${
                      log.level === 'WARNING'
                        ? 'bg-amber-500/20 text-amber-400'
                        : 'bg-blue-500/20 text-blue-400'
                    }`}
                  >
                    {log.level}
                  </span>
                  <span className="text-[#3d8ef7] font-semibold">[{log.module}]</span>
                  <span>{log.message}</span>
                </div>
              ))}
            </div>
          )}
        </div>
    </div>
  );
}
