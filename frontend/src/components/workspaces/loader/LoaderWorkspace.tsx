'use client';

import React, { useState, useEffect, DragEvent } from 'react';
import {
  FolderOpen,
  Camera,
  Smartphone,
  Radio,
  Clock,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  ChevronRight,
  Search,
  Filter,
  ArrowDownToLine,
  Sliders,
  Play,
  RotateCcw,
  SlidersHorizontal,
  Table as TableIcon,
  List,
  Layers,
  FileCheck,
  Check,
  Zap,
  Info,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import { useREStore, Session, SessionSourceType, ValidationIssue } from '@/store/re-store';

export default function LoaderWorkspace() {
  const {
    sessions,
    project,
    loadMockData,
    selectedSessionId,
    setSelectedSessionId,
    validationIssues,
    timelineScrubSec,
    setTimelineScrubSec,
    selectedSourceTypeFilter,
    setSelectedSourceTypeFilter,
    setActiveWorkspace,
  } = useREStore();

  const [activeBottomTab, setActiveBottomTab] = useState<'queue' | 'timeline' | 'validation'>('validation');
  const [searchQuery, setSearchQuery] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (sessions.length === 0) {
      loadMockData();
    }
  }, [sessions.length, loadMockData]);

  const selectedSession =
    sessions.find((s) => s.id === selectedSessionId) || sessions[0] || null;

  const filteredSessions = sessions.filter((s) => {
    const matchesSource =
      !selectedSourceTypeFilter ||
      s.sources.some((src) => src.type === selectedSourceTypeFilter);
    const matchesSearch =
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.id.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSource && matchesSearch;
  });

  const getSourceIcon = (type?: SessionSourceType) => {
    switch (type) {
      case 'DRONE':
        return <Radio className="w-3.5 h-3.5 text-[#3d8ef7]" />;
      case 'PHONE':
        return <Smartphone className="w-3.5 h-3.5 text-[#2ecc71]" />;
      case 'LIDAR':
        return <Zap className="w-3.5 h-3.5 text-amber-400" />;
      case 'CAMERA_RIG':
        return <Camera className="w-3.5 h-3.5 text-purple-400" />;
      default:
        return <Camera className="w-3.5 h-3.5 text-[#9296a6]" />;
    }
  };

  const getStatusBadge = (status: Session['status']) => {
    switch (status) {
      case 'RECONSTRUCTED':
        return (
          <span className="px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 text-[9px] font-mono font-semibold">
            RECONSTRUCTED
          </span>
        );
      case 'PROCESSING':
        return (
          <span className="px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30 text-[9px] font-mono font-semibold flex items-center gap-1">
            <RefreshCw className="w-2.5 h-2.5 animate-spin" /> PROCESSING
          </span>
        );
      case 'QUEUED':
        return (
          <span className="px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30 text-[9px] font-mono font-semibold">
            QUEUED
          </span>
        );
      case 'FAILED':
        return (
          <span className="px-1.5 py-0.5 rounded bg-red-500/15 text-red-400 border border-red-500/30 text-[9px] font-mono font-semibold">
            FAILED
          </span>
        );
      default:
        return (
          <span className="px-1.5 py-0.5 rounded bg-gray-500/15 text-gray-400 border border-gray-500/30 text-[9px] font-mono font-semibold">
            {status}
          </span>
        );
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#08090b] text-[#f0f1f6] overflow-hidden select-none">
      {/* ── Top Command Bar ────────────────────────────────────────────── */}
      <header className="h-10 px-4 flex items-center justify-between border-b border-[#1f222b] bg-[#0f1014] shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#3d8ef7] tracking-wider">
            <ArrowDownToLine className="w-4 h-4" />
            DATA LOADER & INGESTION
          </div>
          <span className="text-[#54596b]">/</span>
          <span className="text-[#9296a6] font-mono text-[11px]">
            {project ? project.name.toUpperCase() : 'PROJECT CONTEXT'}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative w-56">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#54596b]" />
            <input
              type="text"
              placeholder="Filter sessions & evidence..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full h-7 pl-8 pr-3 bg-[#08090b] border border-[#1f222b] focus:border-[#3d8ef7] rounded text-xs text-[#f0f1f6] placeholder-[#54596b] focus:outline-none"
            />
          </div>

          <button
            type="button"
            onClick={() => setActiveWorkspace('studio')}
            className="flex items-center gap-1.5 px-3 py-1 rounded bg-[#3d8ef7] hover:bg-[#5ca2f9] text-[#08090b] font-semibold text-xs transition-colors"
          >
            <span>Staging to Studio</span>
            <ExternalLink className="w-3 h-3" />
          </button>
        </div>
      </header>

      {/* ── Main Workstation Ingestion Grid ─────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar: Sources Tree & Ingest Filters */}
        <aside className="w-60 border-r border-[#1f222b] bg-[#0d0e12] flex flex-col shrink-0">
          <div className="p-3 border-b border-[#1f222b]">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#54596b] font-semibold block mb-2">
              Heterogeneous Sources
            </span>
            <div className="space-y-1">
              <button
                type="button"
                onClick={() => setSelectedSourceTypeFilter(null)}
                className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded text-xs transition-colors ${
                  selectedSourceTypeFilter === null
                    ? 'bg-[#3d8ef7]/15 text-[#3d8ef7] font-semibold'
                    : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Layers className="w-3.5 h-3.5" />
                  <span>All Sources</span>
                </div>
                <span className="font-mono text-[10px]">{sessions.length}</span>
              </button>

              {(
                [
                  { type: 'DRONE', label: 'Aerial Drones', icon: <Radio className="w-3.5 h-3.5 text-[#3d8ef7]" /> },
                  { type: 'PHONE', label: 'Mobile Handheld', icon: <Smartphone className="w-3.5 h-3.5 text-[#2ecc71]" /> },
                  { type: 'LIDAR', label: 'Terrestrial LiDAR', icon: <Zap className="w-3.5 h-3.5 text-amber-400" /> },
                  { type: 'CAMERA_RIG', label: 'Rigid Camera Arrays', icon: <Camera className="w-3.5 h-3.5 text-purple-400" /> },
                ] as const
              ).map((src) => {
                const count = sessions.filter((s) => s.sources.some((x) => x.type === src.type)).length;
                const isSelected = selectedSourceTypeFilter === src.type;
                return (
                  <button
                    key={src.type}
                    type="button"
                    onClick={() => setSelectedSourceTypeFilter(isSelected ? null : src.type)}
                    className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded text-xs transition-colors ${
                      isSelected
                        ? 'bg-[#3d8ef7]/15 text-[#3d8ef7] font-semibold'
                        : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {src.icon}
                      <span>{src.label}</span>
                    </div>
                    <span className="font-mono text-[10px]">{count}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Staging Pipeline Status */}
          <div className="p-3 border-b border-[#1f222b]">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#54596b] font-semibold block mb-2">
              Ingestion Staging
            </span>
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between text-[#9296a6]">
                <span>Corrupted Files</span>
                <span className="text-[#2ecc71]">0 (None)</span>
              </div>
              <div className="flex justify-between text-[#9296a6]">
                <span>Duplicate Rejection</span>
                <span className="text-[#3d8ef7]">23 frames</span>
              </div>
              <div className="flex justify-between text-[#9296a6]">
                <span>GNSS Coverage</span>
                <span className="text-[#2ecc71]">99.2% Fixed</span>
              </div>
            </div>
          </div>

          {/* Quick Drop Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`m-3 p-4 rounded-xl border border-dashed text-center flex flex-col items-center justify-center transition-colors ${
              isDragging
                ? 'border-[#3d8ef7] bg-[#3d8ef7]/10 text-[#3d8ef7]'
                : 'border-[#1f222b] hover:border-[#3d8ef7]/40 text-[#54596b]'
            }`}
          >
            <FolderOpen className="w-5 h-5 mb-1.5" />
            <span className="text-[11px] font-medium block text-[#9296a6]">Drag Raw Evidence</span>
            <span className="text-[9px] font-mono">RAW, DNG, E57, LAS, MP4</span>
          </div>
        </aside>

        {/* Center: Engineering Sessions Data Table */}
        <div className="flex-1 flex flex-col overflow-hidden bg-[#090a0d]">
          <div className="flex-1 overflow-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead className="sticky top-0 bg-[#0f1014] border-b border-[#1f222b] text-[10px] font-mono text-[#54596b] uppercase tracking-wider">
                <tr>
                  <th className="py-2.5 px-3">Session Identifier</th>
                  <th className="py-2.5 px-2">Source Type</th>
                  <th className="py-2.5 px-2 text-right">Images</th>
                  <th className="py-2.5 px-2 text-right">Points</th>
                  <th className="py-2.5 px-2">GNSS Fix</th>
                  <th className="py-2.5 px-2">IMU Hz</th>
                  <th className="py-2.5 px-2">Reproj Err</th>
                  <th className="py-2.5 px-2">Quality</th>
                  <th className="py-2.5 px-3 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#171920] font-mono text-[11px]">
                {filteredSessions.map((s) => {
                  const isSelected = s.id === selectedSession?.id;
                  const primarySource = s.sources[0];
                  return (
                    <tr
                      key={s.id}
                      onClick={() => setSelectedSessionId(s.id)}
                      className={`cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-[#151720] text-[#f0f1f6]'
                          : 'hover:bg-[#101217] text-[#9296a6]'
                      }`}
                    >
                      <td className="py-2.5 px-3 font-medium text-[#ededf2]">
                        <div className="flex items-center gap-2">
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              isSelected ? 'bg-[#3d8ef7]' : 'bg-transparent'
                            }`}
                          />
                          <span>{s.name}</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-2">
                        <div className="flex items-center gap-1.5">
                          {getSourceIcon(primarySource?.type)}
                          <span className="text-[10px]">{primarySource?.type || 'UNKNOWN'}</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-2 text-right num-tabular">
                        {s.imageCount ? s.imageCount.toLocaleString() : '—'}
                      </td>
                      <td className="py-2.5 px-2 text-right num-tabular">
                        {s.pointCount ? (s.pointCount / 1000000).toFixed(2) + 'M' : '—'}
                      </td>
                      <td className="py-2.5 px-2">
                        {primarySource?.hasGNSS ? (
                          <span className="text-[#2ecc71] text-[10px]">100% Fixed</span>
                        ) : (
                          <span className="text-[#54596b] text-[10px]">N/A</span>
                        )}
                      </td>
                      <td className="py-2.5 px-2">
                        {primarySource?.hasIMU ? (
                          <span className="text-[#2ecc71] text-[10px]">200 Hz</span>
                        ) : (
                          <span className="text-[#54596b] text-[10px]">N/A</span>
                        )}
                      </td>
                      <td className="py-2.5 px-2 num-tabular">
                        {s.reprojectionError ? `${s.reprojectionError.toFixed(2)} px` : '—'}
                      </td>
                      <td className="py-2.5 px-2 num-tabular">
                        {s.quality ? (
                          <span className="text-[#2ecc71]">{(s.quality * 100).toFixed(0)}%</span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-right">
                        {getStatusBadge(s.status)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ── Bottom Diagnostic Panel (Queue / Timeline / Validation) ── */}
          <div className="h-52 border-t border-[#1f222b] bg-[#0c0d11] flex flex-col shrink-0">
            {/* Tabs Header */}
            <div className="h-8 px-3 border-b border-[#1f222b] flex items-center justify-between bg-[#0f1014] text-xs">
              <div className="flex items-center gap-1">
                {(
                  [
                    { id: 'validation' as const, label: 'Validation Engine', badge: validationIssues.length as number | undefined },
                    { id: 'timeline' as const, label: 'Interactive Session Timeline', badge: undefined as number | undefined },
                    { id: 'queue' as const, label: 'Import Queue Diagnostics (3 Jobs)', badge: undefined as number | undefined },
                  ]
                ).map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveBottomTab(tab.id)}
                    className={`px-3 py-1 rounded text-[11px] font-mono flex items-center gap-1.5 transition-colors ${
                      activeBottomTab === tab.id
                        ? 'bg-[#1a1d26] text-[#3d8ef7] font-semibold border border-[#3d8ef7]/30'
                        : 'text-[#9296a6] hover:text-[#f0f1f6]'
                    }`}
                  >
                    <span>{tab.label}</span>
                    {tab.badge !== undefined && (
                      <span className="px-1.5 py-0.2 rounded-full bg-amber-500/20 text-amber-400 text-[9px]">
                        {tab.badge}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>

            {/* Bottom Tab Contents */}
            <div className="flex-1 p-3 overflow-y-auto">
              {activeBottomTab === 'validation' && (
                <div className="space-y-2">
                  {validationIssues.map((v) => (
                    <div
                      key={v.id}
                      className={`p-2.5 rounded-lg border text-xs ${
                        v.severity === 'ERROR'
                          ? 'bg-red-500/10 border-red-500/30'
                          : v.severity === 'WARNING'
                          ? 'bg-amber-500/10 border-amber-500/30'
                          : 'bg-emerald-500/10 border-emerald-500/30'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1.5 font-semibold">
                          {v.severity === 'ERROR' && <AlertOctagon className="w-3.5 h-3.5 text-red-400" />}
                          {v.severity === 'WARNING' && <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />}
                          {v.severity === 'PASS' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
                          <span
                            className={
                              v.severity === 'ERROR'
                                ? 'text-red-400'
                                : v.severity === 'WARNING'
                                ? 'text-amber-400'
                                : 'text-emerald-400'
                            }
                          >
                            {v.title}
                          </span>
                        </div>
                        <span className="font-mono text-[10px] text-[#9296a6]">
                          {v.affectedCount} {v.affectedUnit} affected
                        </span>
                      </div>

                      <p className="text-[11px] text-[#ededf2] mb-1">{v.description}</p>
                      <div className="flex items-center justify-between text-[10px] font-mono text-[#9296a6] pt-1 border-t border-white/5">
                        <span><strong>Evidence:</strong> {v.evidence}</span>
                        <span><strong>Action:</strong> {v.recommendedAction}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeBottomTab === 'timeline' && (
                <div className="flex flex-col h-full justify-between font-mono text-xs">
                  <div className="flex items-center justify-between text-[10px] text-[#9296a6] mb-1">
                    <span>TIMELINE SCRUB: T + {timelineScrubSec.toFixed(1)}s</span>
                    <span>PASS DURATION: 180.0s (100% COVERAGE)</span>
                  </div>

                  {/* Interactive Timeline Track */}
                  <div className="relative h-12 bg-[#12141a] rounded-lg border border-[#1f222b] overflow-hidden px-2 flex items-center">
                    {/* Simulated event markers */}
                    <div className="absolute inset-x-4 top-2 h-2 flex gap-1">
                      {Array.from({ length: 40 }).map((_, i) => (
                        <div
                          key={i}
                          className={`flex-1 rounded-sm ${
                            i === 12
                              ? 'bg-amber-400' // Drift warning marker
                              : i % 2 === 0
                              ? 'bg-[#3d8ef7]'
                              : 'bg-[#2ecc71]'
                          }`}
                        />
                      ))}
                    </div>

                    {/* Scrubber needle */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-red-400 z-10"
                      style={{ left: `${(timelineScrubSec / 180) * 100}%` }}
                    >
                      <div className="w-2.5 h-2.5 bg-red-400 -ml-1 rounded-full shadow" />
                    </div>

                    {/* Time ticks */}
                    <div className="absolute bottom-1 inset-x-3 flex justify-between text-[8px] text-[#54596b]">
                      <span>00:00</span>
                      <span>00:45</span>
                      <span>01:30</span>
                      <span>02:15</span>
                      <span>03:00</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-[#9296a6] pt-2">
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-[#3d8ef7]" /> Drone Photogrammetry
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-[#2ecc71]" /> Handheld Walkthrough
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-amber-400" /> PTP Drift Warning
                    </span>
                  </div>
                </div>
              )}

              {activeBottomTab === 'queue' && (
                <div className="space-y-2 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-[#12141a] border border-[#1f222b] flex items-center justify-between">
                    <div>
                      <span className="font-semibold text-[#f0f1f6] block">
                        JOB-001: Extract Keyframes & Intrinsics
                      </span>
                      <span className="text-[10px] text-[#9296a6]">
                        Imported 1,248 frames • 23 duplicate frames rejected • 4 corrupted files
                      </span>
                    </div>
                    <span className="text-xs text-[#2ecc71] font-bold">100% COMPLETE</span>
                  </div>

                  <div className="p-2.5 rounded-lg bg-[#12141a] border border-[#1f222b] flex items-center justify-between">
                    <div>
                      <span className="font-semibold text-[#f0f1f6] block">
                        JOB-002: GNSS Carrier-Phase RTK Triangulation
                      </span>
                      <span className="text-[10px] text-[#9296a6]">
                        GNSS present for 91.3% of frames • Fixed Ambiguity Mode
                      </span>
                    </div>
                    <span className="text-xs text-[#3d8ef7] font-bold">RUNNING (84%)</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Sidebar: Property-Grid Metadata Inspector */}
        <aside className="w-72 border-l border-[#1f222b] bg-[#0c0d11] flex flex-col shrink-0 overflow-y-auto">
          <div className="p-3 border-b border-[#1f222b]">
            <span className="text-[10px] font-mono uppercase tracking-wider text-[#54596b] font-semibold block mb-1">
              Metadata Inspector
            </span>
            <h3 className="font-bold text-xs text-[#f0f1f6] truncate">
              {selectedSession ? selectedSession.name : 'No Session Selected'}
            </h3>
          </div>

          {selectedSession ? (
            <div className="p-3 space-y-4 text-xs font-mono">
              {/* Property Grid: Camera Intrinsics */}
              <div>
                <span className="text-[10px] text-[#3d8ef7] font-semibold uppercase tracking-wider block mb-2">
                  Sensor Specifications
                </span>
                <div className="space-y-1.5 text-[11px]">
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Device</span>
                    <span className="text-[#ededf2]">Sony α7R V / Hasselblad</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Sensor Size</span>
                    <span className="text-[#ededf2]">35.7 × 23.8 mm (Full-Frame)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Focal Length</span>
                    <span className="text-[#ededf2]">24.0 mm (Calibrated)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Resolution</span>
                    <span className="text-[#ededf2]">9504 × 6336 px (61 MP)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Distortion</span>
                    <span className="text-[#ededf2]">k1=-0.042, k2=0.012</span>
                  </div>
                </div>
              </div>

              {/* Property Grid: Geodetic & IMU Telemetry */}
              <div>
                <span className="text-[10px] text-[#2ecc71] font-semibold uppercase tracking-wider block mb-2">
                  Geospatial Positioning
                </span>
                <div className="space-y-1.5 text-[11px]">
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Coordinate Ref</span>
                    <span className="text-[#ededf2]">EPSG:32645 (UTM 45N)</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Datum</span>
                    <span className="text-[#ededf2]">WGS 84</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">RTK Covariance</span>
                    <span className="text-[#2ecc71]">± 0.014 m</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">IMU Rate</span>
                    <span className="text-[#ededf2]">200 Hz Tri-axial</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#171920]">
                    <span className="text-[#54596b]">Clock Drift</span>
                    <span className="text-amber-400">1.8 ms (Synced)</span>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => setActiveWorkspace('studio')}
                  className="w-full py-2.5 rounded-lg bg-[#3d8ef7] hover:bg-[#5ca2f9] text-[#08090b] font-bold text-xs uppercase tracking-wider transition-colors"
                >
                  Send to Reconstruction
                </button>
              </div>
            </div>
          ) : (
            <div className="p-4 text-center text-xs text-[#54596b]">
              Select a session from the table to inspect detailed sensor properties.
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
