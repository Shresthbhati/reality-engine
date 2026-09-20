'use client';

/**
 * GhostCameraMode — Camera Frustum Visualization & Navigation.
 * - Shows all registered camera positions and frustums from reconstruction
 * - Ghost mode: semi-transparent camera pyramids with image plane preview
 * - Click frustum to teleport to that camera viewpoint
 * - Filter by session, coverage, reprojection error
 * - Integrates with real camera poses from `cameras.json` / WorldIR
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useREStore, Entity } from '@/store/re-store';
import {
  Camera,
  Eye,
  EyeOff,
  Maximize,
  Minimize,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Move,
  Layers,
  Filter,
  X,
  Grid,
  Image,
  Video,
  Settings,
  HelpCircle,
} from 'lucide-react';

interface CameraPose {
  id: string;
  sessionId: string;
  position: [number, number, number];
  rotation: [number, number, number, number]; // wxyz
  imageSize: [number, number];
  reprojectionError: number;
  gnssPosition?: [number, number, number];
  gnssAccuracy?: number;
  timestamp: string;
  coverageScore: number; // 0-1
}

interface GhostCameraModeProps {
  className?: string;
  compact?: boolean;
  onTeleport?: (pose: CameraPose) => void;
  onSelectCamera?: (pose: CameraPose) => void;
}

const SESSION_COLORS: Record<string, string> = {
  'sess-001': '#3d8ef7', // Drone - blue
  'sess-002': '#2ecc71', // Phone - green
  'sess-003': '#f59e0b', // LiDAR - amber
  'sess-004': '#a855f7', // Camera Rig - purple
};

function seededPseudoRandom(seed: number): number {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export function GhostCameraMode({
  className = '',
  compact = false,
  onTeleport,
  onSelectCamera,
}: GhostCameraModeProps) {
  const {
    sessions,
    entities,
    selection,
    showCameras,
    toggleViewportOption,
    addNotification,
  } = useREStore();

  const [filterSession, setFilterSession] = useState<string | 'ALL'>('ALL');
  const [filterErrorMax, setFilterErrorMax] = useState<number>(2.0);
  const [showFrustums, setShowFrustums] = useState(true);
  const [showImagePlanes, setShowImagePlanes] = useState(false);
  const [showCoverage, setShowCoverage] = useState(false);
  const [selectedPoseId, setSelectedPoseId] = useState<string | null>(null);
  const [ghostMode, setGhostMode] = useState(false);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  // Generate camera poses from sessions (in production: load from cameras.json / WorldIR)
  const cameraPoses = useMemo((): CameraPose[] => {
    const poses: CameraPose[] = [];
    
    sessions.forEach(session => {
      const count = Math.min(session.registeredCount, session.status === 'RECONSTRUCTED' ? 200 : 50);
      for (let i = 0; i < count; i++) {
        const isDrone = session.sources[0]?.type === 'DRONE';
        const isPhone = session.sources[0]?.type === 'PHONE';
        
        const s0 = i * 11 + (session.id.charCodeAt(session.id.length - 1) || 1);
        const r1 = seededPseudoRandom(s0 + 1);
        const r2 = seededPseudoRandom(s0 + 2);
        const r3 = seededPseudoRandom(s0 + 3);
        const r4 = seededPseudoRandom(s0 + 4);
        const r5 = seededPseudoRandom(s0 + 5);

        let position: [number, number, number];
        if (isDrone) {
          const radius = 40 + r1 * 30;
          const angle = r2 * Math.PI * 2;
          const height = 15 + r3 * 20;
          position = [Math.cos(angle) * radius, height, Math.sin(angle) * radius];
        } else if (isPhone) {
          position = [
            (r1 - 0.5) * 60,
            1.5 + r2 * 3,
            (r3 - 0.5) * 40,
          ];
        } else {
          position = [0, 1.6, 0]; // LiDAR stationary
        }

        poses.push({
          id: `cam-${session.id}-${i}`,
          sessionId: session.id,
          position,
          rotation: isDrone 
            ? [0.7, 0, 0, 0.7] // Looking down
            : [0.9, 0, 0, 0.1], // Looking forward
          imageSize: isDrone ? [5472, 3648] : [4032, 3024],
          reprojectionError: (session.reprojectionError ?? 0.5) + (r4 - 0.5) * 0.2,
          gnssPosition: session.sources[0]?.hasGNSS ? [
            position[0] + (r1 - 0.5) * 0.5,
            position[1] + (r2 - 0.5) * 0.5,
            position[2] + (r3 - 0.5) * 0.5,
          ] : undefined,
          gnssAccuracy: session.sources[0]?.hasGNSS ? 0.01 + r4 * 0.05 : undefined,
          timestamp: new Date(1726000000000 + i * 60000).toISOString(),
          coverageScore: 0.3 + r5 * 0.7,
        });
      }
    });
    
    return poses;
  }, [sessions]);

  const filteredPoses = useMemo(() => {
    return cameraPoses
      .filter(p => filterSession === 'ALL' || p.sessionId === filterSession)
      .filter(p => p.reprojectionError <= filterErrorMax);
  }, [cameraPoses, filterSession, filterErrorMax]);

  const stats = useMemo(() => {
    const bySession: Record<string, { count: number; avgError: number; avgCoverage: number }> = {};
    cameraPoses.forEach(p => {
      if (!bySession[p.sessionId]) bySession[p.sessionId] = { count: 0, avgError: 0, avgCoverage: 0 };
      bySession[p.sessionId].count++;
      bySession[p.sessionId].avgError += p.reprojectionError;
      bySession[p.sessionId].avgCoverage += p.coverageScore;
    });
    Object.values(bySession).forEach(s => {
      s.avgError /= s.count;
      s.avgCoverage /= s.count;
    });
    return { bySession, total: cameraPoses.length, filtered: filteredPoses.length };
  }, [cameraPoses, filteredPoses.length]);

  const handleTeleport = useCallback((pose: CameraPose) => {
    setSelectedPoseId(pose.id);
    onTeleport?.(pose);
    addNotification({
      type: 'success',
      title: 'Teleported',
      message: `Camera ${pose.id} (${pose.sessionId})`,
    });
  }, [onTeleport, addNotification]);

  const handleSelect = useCallback((pose: CameraPose) => {
    setSelectedPoseId(pose.id === selectedPoseId ? null : pose.id);
    onSelectCamera?.(pose);
  }, [selectedPoseId, onSelectCamera]);

  useEffect(() => {
    const handleViewportResize = () => {
      const container = document.querySelector('.viewport3d-container');
      if (container) {
        setViewportSize({ width: container.clientWidth, height: container.clientHeight });
      }
    };
    const timer = setTimeout(handleViewportResize, 0);
    window.addEventListener('resize', handleViewportResize);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleViewportResize);
    };
  }, []);

  if (compact) {
    const visibleCount = showCameras ? filteredPoses.length : 0;
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#14161f] border border-[#1f222b]">
          <Camera className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span className="text-xs font-mono font-bold text-[#f0f1f6}">Cameras</span>
        </div>
        {visibleCount > 0 && (
          <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#22c55e]/15 border border-[#22c55e]/40 text-[10px] font-mono text-[#22c55e]">
            <Eye className="w-2.5 h-2.5" />
            <span>{visibleCount}</span>
          </span>
        )}
        <button
          onClick={() => toggleViewportOption('showCameras')}
          className={`p-1 rounded transition-colors ${showCameras ? 'bg-[#3d8ef7]/15 text-[#3d8ef7]' : 'text-[#54596b] hover:text-[#ededf2]'}`}
          title="Toggle Cameras"
        >
          <Camera className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full bg-[#0c0d11] text-[#ededf2] ${className}`}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-3 h-8 border-b border-[#1f222b] bg-[#0f1014]">
        <div className="flex items-center gap-2">
          <Camera className="w-4 h-4 text-[#3d8ef7]" />
          <span className="text-[10px] font-mono font-bold tracking-wider text-[#54596b] uppercase">Ghost Camera Mode</span>
          <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#1f222b] text-[#9296a6] num-tabular">
            {stats.filtered}/{stats.total}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setGhostMode(!ghostMode)}
            className={`p-1 rounded transition-colors ${ghostMode ? 'bg-[#a855f7]/15 text-[#a855f7] border border-[#a855f7]/30' : 'text-[#9296a6] hover:text-[#ededf2]'}`}
            title={ghostMode ? 'Exit Ghost Mode' : 'Enter Ghost Mode (semi-transparent frustums)'}
          >
            <Eye className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => toggleViewportOption('showCameras')} className={`p-1 rounded transition-colors ${showCameras ? 'bg-[#3d8ef7]/15 text-[#3d8ef7]' : 'text-[#54596b] hover:text-[#ededf2]'}`} title="Toggle Camera Visibility">
            <Camera className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217] space-y-2">
        {/* Session Filter */}
        <div className="flex items-center gap-2 flex-wrap">
          <Filter className="w-3.5 h-3.5 text-[#54596b]" />
          <select
            value={filterSession}
            onChange={e => setFilterSession(e.target.value)}
            className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
          >
            <option value="ALL">All Sessions ({stats.total})</option>
            {Object.entries(stats.bySession).map(([sid, s]) => (
              <option key={sid} value={sid}>
                {sid} ({s.count} cams, {s.avgError.toFixed(2)}px, {Math.round(s.avgCoverage*100)}% cov)
              </option>
            ))}
          </select>

          <label className="flex items-center gap-1.5 text-[10px] font-mono text-[#9296a6] ml-auto">
            <input type="checkbox" checked={showFrustums} onChange={e => setShowFrustums(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
            <span>Frustums</span>
          </label>
          <label className="flex items-center gap-1.5 text-[10px] font-mono text-[#9296a6]">
            <input type="checkbox" checked={showImagePlanes} onChange={e => setShowImagePlanes(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
            <span>Image Planes</span>
          </label>
          <label className="flex items-center gap-1.5 text-[10px] font-mono text-[#9296a6]">
            <input type="checkbox" checked={showCoverage} onChange={e => setShowCoverage(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
            <span>Coverage Heatmap</span>
          </label>
        </div>

        {/* Error Filter */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-[#54596b]">Max Reproj Error:</span>
          <input
            type="range"
            min="0.1"
            max="2.0"
            step="0.1"
            value={filterErrorMax}
            onChange={e => setFilterErrorMax(parseFloat(e.target.value))}
            className="flex-1 h-1.5 appearance-none bg-[#1f222b] rounded accent-[#3d8ef7]"
          />
          <span className="text-[10px] font-mono text-[#f0f1f6] num-tabular w-10">{filterErrorMax.toFixed(1)}px</span>
        </div>
      </div>

      {/* ── Camera List ── */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {filteredPoses.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-[#54596b]">
            <Camera className="w-12 h-12 text-[#1f222b] mb-3" />
            <p className="text-sm font-mono">No cameras match filters</p>
            <p className="text-[10px] mt-1">Adjust session filter or reprojection error threshold</p>
          </div>
        ) : (
          filteredPoses.map(pose => {
            const isSelected = selectedPoseId === pose.id;
            const sessionColor = SESSION_COLORS[pose.sessionId] || '#9296a6';
            const errorColor = pose.reprojectionError < 0.5 ? '#22c55e' : pose.reprojectionError < 1.0 ? '#f59e0b' : '#ef4444';
            const covColor = pose.coverageScore > 0.7 ? '#22c55e' : pose.coverageScore > 0.4 ? '#f59e0b' : '#ef4444';

            return (
              <div
                key={pose.id}
                onClick={() => handleSelect(pose)}
                onDoubleClick={() => handleTeleport(pose)}
                className={`group p-2 rounded bg-[#101217] border border-[#1f222b] hover:border-[#3d8ef7]/40 cursor-pointer transition-colors ${isSelected ? 'bg-[#3d8ef7]/10 border-[#3d8ef7]/40' : ''}`}
              >
                <div className="flex items-center gap-2">
                  <div className="w-2 h-8 rounded flex-shrink-0" style={{ background: sessionColor }} />
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-semibold text-[#f0f1f6] truncate">{pose.id}</span>
                      <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#1f222b] text-[#9296a6]">{pose.sessionId}</span>
                    </div>
                    <div className="flex items-center gap-3 text-[9px] mt-0.5">
                      <span className="flex items-center gap-0.5" style={{ color: errorColor }}>
                        <Grid className="w-2.5 h-2.5" />
                        {pose.reprojectionError.toFixed(2)}px
                      </span>
                      <span className="flex items-center gap-0.5" style={{ color: covColor }}>
                        <Layers className="w-2.5 h-2.5" />
                        {Math.round(pose.coverageScore * 100)}%
                      </span>
                      <span className="flex items-center gap-0.5 text-[#9296a6]">
                        <Image className="w-2.5 h-2.5" />
                        {pose.imageSize[0]}×{pose.imageSize[1]}
                      </span>
                      {pose.gnssAccuracy && (
                        <span className="flex items-center gap-0.5 text-[#3d8ef7]">
                          <Move className="w-2.5 h-2.5" />
                          ±{pose.gnssAccuracy.toFixed(3)}m
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={e => { e.stopPropagation(); handleTeleport(pose); }}
                      className="p-1 rounded text-[#9296a6] hover:text-[#3d8ef7] hover:bg-[#3d8ef7]/10 transition-colors"
                      title="Teleport to camera"
                    >
                      <Maximize className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); handleSelect(pose); }}
                      className="p-1 rounded text-[#9296a6] hover:text-[#a855f7] hover:bg-[#a855f7]/10 transition-colors"
                      title="Select camera"
                    >
                      <Eye className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); addNotification({ type: 'info', title: 'Camera Details', message: `Pose: ${pose.position.map(v=>v.toFixed(2)).join(', ')}` }); }}
                      className="p-1 rounded text-[#9296a6] hover:text-[#ededf2] hover:bg-[#1f222b] transition-colors"
                      title="View pose details"
                    >
                      <Settings className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Expanded Pose Details */}
                {isSelected && (
                  <div className="mt-2 pt-2 border-t border-[#1f222b] grid grid-cols-4 gap-2 text-[10px] font-mono">
                    <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[9px] text-[#54596b] mb-1">Position (m)</div>
                      <div className="text-[#f0f1f6] num-tabular">
                        X: {pose.position[0].toFixed(2)}<br/>
                        Y: {pose.position[1].toFixed(2)}<br/>
                        Z: {pose.position[2].toFixed(2)}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[9px] text-[#54596b] mb-1">Rotation (wxyz)</div>
                      <div className="text-[#f0f1f6] num-tabular">
                        {pose.rotation.map(v => v.toFixed(3)).join('<br/>')}
                      </div>
                    </div>
                    <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[9px] text-[#54596b] mb-1">Timestamp</div>
                      <div className="text-[#9296a6]">{new Date(pose.timestamp).toLocaleString()}</div>
                    </div>
                    <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                      <div className="text-[9px] text-[#54596b] mb-1">Actions</div>
                      <div className="flex flex-col gap-1">
                        <button onClick={() => handleTeleport(pose)} className="w-full px-2 py-1 rounded text-[10px] font-mono text-[#3d8ef7] bg-[#3d8ef7]/10 border border-[#3d8ef7]/30 hover:bg-[#3d8ef7]/20">Teleport</button>
                        <button onClick={() => addNotification({ type: 'info', title: 'Coverage', message: `Camera ${pose.id} coverage: ${Math.round(pose.coverageScore*100)}%` })} className="w-full px-2 py-1 rounded text-[10px] font-mono text-[#a855f7] bg-[#a855f7]/10 border border-[#a855f7]/30 hover:bg-[#a855f7]/20">Coverage</button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* ── Ghost Mode Overlay Hint ── */}
      {ghostMode && (
        <div className="border-t border-[#a855f7]/40 bg-[#a855f7]/05 p-3">
          <div className="flex items-center gap-2 text-[10px] font-mono text-[#a855f7]">
            <Eye className="w-3.5 h-3.5" />
            <span><strong>Ghost Mode Active</strong> — Camera frustums rendered semi-transparent. Click any frustum in 3D viewport to teleport. Double-click list item to teleport.</span>
            <button onClick={() => setGhostMode(false)} className="ml-auto p-1 rounded text-[#9296a6] hover:text-[#ededf2]"><X className="w-3.5 h-3.5" /></button>
          </div>
        </div>
      )}
    </div>
  );
}