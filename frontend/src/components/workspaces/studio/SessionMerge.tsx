'use client';

/**
 * SessionMerge — Multi-Source Session Alignment & Conflict Resolution.
 * - Visual interface for merging sessions from different sources (phone, drone, LiDAR, rig)
 * - Shows registration status, overlap quality, GNSS anchor availability
 * - Conflict detection: coordinate frame mismatch, temporal drift, calibration inconsistency
 * - Resolution actions: accept alignment, adjust transform, reject source, manual override
 * - Integrates with `reality register` CLI and registration engine backend
 */

import React, { useState, useMemo, useCallback } from 'react';
import { useREStore, Session, SessionSource, SessionSourceType, SessionStatus } from '@/store/re-store';
import {
  ChevronRight,
  ChevronDown,
  Plus,
  Minus,
  GitMerge,
  GitCompare,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Settings,
  Link2,
  Unlink2,
  Move,
  RotateCcw,
  Scale,
  Layers,
  Eye,
  EyeOff,
  Terminal,
  ExternalLink,
  HelpCircle,
} from 'lucide-react';

export type SessionMergeStatus = 'PENDING' | 'ALIGNED' | 'CONFLICT' | 'MERGED' | 'REJECTED';

export interface SourceAlignment {
  sourceId: string;
  sourceType: SessionSourceType;
  sessionId: string;
  status: SessionMergeStatus;
  gnssAnchors: number;
  overlapPercent: number;
  registrationError?: number; // meters
  transform?: {
    translation: [number, number, number];
    rotation: [number, number, number, number]; // wxyz
    scale: number;
  };
  conflicts: AlignmentConflict[];
  lastUpdated: string;
}

export interface AlignmentConflict {
  id: string;
  type: 'FRAME_MISMATCH' | 'TEMPORAL_DRIFT' | 'CALIBRATION_INCONSISTENT' | 'GNSS_OUTLIER' | 'OVERLAP_INSUFFICIENT';
  severity: 'CRITICAL' | 'WARNING' | 'INFO';
  description: string;
  suggestedAction: string;
  autoResolvable: boolean;
}

interface SessionMergeProps {
  className?: string;
  compact?: boolean;
  onMergeComplete?: (mergedSessionId: string) => void;
  onAlignmentUpdate?: (alignment: SourceAlignment) => void;
}

const STATUS_COLORS: Record<SessionMergeStatus, { bg: string; border: string; text: string }> = {
  PENDING: { bg: '#3d8ef715', border: '#3d8ef7', text: '#3d8ef7' },
  ALIGNED: { bg: '#22c55e15', border: '#22c55e', text: '#22c55e' },
  CONFLICT: { bg: '#ef444415', border: '#ef4444', text: '#ef4444' },
  MERGED: { bg: '#a855f715', border: '#a855f7', text: '#a855f7' },
  REJECTED: { bg: '#54596b15', border: '#54596b', text: '#54596b' },
};

const CONFLICT_SEVERITY_COLORS = {
  CRITICAL: { bg: '#ef444415', border: '#ef4444', text: '#ef4444', icon: AlertTriangle },
  WARNING: { bg: '#f59e0b15', border: '#f59e0b', text: '#f59e0b', icon: AlertTriangle },
  INFO: { bg: '#3d8ef715', border: '#3d8ef7', text: '#3d8ef7', icon: HelpCircle },
};

const SOURCE_TYPE_LABELS: Record<SessionSourceType, string> = {
  PHONE: '📱 Phone',
  DRONE: '🚁 Drone',
  CAMERA_RIG: '🎥 Camera Rig',
  LIDAR: '📡 LiDAR',
  RGB_D: '📷 RGB-D',
  STEREO: '👁 Stereo',
  VIDEO: '📹 Video',
  EXISTING_DATASET: '💾 Dataset',
};

export function SessionMerge({
  className = '',
  compact = false,
  onMergeComplete,
  onAlignmentUpdate,
}: SessionMergeProps) {
  const {
    sessions,
    entities,
    addNotification,
    setActiveWorkspace,
  } = useREStore();

  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [showConflictsOnly, setShowConflictsOnly] = useState(false);
  const [autoResolve, setAutoResolve] = useState(false);

  // Generate mock alignments (in production, from registration engine)
  const alignments = useMemo((): SourceAlignment[] => {
    return sessions.flatMap(session => 
      session.sources.map((source, idx) => {
        const sourceId = `${session.id}-src-${idx}`;
        const hasConflict = session.status === 'PROCESSING' || (source.type === 'LIDAR' && idx === 0);
        
        return {
          sourceId,
          sourceType: source.type,
          sessionId: session.id,
          status: hasConflict ? 'CONFLICT' : session.status === 'RECONSTRUCTED' ? 'ALIGNED' : 'PENDING',
          gnssAnchors: source.hasGNSS ? Math.floor(Math.random() * 8) + 3 : 0,
          overlapPercent: Math.floor(Math.random() * 40) + 50,
          registrationError: hasConflict ? 0.15 + Math.random() * 0.3 : 0.02 + Math.random() * 0.05,
          transform: {
            translation: [
              (Math.random() - 0.5) * 2,
              (Math.random() - 0.5) * 1,
              (Math.random() - 0.5) * 2,
            ],
            rotation: [1, 0, 0, 0],
            scale: 0.999 + Math.random() * 0.002,
          },
          conflicts: hasConflict ? [
            {
              id: `conflict-${sourceId}-1`,
              type: source.type === 'LIDAR' ? 'FRAME_MISMATCH' : 'TEMPORAL_DRIFT',
              severity: 'CRITICAL',
              description: source.type === 'LIDAR' 
                ? 'LiDAR coordinate frame (ENU) differs from photogrammetry frame (local tangent plane)'
                : 'PTP drift of 42ms detected between drone and phone streams',
              suggestedAction: source.type === 'LIDAR'
                ? 'Apply ENU-to-local transform using GNSS anchors. Verify with registration engine.'
                : 'Re-sync PTP. Apply cubic B-spline temporal interpolation.',
              autoResolvable: true,
            },
            {
              id: `conflict-${sourceId}-2`,
              type: 'CALIBRATION_INCONSISTENT',
              severity: 'WARNING',
              description: 'Intrinsics calibration differs between sessions (focal length variance > 2%)',
              suggestedAction: 'Re-calibrate with shared calibration target. Use Brown-Conrady model.',
              autoResolvable: false,
            },
          ] : [],
          lastUpdated: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        };
      })
    );
  }, [sessions]);

  const filteredAlignments = useMemo(() => {
    return alignments.filter(a => !showConflictsOnly || a.status === 'CONFLICT');
  }, [alignments, showConflictsOnly]);

  const stats = useMemo(() => {
    const counts: Record<SessionMergeStatus, number> = { PENDING: 0, ALIGNED: 0, CONFLICT: 0, MERGED: 0, REJECTED: 0 };
    alignments.forEach(a => counts[a.status]++);
    return counts;
  }, [alignments]);

  const allConflicts = useMemo(() => 
    alignments.flatMap(a => a.conflicts.map(c => ({ ...c, sourceId: a.sourceId, sourceType: a.sourceType }))),
    [alignments]
  );

  const toggleExpand = useCallback((id: string) => {
    setExpandedSources(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const resolveConflict = useCallback((conflict: AlignmentConflict, sourceId: string) => {
    if (!conflict.autoResolvable) {
      addNotification({
        type: 'warning',
        title: 'Manual Resolution Required',
        message: conflict.description,
      });
      return;
    }

    // Simulate auto-resolution
    addNotification({
      type: 'success',
      title: 'Conflict Auto-Resolved',
      message: `${conflict.type}: ${conflict.suggestedAction}`,
    });

    // Update alignment status
    const alignment = alignments.find(a => a.sourceId === sourceId);
    if (alignment) {
      alignment.conflicts = alignment.conflicts.filter(c => c.id !== conflict.id);
      if (alignment.conflicts.length === 0) {
        alignment.status = 'ALIGNED';
        alignment.registrationError = 0.02;
      }
      onAlignmentUpdate?.(alignment);
    }
  }, [onAlignmentUpdate, addNotification]);

  const rejectSource = useCallback((sourceId: string) => {
    const alignment = alignments.find(a => a.sourceId === sourceId);
    if (alignment) {
      alignment.status = 'REJECTED';
      onAlignmentUpdate?.(alignment);
      addNotification({
        type: 'warning',
        title: 'Source Rejected',
        message: `${SOURCE_TYPE_LABELS[alignment.sourceType]} removed from merge`,
      });
    }
  }, [onAlignmentUpdate, addNotification]);

  const mergeSessions = useCallback(() => {
    const conflictCount = allConflicts.length;
    if (conflictCount > 0 && !autoResolve) {
      addNotification({
        type: 'warning',
        title: 'Unresolved Conflicts',
        message: `${conflictCount} conflicts must be resolved before merge. Enable auto-resolve or fix manually.`,
      });
      return;
    }

    const mergedSessionId = `sess-merged-${Date.now()}`;
    addNotification({
      type: 'success',
      title: 'Sessions Merged',
      message: `Created unified session ${mergedSessionId} from ${alignments.length} sources`,
    });
    onMergeComplete?.(mergedSessionId);
    setActiveWorkspace('studio');
  }, [allConflicts.length, autoResolve, alignments.length, onMergeComplete, addNotification, setActiveWorkspace]);

  if (compact) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#14161f] border border-[#1f222b]">
          <GitMerge className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span className="text-xs font-mono font-bold text-[#f0f1f6]">Session Merge</span>
        </div>
        {stats.CONFLICT > 0 && (
          <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#ef4444]/15 border border-[#ef4444]/40 text-[10px] font-mono text-[#ef4444]">
            <AlertTriangle className="w-2.5 h-2.5" />
            <span>{stats.CONFLICT} conflicts</span>
          </span>
        )}
        {stats.PENDING > 0 && (
          <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#3d8ef7]/15 border border-[#3d8ef7]/40 text-[10px] font-mono text-[#3d8ef7]">
            <RefreshCw className="w-2.5 h-2.5" />
            <span>{stats.PENDING} pending</span>
          </span>
        )}
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full bg-[#0c0d11] text-[#ededf2] ${className}`}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-3 h-8 border-b border-[#1f222b] bg-[#0f1014]">
        <div className="flex items-center gap-2">
          <GitMerge className="w-4 h-4 text-[#3d8ef7]" />
          <span className="text-[10px] font-mono font-bold tracking-wider text-[#54596b] uppercase">Session Merge & Alignment</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={mergeSessions}
            disabled={allConflicts.length > 0 && !autoResolve}
            className="flex items-center gap-1 px-3 py-1.5 rounded bg-[#3d8ef7] hover:bg-[#2b7ae2] text-[#08090b] font-bold text-xs text-[#08090b] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Merge Sessions</span>
          </button>
        </div>
      </div>

      {/* ── Summary Stats ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217]">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#3d8ef7]/15 border border-[#3d8ef7]/40">
            <span className="text-xs font-bold text-[#3d8ef7]">{stats.PENDING}</span>
            <span className="text-[10px] text-[#9296a6]">Pending</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#22c55e]/15 border border-[#22c55e]/40">
            <span className="text-xs font-bold text-[#22c55e]">{stats.ALIGNED}</span>
            <span className="text-[10px] text-[#9296a6]">Aligned</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#ef4444]/15 border border-[#ef4444]/40">
            <span className="text-xs font-bold text-[#ef4444]">{stats.CONFLICT}</span>
            <span className="text-[10px] text-[#9296a6]">Conflicts</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#a855f7]/15 border border-[#a855f7]/40">
            <span className="text-xs font-bold text-[#a855f7]">{stats.MERGED}</span>
            <span className="text-[10px] text-[#9296a6]">Merged</span>
          </div>
          <label className="flex items-center gap-1 text-[10px] text-[#9296a6] cursor-pointer ml-auto">
            <input
              type="checkbox"
              checked={showConflictsOnly}
              onChange={e => setShowConflictsOnly(e.target.checked)}
              className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7] focus:ring-[#3d8ef7]"
            />
            <span>Conflicts Only</span>
          </label>
          <label className="flex items-center gap-1 text-[10px] text-[#9296a6] cursor-pointer">
            <input
              type="checkbox"
              checked={autoResolve}
              onChange={e => setAutoResolve(e.target.checked)}
              className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7] focus:ring-[#3d8ef7]"
            />
            <span>Auto-Resolve</span>
          </label>
        </div>
      </div>

      {/* ── Source Alignments ── */}
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {filteredAlignments.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-[#54596b]">
            <GitMerge className="w-12 h-12 text-[#1f222b] mb-3" />
            <p className="text-sm font-mono">No sessions to merge</p>
            <p className="text-[10px] mt-1">Create sessions in Loader workspace or import evidence packages</p>
          </div>
        ) : (
          filteredAlignments.map(alignment => {
            const statusColors = STATUS_COLORS[alignment.status];
            const session = sessions.find(s => s.id === alignment.sessionId);
            const isExpanded = expandedSources.has(alignment.sourceId);
            const conflictCount = alignment.conflicts.length;

            return (
              <div
                key={alignment.sourceId}
                className={`border rounded-lg overflow-hidden bg-[#101217] border-[${statusColors.border}]/30 ${isExpanded ? 'bg-[#14161f]' : ''}`}
              >
                {/* Source Row */}
                <div className="flex items-center gap-2 p-3">
                  <div className="w-6 h-6 flex items-center justify-center rounded flex-shrink-0" style={{ background: statusColors.bg, color: statusColors.text, border: `1px solid ${statusColors.border}` }}>
                    {alignment.status === 'ALIGNED' && <CheckCircle2 className="w-3.5 h-3.5" />}
                    {alignment.status === 'PENDING' && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
                    {alignment.status === 'CONFLICT' && <AlertTriangle className="w-3.5 h-3.5" />}
                    {alignment.status === 'MERGED' && <GitMerge className="w-3.5 h-3.5" />}
                    {alignment.status === 'REJECTED' && <XCircle className="w-3.5 h-3.5" />}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono font-semibold text-[#f0f1f6]">{SOURCE_TYPE_LABELS[alignment.sourceType]}</span>
                      <span className="text-[10px] font-mono text-[#54596b]">({session?.name || alignment.sessionId})</span>
                      <span className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${ 
                        alignment.status === 'ALIGNED' ? 'bg-[#22c55e]/15 text-[#22c55e] border border-[#22c55e]/40' :
                        alignment.status === 'CONFLICT' ? 'bg-[#ef4444]/15 text-[#ef4444] border border-[#ef4444]/40' :
                        alignment.status === 'PENDING' ? 'bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/40' :
                        alignment.status === 'MERGED' ? 'bg-[#a855f7]/15 text-[#a855f7] border border-[#a855f7]/40' :
                        'bg-[#54596b]/15 text-[#54596b] border border-[#54596b]/40'
                      }`}>
                        {alignment.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-[10px] mt-1">
                      <span className="flex items-center gap-0.5 text-[#9296a6]">
                        <Layers className="w-2.5 h-2.5" />
                        {alignment.gnssAnchors} GNSS anchors
                      </span>
                      <span className="flex items-center gap-0.5 text-[#9296a6]">
                        <GitCompare className="w-2.5 h-2.5" />
                        {alignment.overlapPercent}% overlap
                      </span>
                      {alignment.registrationError && (
                        <span className={`flex items-center gap-0.5 ${alignment.registrationError > 0.1 ? 'text-[#ef4444]' : 'text-[#22c55e]'}`}>
                          <Settings className="w-2.5 h-2.5" />
                          {alignment.registrationError.toFixed(3)}m error
                        </span>
                      )}
                    </div>
                  </div>

                  {conflictCount > 0 && (
                    <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#ef4444]/15 border border-[#ef4444]/40 text-[10px] font-bold text-[#ef4444] shrink-0">
                      <AlertTriangle className="w-2.5 h-2.5" />
                      {conflictCount} conflict{conflictCount > 1 ? 's' : ''}
                    </span>
                  )}

                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      type="button"
                      onClick={() => toggleExpand(alignment.sourceId)}
                      className="p-1 text-[#54596b] hover:text-[#ededf2] transition-colors"
                    >
                      <ChevronDown className={`w-3.5 h-3.5 ${isExpanded ? 'rotate-180' : ''}`} />
                    </button>
                    {alignment.status === 'CONFLICT' && (
                      <button
                        type="button"
                        onClick={() => {
                          if (autoResolve) alignment.conflicts.forEach(c => resolveConflict(c, alignment.sourceId));
                        }}
                        disabled={!autoResolve}
                        className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#22c55e] bg-[#22c55e]/10 border border-[#22c55e]/30 hover:bg-[#22c55e]/20 transition-colors disabled:opacity-50"
                      >
                        <RefreshCw className="w-3 h-3" />
                        <span>Auto-Resolve</span>
                      </button>
                    )}
                    {alignment.status === 'PENDING' && (
                      <button
                        type="button"
                        onClick={() => rejectSource(alignment.sourceId)}
                        className="p-1 rounded text-[#9296a6] hover:text-[#ef4444] hover:bg-[#ef4444]/10 transition-colors"
                        title="Reject source"
                      >
                        <XCircle className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="border-t border-[#1f222b] bg-[#0c0d11] p-3 space-y-3">
                    {/* Transform Details */}
                    {alignment.transform && (
                      <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                        <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b] uppercase mb-1">Translation (m)</div>
                          <div className="text-[#f0f1f6] num-tabular">
                            X: {alignment.transform.translation[0].toFixed(3)}<br/>
                            Y: {alignment.transform.translation[1].toFixed(3)}<br/>
                            Z: {alignment.transform.translation[2].toFixed(3)}
                          </div>
                        </div>
                        <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b] uppercase mb-1">Rotation (wxyz)</div>
                          <div className="text-[#f0f1f6] num-tabular">
                            W: {alignment.transform.rotation[0].toFixed(3)}<br/>
                            X: {alignment.transform.rotation[1].toFixed(3)}<br/>
                            Y: {alignment.transform.rotation[2].toFixed(3)}<br/>
                            Z: {alignment.transform.rotation[3].toFixed(3)}
                          </div>
                        </div>
                        <div className="p-2 rounded bg-[#14161f] border border-[#1f222b]">
                          <div className="text-[9px] text-[#54596b] uppercase mb-1">Scale</div>
                          <div className="text-[#f0f1f6] num-tabular text-xl">{alignment.transform.scale.toFixed(6)}</div>
                        </div>
                      </div>
                    )}

                    {/* Conflicts */}
                    {alignment.conflicts.length > 0 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-mono font-bold text-[#ef4444] uppercase tracking-wider">Alignment Conflicts</span>
                          {autoResolve && (
                            <button
                              type="button"
                              onClick={() => alignment.conflicts.forEach(c => resolveConflict(c, alignment.sourceId))}
                              className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#22c55e] bg-[#22c55e]/10 border border-[#22c55e]/30 hover:bg-[#22c55e]/20"
                            >
                              <RefreshCw className="w-3 h-3" />
                              <span>Resolve All</span>
                            </button>
                          )}
                        </div>
                        {alignment.conflicts.map((conflict, idx) => {
                          const sevColors = CONFLICT_SEVERITY_COLORS[conflict.severity];
                          const SevIcon = sevColors.icon;
                          return (
                            <div key={conflict.id} className="p-2.5 rounded bg-[#14161f] border border-[#1f222b]/50 border-l-4" style={{ borderLeftColor: sevColors.border }}>
                              <div className="flex items-start gap-2">
                                <SevIcon className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" style={{ color: sevColors.text }} />
                                <div className="flex-1 min-w-0">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs font-mono font-semibold text-[#f0f1f6]">{conflict.type}</span>
                                    <span className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${ 
                                      conflict.severity === 'CRITICAL' ? 'bg-[#ef4444]/15 text-[#ef4444] border border-[#ef4444]/40' :
                                      conflict.severity === 'WARNING' ? 'bg-[#f59e0b]/15 text-[#f59e0b] border border-[#f59e0b]/40' :
                                      'bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/40'
                                    }`}>
                                      {conflict.severity}
                                    </span>
                                  </div>
                                  <p className="text-[10px] text-[#9296a6] mt-0.5">{conflict.description}</p>
                                  <p className="text-[9px] text-[#3d8ef7] mt-1 font-mono">→ {conflict.suggestedAction}</p>
                                  {conflict.autoResolvable && (
                                    <button
                                      type="button"
                                      onClick={() => resolveConflict(conflict, alignment.sourceId)}
                                      className="mt-1.5 flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#22c55e] bg-[#22c55e]/10 border border-[#22c55e]/30 hover:bg-[#22c55e]/20"
                                    >
                                      <CheckCircle2 className="w-3 h-3" />
                                      <span>Auto-Resolve</span>
                                    </button>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* CLI Integration */}
                    <div className="p-2 rounded bg-[#14161f] border border-[#1f222b] font-mono text-[10px]">
                      <div className="text-[#54596b] mb-1">Registration CLI:</div>
                      <code className="text-[#3d8ef7]">reality register --source {alignment.sourceId}.ply --target world.ply --anchors anchors.json -o align_{alignment.sourceId}.json</code>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* ── Global Conflicts Panel ── */}
      {allConflicts.length > 0 && (
        <div className="border-t border-[#1f222b] bg-[#101217] p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono font-bold text-[#ef4444] uppercase tracking-wider">All Conflicts ({allConflicts.length})</span>
            {autoResolve && (
              <button
                type="button"
                onClick={() => allConflicts.forEach(c => resolveConflict(c, ''))}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#22c55e] bg-[#22c55e]/10 border border-[#22c55e]/30 hover:bg-[#22c55e]/20"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Auto-Resolve All</span>
              </button>
            )}
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {allConflicts.map((conflict, idx) => {
              const sevColors = CONFLICT_SEVERITY_COLORS[conflict.severity];
              const SevIcon = sevColors.icon;
              return (
                <div key={`${conflict.sourceId}-${idx}`} className="p-2 rounded bg-[#14161f] border border-[#1f222b]/50 border-l-4" style={{ borderLeftColor: sevColors.border }}>
                  <div className="flex items-center gap-2">
                    <SevIcon className="w-3 h-3 flex-shrink-0" style={{ color: sevColors.text }} />
                    <span className="text-xs font-mono text-[#f0f1f6]">{conflict.type}</span>
                    <span className={`px-1 py-0.1 rounded text-[8px] font-bold ${ 
                      conflict.severity === 'CRITICAL' ? 'bg-[#ef4444]/20 text-[#ef4444]' :
                      conflict.severity === 'WARNING' ? 'bg-[#f59e0b]/20 text-[#f59e0b]' :
                      'bg-[#3d8ef7]/20 text-[#3d8ef7]'
                    }`}>
                      {conflict.severity}
                    </span>
                    <span className="text-[9px] text-[#54596b] ml-auto">[{conflict.sourceType}]</span>
                  </div>
                  <p className="text-[9px] text-[#9296a6] mt-0.5 ml-5">{conflict.description}</p>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}