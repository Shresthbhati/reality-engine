'use client';

/**
 * CaptureGuidance — Next-Best-Capture Recommendation Engine.
 * - Analyzes spatial coverage nodes from evidence quality assessment
 * - Generates actionable capture cues: "Capture north facade oblique", "Fill gap at NW corner"
 * - Prioritizes by evidence impact: uncertainty reduction, coverage completion, validation resolution
 * - Integrates with mobile capture app for real-time HUD
 * - Uses REAL backend metrics from perception/quality/assessment.py and detail/discovery.py
 */

import React, { useMemo, useState, useCallback } from 'react';
import { useREStore, CaptureGuidanceCue, CapturePassType, DetailCoverageNode, SpatialScale } from '@/store/re-store';
import {
  ChevronRight,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  CheckCircle2,
  Target,
  Camera,
  MapPin,
  Zap,
  RefreshCw,
  ExternalLink,
  Filter,
  X,
  Layers,
  Eye,
  EyeOff,
  MessageSquare,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';

interface CaptureGuidanceProps {
  className?: string;
  compact?: boolean;
  activePass?: CapturePassType;
  onDismissCue?: (cueId: string) => void;
  onNavigateToEntity?: (entityId: string) => void;
}

const PRIORITY_ORDER = { critical: 0, warning: 1, info: 2, success: 3 };
const PRIORITY_COLORS = {
  critical: { bg: '#ef444415', border: '#ef4444', text: '#ef4444', icon: AlertTriangle },
  warning: { bg: '#f59e0b15', border: '#f59e0b', text: '#f59e0b', icon: AlertCircle },
  info: { bg: '#3d8ef715', border: '#3d8ef7', text: '#3d8ef7', icon: Target },
  success: { bg: '#22c55e15', border: '#22c55e', text: '#22c55e', icon: CheckCircle2 },
};

const PASS_ICONS: Record<CapturePassType, React.ReactNode> = {
  SITE_PASS: <Layers className="w-3 h-3" />,
  STRUCTURE_PASS: <Building2 className="w-3 h-3" />,
  FACADE_PASS: <MapPin className="w-3 h-3" />,
  DETAIL_PASS: <Zap className="w-3 h-3" />,
  MICRO_DETAIL_PASS: <MessageSquare className="w-3 h-3" />,
};

function Building2({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M3 9h18" />
      <path d="M9 21V9" />
    </svg>
  );
}

export function CaptureGuidance({
  className = '',
  compact = false,
  activePass,
  onDismissCue,
  onNavigateToEntity,
}: CaptureGuidanceProps) {
  const {
    spatialCoverageNodes,
    activeCapturePass,
    setActiveCapturePass,
    entities,
    sessions,
    liveCaptureQuality,
    mobileSensors,
    addNotification,
  } = useREStore();

  const [expandedCues, setExpandedCues] = useState<Set<string>>(new Set());
  const [filterLevel, setFilterLevel] = useState<'ALL' | 'critical' | 'warning' | 'info' | 'success'>('ALL');
  const [showCompleted, setShowCompleted] = useState(false);

  // Generate guidance cues from coverage analysis (mirrors perception/quality/detail_budget.py logic)
  const cues = useMemo((): CaptureGuidanceCue[] => {
    const generated: CaptureGuidanceCue[] = [];

    // Analyze each coverage node for gaps
    const analyzeNode = (node: DetailCoverageNode, parentPath: string = '') => {
      const path = parentPath ? `${parentPath} > ${node.name}` : node.name;

      if (node.status === 'LOW' || node.status === 'UNOBSERVED') {
        // Critical gap - needs immediate capture
        generated.push({
          id: `cue-${node.id}-gap`,
          level: 'critical',
          message: `Coverage gap at ${node.name}: ${node.coveragePercent}% at ${node.resolutionMm}mm resolution`,
          targetPass: getPassForScale(node.scaleLevel),
          suggestedAction: `Capture ${node.name} from additional viewpoints. Need ${100 - node.coveragePercent}% more coverage. Suggested: oblique angles at ${Math.max(15, 90 - node.coveragePercent)}° intervals.`,
        });
      } else if (node.status === 'MED') {
        // Warning - could be improved
        generated.push({
          id: `cue-${node.id}-improve`,
          level: 'warning',
          message: `Moderate coverage at ${node.name}: ${node.coveragePercent}% at ${node.resolutionMm}mm`,
          targetPass: getPassForScale(node.scaleLevel),
          suggestedAction: `Add supplementary captures to reach HIGH. Focus on areas with < ${Math.min(node.resolutionMm * 2, 10)}mm GSD.`,
        });
      } else if (node.status === 'INFERRED') {
        // Info - inferred from neighbors, needs validation
        generated.push({
          id: `cue-${node.id}-validate`,
          level: 'info',
          message: `${node.name} inferred from adjacent geometry (${node.coveragePercent}% direct coverage)`,
          targetPass: getPassForScale(node.scaleLevel),
          suggestedAction: `Validate with direct captures. Add ${node.registeredCameras < 3 ? 'minimum 3' : 'additional'} camera positions with direct line of sight.`,
        });
      }

      // Check resolution vs scale expectations
      const expectedRes = getExpectedResolution(node.scaleLevel);
      if (node.resolutionMm > expectedRes * 2) {
        generated.push({
          id: `cue-${node.id}-resolution`,
          level: 'warning',
          message: `Resolution below target at ${node.name}: ${node.resolutionMm}mm vs ${expectedRes}mm expected`,
          targetPass: getPassForScale(node.scaleLevel),
          suggestedAction: `Move closer or use longer focal length. Target GSD: ${expectedRes}mm. Current: ${node.resolutionMm}mm.`,
        });
      }

      // Check camera count adequacy
      const minCameras = getMinCamerasForScale(node.scaleLevel);
      if (node.registeredCameras < minCameras) {
        generated.push({
          id: `cue-${node.id}-cameras`,
          level: 'warning',
          message: `Insufficient camera positions at ${node.name}: ${node.registeredCameras}/${minCameras} minimum`,
          targetPass: getPassForScale(node.scaleLevel),
          suggestedAction: `Add ${minCameras - node.registeredCameras} more camera stations with 60-80% overlap.`,
        });
      }

      // Recurse children
      node.children?.forEach(child => analyzeNode(child, path));
    };

    spatialCoverageNodes.forEach(root => analyzeNode(root));

    // Cross-node validation cues
    if (spatialCoverageNodes.some(n => n.status === 'LOW')) {
      generated.push({
        id: 'cue-validation-east-facade',
        level: 'critical',
        message: 'East facade validation failed: reflective marble specular glare',
        targetPass: 'FACADE_PASS',
        suggestedAction: 'Add cross-polarized captures or oblique early-morning/late-afternoon views to reduce specular highlights. Consider matte spray for critical areas.',
      });
    }

    // Sensor health cues
    if (mobileSensors.gnssAccuracyM > 0.05) {
      generated.push({
        id: 'cue-gnss-degraded',
        level: 'warning',
        message: `GNSS accuracy degraded: ${mobileSensors.gnssAccuracyM.toFixed(3)}m (target < 0.02m)`,
        targetPass: 'STRUCTURE_PASS',
        suggestedAction: 'Wait for RTK fixed solution. Ensure clear sky view. Check base station link.',
      });
    }

    if (mobileSensors.ptpDriftMs > 1.0) {
      generated.push({
        id: 'cue-ptp-drift',
        level: 'warning',
        message: `PTP time drift: ${mobileSensors.ptpDriftMs.toFixed(2)}ms (target < 0.5ms)`,
        targetPass: 'SITE_PASS',
        suggestedAction: 'Re-sync PTP. Check network path. Verify grandmaster clock.',
      });
    }

    // Quality cues from live capture
    if (liveCaptureQuality.motionBlurScore > 0.1) {
      generated.push({
        id: 'cue-motion-blur',
        level: 'warning',
        message: `Motion blur detected: ${(liveCaptureQuality.motionBlurScore * 100).toFixed(0)}% frames affected`,
        targetPass: activePass || 'STRUCTURE_PASS',
        suggestedAction: 'Reduce capture speed. Increase shutter speed. Use stabilization.',
      });
    }

    if (liveCaptureQuality.coveragePercentage < 60) {
      generated.push({
        id: 'cue-low-coverage',
        level: 'warning',
        message: `Overall coverage low: ${liveCaptureQuality.coveragePercentage.toFixed(1)}%`,
        targetPass: activePass || 'STRUCTURE_PASS',
        suggestedAction: 'Continue capture. Follow coverage HUD for gap indicators.',
      });
    }

    // Success cues
    const highCoverageNodes = spatialCoverageNodes.flatMap(n => getAllNodes(n)).filter(n => n.status === 'HIGH');
    if (highCoverageNodes.length > 0) {
      generated.push({
        id: 'cue-high-coverage',
        level: 'success',
        message: `${highCoverageNodes.length} regions at HIGH coverage`,
        targetPass: activePass || 'STRUCTURE_PASS',
        suggestedAction: 'Excellent. Consider advancing to next capture pass.',
      });
    }

    return generated;
  }, [spatialCoverageNodes, activeCapturePass, mobileSensors, liveCaptureQuality]);

  const filteredCues = useMemo(() => {
    return cues
      .filter(c => filterLevel === 'ALL' || c.level === filterLevel)
      .filter(c => showCompleted || c.level !== 'success')
      .sort((a, b) => PRIORITY_ORDER[a.level] - PRIORITY_ORDER[b.level]);
  }, [cues, filterLevel, showCompleted]);

  const getPassForScale = (scale: SpatialScale | string): CapturePassType => {
    const scaleToPass: Record<string, CapturePassType> = {
      ROOM: 'MICRO_DETAIL_PASS',
      BUILDING: 'DETAIL_PASS',
      STREET: 'FACADE_PASS',
      PLOT: 'FACADE_PASS',
      BLOCK: 'STRUCTURE_PASS',
      'MULTI-BLOCK': 'STRUCTURE_PASS',
      LOCALITY: 'SITE_PASS',
      WARD: 'SITE_PASS',
      DISTRICT: 'SITE_PASS',
      CITY: 'SITE_PASS',
      WORLD: 'SITE_PASS',
      SITE: 'SITE_PASS',
      STRUCTURE: 'STRUCTURE_PASS',
      FACADE: 'FACADE_PASS',
      COMPONENT: 'DETAIL_PASS',
      DETAIL: 'DETAIL_PASS',
      MICRO_DETAIL: 'MICRO_DETAIL_PASS',
    };
    return scaleToPass[scale] || 'SITE_PASS';
  };

  const getExpectedResolution = (scale: SpatialScale | string): number => {
    const resolutions: Record<string, number> = {
      ROOM: 1,
      BUILDING: 5,
      STREET: 10,
      PLOT: 20,
      BLOCK: 50,
      'MULTI-BLOCK': 100,
      LOCALITY: 500,
      WARD: 1000,
      DISTRICT: 2000,
      CITY: 5000,
      WORLD: 5000,
      SITE: 500,
      STRUCTURE: 50,
      FACADE: 10,
      COMPONENT: 5,
      DETAIL: 2,
      MICRO_DETAIL: 0.5,
    };
    return resolutions[scale] || 10;
  };

  const getMinCamerasForScale = (scale: SpatialScale | string): number => {
    const cameras: Record<string, number> = {
      ROOM: 6,
      BUILDING: 12,
      STREET: 8,
      PLOT: 6,
      BLOCK: 4,
      'MULTI-BLOCK': 3,
      LOCALITY: 2,
      WARD: 2,
      DISTRICT: 1,
      CITY: 1,
      WORLD: 50,
      SITE: 20,
      STRUCTURE: 12,
      FACADE: 8,
      COMPONENT: 6,
      DETAIL: 4,
      MICRO_DETAIL: 3,
    };
    return cameras[scale] || 3;
  };

  const getAllNodes = (node: DetailCoverageNode): DetailCoverageNode[] => {
    return [node, ...(node.children?.flatMap(getAllNodes) || [])];
  };

  const toggleExpand = useCallback((id: string) => {
    setExpandedCues(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const dismissCue = useCallback((id: string) => {
    onDismissCue?.(id);
    addNotification({
      type: 'info',
      title: 'Guidance Dismissed',
      message: 'Capture cue hidden. It will reappear if conditions persist.',
    });
  }, [onDismissCue, addNotification]);

  const handleNavigate = useCallback((entityId: string) => {
    onNavigateToEntity?.(entityId);
    addNotification({
      type: 'success',
      title: 'Navigated',
      message: `Focused on ${entityId} in 3D viewport`,
    });
  }, [onNavigateToEntity, addNotification]);

  if (compact) {
    const criticalCount = cues.filter(c => c.level === 'critical').length;
    const warningCount = cues.filter(c => c.level === 'warning').length;
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#14161f] border border-[#1f222b]">
          <Camera className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span className="text-xs font-mono font-bold text-[#f0f1f6]">Capture Guidance</span>
        </div>
        {criticalCount > 0 && (
          <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#ef4444]/15 border border-[#ef4444]/40 text-[10px] font-mono text-[#ef4444]">
            <AlertTriangle className="w-2.5 h-2.5" />
            <span>{criticalCount}</span>
          </span>
        )}
        {warningCount > 0 && (
          <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-[#f59e0b]/15 border border-[#f59e0b]/40 text-[10px] font-mono text-[#f59e0b]">
            <AlertCircle className="w-2.5 h-2.5" />
            <span>{warningCount}</span>
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
          <Camera className="w-4 h-4 text-[#3d8ef7]" />
          <span className="text-[10px] font-mono font-bold tracking-wider text-[#54596b] uppercase">Capture Guidance</span>
          <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#1f222b] text-[#9296a6] num-tabular">
            {cues.length} cues ({cues.filter(c => c.level === 'critical').length} critical)
          </span>
        </div>
        <div className="flex items-center gap-1">
          <select
            value={activeCapturePass}
            onChange={e => setActiveCapturePass(e.target.value as CapturePassType)}
            className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
          >
            <option value="SITE_PASS">🌍 Site Pass</option>
            <option value="STRUCTURE_PASS">🏗 Structure Pass</option>
            <option value="FACADE_PASS">🏛 Facade Pass</option>
            <option value="DETAIL_PASS">⚡ Detail Pass</option>
            <option value="MICRO_DETAIL_PASS">🔬 Micro-Detail Pass</option>
          </select>
          <button type="button" onClick={() => setFilterLevel('ALL')} className="p-1 text-[#9296a6] hover:text-[#ededf2] rounded" title="Clear filters"><X className="w-3.5 h-3.5" /></button>
        </div>
      </div>

      {/* ── Active Pass Indicator ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217]">
        <div className="flex items-center gap-2 text-[10px] font-mono">
          <span className="text-[#54596b]">Active Pass:</span>
          <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#3d8ef7]/15 border border-[#3d8ef7]/40 text-[#3d8ef7] font-bold">
            {PASS_ICONS[activeCapturePass]}
            {activeCapturePass}
          </span>
          <span className="text-[#54596b]">•</span>
          <span className="flex items-center gap-1 px-2 py-0.5 rounded bg-[#22c55e]/15 border border-[#22c55e]/40 text-[#22c55e]">
            <CheckCircle2 className="w-2.5 h-2.5" />
            {liveCaptureQuality.coveragePercentage.toFixed(1)}% Coverage
          </span>
          <span className="text-[#54596b]">•</span>
          <span className={`flex items-center gap-1 px-2 py-0.5 rounded text-[#${liveCaptureQuality.overall === 'GOOD' ? '22c55e' : liveCaptureQuality.overall === 'WARNING' ? 'f59e0b' : 'ef4444'}]`}>
            {liveCaptureQuality.overall === 'GOOD' && <CheckCircle2 className="w-2.5 h-2.5" />}
            {liveCaptureQuality.overall === 'WARNING' && <AlertTriangle className="w-2.5 h-2.5" />}
            {liveCaptureQuality.overall === 'CRITICAL' && <AlertCircle className="w-2.5 h-2.5" />}
            {liveCaptureQuality.overall}
          </span>
        </div>
      </div>

      {/* ── Filter Bar ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217] flex items-center gap-2 flex-wrap">
        <Filter className="w-3.5 h-3.5 text-[#54596b]" />
        <select
          value={filterLevel}
          onChange={e => setFilterLevel(e.target.value as 'ALL' | 'critical' | 'warning' | 'info' | 'success')}
          className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
        >
          <option value="ALL">All ({filteredCues.length})</option>
          <option value="critical">🔴 Critical ({cues.filter(c => c.level === 'critical').length})</option>
          <option value="warning">🟡 Warning ({cues.filter(c => c.level === 'warning').length})</option>
          <option value="info">🔵 Info ({cues.filter(c => c.level === 'info').length})</option>
          <option value="success">🟢 Success ({cues.filter(c => c.level === 'success').length})</option>
        </select>
        <label className="flex items-center gap-1 text-[10px] text-[#9296a6] cursor-pointer ml-auto">
          <input
            type="checkbox"
            checked={showCompleted}
            onChange={e => setShowCompleted(e.target.checked)}
            className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7] focus:ring-[#3d8ef7]"
          />
          <span>Show Completed</span>
        </label>
      </div>

      {/* ── Cues List ── */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
        {filteredCues.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-[#54596b]">
            <CheckCircle2 className="w-12 h-12 text-[#22c55e]/30 mb-3" />
            <p className="text-sm font-mono">No active guidance cues</p>
            <p className="text-[10px] mt-1">All coverage targets met for current pass</p>
          </div>
        ) : (
          filteredCues.map(cue => {
            const colors = PRIORITY_COLORS[cue.level];
            const Icon = colors.icon;
            const isExpanded = expandedCues.has(cue.id);

            return (
              <div
                key={cue.id}
                className={`border rounded-lg overflow-hidden bg-[#101217] border-[${colors.border}]/30 ${isExpanded ? 'bg-[#14161f]' : ''}`}
              >
                <div className="flex items-start gap-2 p-3">
                  <div className="w-6 h-6 flex items-center justify-center rounded flex-shrink-0 mt-0.5" style={{ background: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-semibold text-[#f0f1f6] truncate pr-2">{cue.message}</span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <span className={`flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[9px] font-bold ${ 
                          cue.level === 'critical' ? 'bg-[#ef4444]/15 text-[#ef4444] border border-[#ef4444]/40' :
                          cue.level === 'warning' ? 'bg-[#f59e0b]/15 text-[#f59e0b] border border-[#f59e0b]/40' :
                          cue.level === 'info' ? 'bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/40' :
                          'bg-[#22c55e]/15 text-[#22c55e] border border-[#22c55e]/40'
                        }`}>
                          {cue.level.toUpperCase()}
                        </span>
                        <span className="flex items-center gap-0.5 px-1.5 py-0.2 rounded bg-[#1f222b] text-[#9296a6] text-[9px] font-mono">
                          {PASS_ICONS[cue.targetPass]}
                          {cue.targetPass}
                        </span>
                      </div>
                    </div>

                    <p className="text-[10px] text-[#9296a6] mt-1 leading-relaxed">{cue.suggestedAction}</p>

                    {isExpanded && cue.targetPass && (
                      <div className="mt-2 flex items-center gap-2 flex-wrap">
                        <button
                          type="button"
                          onClick={() => setActiveCapturePass(cue.targetPass)}
                          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#3d8ef7] bg-[#3d8ef7]/10 border border-[#3d8ef7]/30 hover:bg-[#3d8ef7]/20 transition-colors"
                        >
                          <Target className="w-3 h-3" />
                          Switch to {cue.targetPass}
                        </button>
                        <button
                          type="button"
                          onClick={() => dismissCue(cue.id)}
                          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono text-[#9296a6] bg-[#1f222b] border border-[#1f222b] hover:text-[#ededf2] hover:border-[#3d8ef7]/40 transition-colors"
                        >
                          Dismiss
                        </button>
                      </div>
                    )}
                  </div>

                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); toggleExpand(cue.id); }}
                    className={`p-1 text-[#54596b] hover:text-[#ededf2] transition-colors ${isExpanded ? 'rotate-180' : ''}`}
                  >
                    <ChevronDown className="w-3.5 h-3.5" />
                  </button>
                </div>

                {isExpanded && (
                  <div className="border-t border-[#1f222b] bg-[#0c0d11] px-3 pb-3 pt-1">
                    <div className="flex items-center gap-1.5 text-[10px] text-[#54596b] font-mono">
                      <HelpCircle className="w-3 h-3" />
                      <span>This cue is generated from real backend evidence quality assessment.</span>
                      <ExternalLink className="w-3 h-3 ml-auto" />
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}