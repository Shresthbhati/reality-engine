'use client';

/**
 * ScaleNavigation — 10-Level Spatial Scale Navigator (ROOM → CITY).
 * - Visual source of truth for global spatial context
 * - Scale-aware UI: adapts LOD, units, navigation behavior per level
 * - Click to jump; drag to scrub; keyboard shortcuts (⌘↑/⌘↓)
 * - Shows coverage status and entity count at each scale
 */

import React, { useState, useCallback, useMemo } from 'react';
import { useREStore, SpatialScale, SpatialScaleInfo, SPATIAL_SCALES } from '@/store/re-store';
import {
  ChevronUp,
  ChevronDown,
  Maximize,
  Minimize,
  Search,
  MapPin,
  Layers,
  ZoomIn,
  ZoomOut,
  Home,
  Grid,
  Building2,
  Map,
  Landmark,
  Castle,
} from 'lucide-react';

const SCALE_ICONS: Record<SpatialScale, React.ReactNode> = {
  ROOM: <Grid className="w-3.5 h-3.5" />,
  BUILDING: <Building2 className="w-3.5 h-3.5" />,
  STREET: <Map className="w-3.5 h-3.5" />,
  PLOT: <MapPin className="w-3.5 h-3.5" />,
  BLOCK: <Layers className="w-3.5 h-3.5" />,
  'MULTI-BLOCK': <Layers className="w-3.5 h-3.5" />,
  LOCALITY: <Landmark className="w-3.5 h-3.5" />,
  WARD: <Castle className="w-3.5 h-3.5" />,
  DISTRICT: <Castle className="w-3.5 h-3.5" />,
  CITY: <Globe className="w-3.5 h-3.5" />,
};

function Globe({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="12" cy="12" r="10" />
      <line x1="2" y1="12" x2="22" y2="12" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

interface ScaleNavigationProps {
  className?: string;
  compact?: boolean;
  showCoverage?: boolean;
  onScaleChange?: (scale: SpatialScale) => void;
}

export function ScaleNavigation({
  className = '',
  compact = false,
  showCoverage = true,
  onScaleChange,
}: ScaleNavigationProps) {
  const {
    spatialScale,
    setSpatialScale,
    entities,
    spatialCoverageNodes,
    activeScaleLevel,
  } = useREStore();

  const [isExpanded, setIsExpanded] = useState(!compact);
  const [hoveredScale, setHoveredScale] = useState<SpatialScale | null>(null);

  const currentScaleInfo = useMemo(() => SPATIAL_SCALES.find(s => s.scale === spatialScale), [spatialScale]);

  const getCoverageAtScale = useCallback((scale: SpatialScale) => {
    const node = spatialCoverageNodes.find(n => n.scaleLevel === scale);
    return node ? { coverage: node.coveragePercent, status: node.status } : { coverage: 0, status: 'UNOBSERVED' as const };
  }, [spatialCoverageNodes]);

  const getEntityCountAtScale = useCallback((scale: SpatialScale) => {
    const scaleToEntityTypes: Record<SpatialScale, string[]> = {
      ROOM: ['ROOM', 'OBJECT', 'COMPONENT', 'CAPITAL', 'ORNAMENT', 'RELIEF'],
      BUILDING: ['BUILDING', 'FLOOR', 'WALL', 'FACADE', 'ROOM'],
      STREET: ['STREET', 'ROAD', 'INFRASTRUCTURE', 'VEGETATION'],
      PLOT: ['SITE', 'BUILDING', 'TERRAIN'],
      BLOCK: ['SITE', 'BUILDING', 'ROAD', 'TERRAIN'],
      'MULTI-BLOCK': ['SITE', 'BUILDING', 'ROAD', 'TERRAIN'],
      LOCALITY: ['SITE', 'BUILDING', 'ROAD', 'TERRAIN', 'VEGETATION'],
      WARD: ['SITE', 'BUILDING', 'ROAD', 'TERRAIN', 'VEGETATION', 'INFRASTRUCTURE'],
      DISTRICT: ['SITE', 'BUILDING', 'ROAD', 'TERRAIN', 'VEGETATION', 'INFRASTRUCTURE'],
      CITY: ['WORLD', 'SITE', 'BUILDING', 'ROAD', 'TERRAIN', 'VEGETATION', 'INFRASTRUCTURE'],
    };
    const types = scaleToEntityTypes[scale] || [];
    return Array.from(entities.values()).filter(e => types.includes(e.type)).length;
  }, [entities]);

  const handleScaleClick = useCallback((scale: SpatialScale) => {
    setSpatialScale(scale);
    onScaleChange?.(scale);
  }, [setSpatialScale, onScaleChange]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();
      const idx = SPATIAL_SCALES.findIndex(s => s.scale === spatialScale);
      if (e.key === 'ArrowUp' && idx < SPATIAL_SCALES.length - 1) {
        handleScaleClick(SPATIAL_SCALES[idx + 1].scale);
      } else if (e.key === 'ArrowDown' && idx > 0) {
        handleScaleClick(SPATIAL_SCALES[idx - 1].scale);
      }
    }
  }, [spatialScale, handleScaleClick]);

  // Attach keyboard listener
  React.useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (compact) {
    return (
      <div className={`flex items-center gap-1 ${className}`} role="group" aria-label="Spatial scale">
        <button
          type="button"
          onClick={() => setIsExpanded(true)}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#14161f] border border-[#1f222b] hover:border-[#3d8ef7]/40 transition-colors text-[#3d8ef7]"
          title="Spatial Scale (⌘↑/⌘↓ to navigate)"
        >
          {SCALE_ICONS[spatialScale]}
          <span className="text-xs font-mono font-bold text-[#f0f1f6]">{spatialScale}</span>
          <ChevronUp className="w-3 h-3 text-[#54596b]" />
        </button>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      {/* ── Collapsed Trigger (when not expanded) ─────────────────────── */}
      {!isExpanded && (
        <button
          type="button"
          onClick={() => setIsExpanded(true)}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-[#14161f] border border-[#1f222b] hover:border-[#3d8ef7]/40 transition-colors text-[#3d8ef7]"
          title="Spatial Scale Navigator (⌘↑/⌘↓)"
        >
          {SCALE_ICONS[spatialScale]}
          <span className="text-xs font-mono font-bold text-[#f0f1f6]">{spatialScale}</span>
          <span className="text-[10px] text-[#54596b]">({currentScaleInfo?.typicalUnits || 'm'})</span>
          <ChevronDown className="w-3 h-3 text-[#54596b]" />
        </button>
      )}

      {/* ── Expanded Scale Navigator ──────────────────────────────────── */}
      {isExpanded && (
        <div
          className="absolute right-0 top-full mt-1 w-64 bg-[#0f1014] border border-[#1f222b] rounded-lg shadow-2xl overflow-hidden z-50 animate-in fade-in-0 zoom-in-95 duration-150"
          role="listbox"
          aria-label="Spatial scales"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-[#1f222b] bg-[#14161f]">
            <span className="text-[10px] font-mono font-bold tracking-wider text-[#54596b] uppercase">Spatial Scale</span>
            <button
              type="button"
              onClick={() => setIsExpanded(false)}
              className="p-1 text-[#9296a6] hover:text-[#ededf2] rounded transition-colors"
              title="Collapse"
            >
              <Minimize className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Scale List */}
          <div className="max-h-[500px] overflow-y-auto p-1">
            {SPATIAL_SCALES.map((scaleInfo, idx) => {
              const isActive = scaleInfo.scale === spatialScale;
              const coverage = getCoverageAtScale(scaleInfo.scale);
              const entityCount = getEntityCountAtScale(scaleInfo.scale);
              const Icon = SCALE_ICONS[scaleInfo.scale];

              const statusColors: Record<string, string> = {
                HIGH: '#22c55e',
                MED: '#eab308',
                LOW: '#ef4444',
                UNOBSERVED: '#54596b',
                INFERRED: '#a855f7',
              };

              return (
                <button
                  key={scaleInfo.scale}
                  type="button"
                  role="option"
                  aria-selected={isActive}
                  onClick={() => handleScaleClick(scaleInfo.scale)}
                  onMouseEnter={() => setHoveredScale(scaleInfo.scale)}
                  onMouseLeave={() => setHoveredScale(null)}
                  className={`w-full flex items-center gap-2 px-2.5 py-2 rounded transition-colors ${
                    isActive
                      ? 'bg-[#3d8ef7]/15 text-[#3d8ef7]'
                      : 'text-[#c4c7d4] hover:bg-[#181b23]'
                  }`}
                >
                  <span className={`w-6 h-6 flex items-center justify-center rounded flex-shrink-0 ${
                    isActive ? 'bg-[#3d8ef7]/20' : 'bg-[#14161f]'
                  }`} style={{ color: isActive ? '#3d8ef7' : '#9296a6' }}>
                    {Icon}
                  </span>

                  <div className="flex-1 min-w-0 text-left">
                    <div className="flex items-center justify-between">
                      <span className={`text-xs font-mono ${isActive ? 'font-bold' : ''} truncate`}>
                        {scaleInfo.scale}
                      </span>
                      {entityCount > 0 && (
                        <span className="text-[9px] font-mono text-[#54596b] num-tabular">
                          {entityCount} ent.
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 text-[9px]">
                      <span className="text-[#54596b]">{scaleInfo.approxDistance}</span>
                      <span className="text-[#3d8ef7]">•</span>
                      <span className="text-[#9296a6] font-mono">{scaleInfo.typicalUnits}</span>
                      {showCoverage && coverage.coverage > 0 && (
                        <span
                          className="flex items-center gap-0.5 px-1.5 py-0.5 rounded"
                          style={{ background: `${statusColors[coverage.status]}20`, color: statusColors[coverage.status] }}
                        >
                          {coverage.coverage}%
                        </span>
                      )}
                    </div>
                  </div>

                  {isActive && (
                    <span className="text-[10px] text-[#3d8ef7] font-bold">▸</span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Footer with shortcuts */}
          <div className="border-t border-[#1f222b] bg-[#14161f] px-3 py-2 text-[9px] text-[#54596b] font-mono">
            <div className="flex items-center justify-between">
              <span>⌘↑ / ⌘↓ navigate</span>
              <span>{spatialScale} • LOD {currentScaleInfo?.defaultLOD}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}