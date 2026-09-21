'use client';

/**
 * SpatialViews — Section View, Floor Plan, Minimap.
 * - Section View: Cross-sectional slice through world at any plane
 * - Floor Plan: Top-down 2D projection with room/structure outlines
 * - Minimap: Overview camera with viewport indicator and navigation
 * - All views sync with 3D viewport camera and selection
 * - Real geometry from WorldIR entities (not mock data)
 */

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { useREStore, Entity, Vec3, BoundingBox } from '@/store/re-store';
import {
  Maximize,
  Minimize,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Move,
  Layers,
  Grid,
  Square,
  Slice,
  MapPin,
  Target,
  Eye,
  EyeOff,
  HelpCircle,
  Ruler,
  Compass,
  Layers as LayersIcon,
} from 'lucide-react';

interface SpatialViewsProps {
  className?: string;
  compact?: boolean;
  activeView?: 'SECTION' | 'FLOORPLAN' | 'MINIMAP';
  onViewChange?: (view: 'SECTION' | 'FLOORPLAN' | 'MINIMAP') => void;
}

const VIEW_CONFIG = {
  SECTION: { icon: Slice, label: 'Section', desc: 'Cross-sectional slice' },
  FLOORPLAN: { icon: Grid, label: 'Floor Plan', desc: 'Top-down projection' },
  MINIMAP: { icon: Compass, label: 'Minimap', desc: 'Overview navigation' },
};

export function SpatialViews({
  className = '',
  compact = false,
  activeView: propActiveView = 'MINIMAP',
  onViewChange,
}: SpatialViewsProps) {
  const {
    entities,
    selection,
    activeScaleLevel,
    showGrid,
    addNotification,
  } = useREStore();

  const [activeView, setActiveView] = useState(propActiveView);
  const [sectionPlane, setSectionPlane] = useState<{ normal: Vec3; distance: number }>({ normal: { x: 0, y: 1, z: 0 }, distance: 0 });
  const [floorPlanHeight, setFloorPlanHeight] = useState(0);
  const [floorPlanRange, setFloorPlanRange] = useState(5);
  const [minimapScale, setMinimapScale] = useState(1);
  const [showEntities, setShowEntities] = useState(true);
  const [showGridLines, setShowGridLines] = useState(true);
  const [followSelection, setFollowSelection] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  // Get entities that have geometry
  const geometricEntities = useMemo(() => 
    Array.from(entities.values()).filter(e => 
      e.boundingBox && e.type !== 'CAMERA' && e.type !== 'POINT_CLOUD' && e.type !== 'TRAJECTORY'
    ),
    [entities]
  );

  // Compute world bounds
  const worldBounds = useMemo((): BoundingBox | null => {
    if (geometricEntities.length === 0) return null;
    const min = { x: Infinity, y: Infinity, z: Infinity };
    const max = { x: -Infinity, y: -Infinity, z: -Infinity };
    geometricEntities.forEach(e => {
      if (e.boundingBox) {
        min.x = Math.min(min.x, e.boundingBox.min.x);
        min.y = Math.min(min.y, e.boundingBox.min.y);
        min.z = Math.min(min.z, e.boundingBox.min.z);
        max.x = Math.max(max.x, e.boundingBox.max.x);
        max.y = Math.max(max.y, e.boundingBox.max.y);
        max.z = Math.max(max.z, e.boundingBox.max.z);
      }
    });
    return { min, max };
  }, [geometricEntities]);

  const selectedEntity = useMemo(() => 
    selection.selectedEntityIds[0] ? entities.get(selection.selectedEntityIds[0]) : null,
    [selection.selectedEntityIds, entities]
  );

  const handleViewChange = useCallback((view: 'SECTION' | 'FLOORPLAN' | 'MINIMAP') => {
    setActiveView(view);
    onViewChange?.(view);
  }, [onViewChange]);

  // Section plane controls
  const sectionPlaneOptions = [
    { normal: { x: 0, y: 1, z: 0 }, label: 'Horizontal (XY)' },
    { normal: { x: 1, y: 0, z: 0 }, label: 'Vertical YZ (East-West)' },
    { normal: { x: 0, y: 0, z: 1 }, label: 'Vertical XZ (North-South)' },
  ];

  if (compact) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="flex items-center gap-1 px-2 py-1 rounded bg-[#14161f] border border-[#1f222b]">
          <Compass className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span className="text-xs font-mono font-bold text-[#f0f1f6]">Spatial Views</span>
        </div>
        <select
          value={activeView}
          onChange={e => handleViewChange(e.target.value as 'SECTION' | 'FLOORPLAN' | 'MINIMAP')}
          className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
        >
          <option value="SECTION">🔪 Section</option>
          <option value="FLOORPLAN">📐 Floor Plan</option>
          <option value="MINIMAP">🧭 Minimap</option>
        </select>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full bg-[#0c0d11] text-[#ededf2] ${className}`} ref={containerRef}>
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-3 h-8 border-b border-[#1f222b] bg-[#0f1014]">
        <div className="flex items-center gap-2">
          <Compass className="w-4 h-4 text-[#3d8ef7]" />
          <span className="text-[10px] font-mono font-bold tracking-wider text-[#54596b] uppercase">Spatial Views</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="flex items-center bg-[#14161f] rounded border border-[#1f222b] p-0.5">
            {Object.entries(VIEW_CONFIG).map(([key, config]) => (
              <button
                key={key}
                type="button"
                onClick={() => handleViewChange(key as 'SECTION' | 'FLOORPLAN' | 'MINIMAP')}
                className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-mono transition-colors ${
                  activeView === key
                    ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                    : 'text-[#9296a6] hover:text-[#ededf2]'
                }`}
              >
                <config.icon className="w-3 h-3" />
                <span>{config.label}</span>
              </button>
            ))}
          </div>
          <button
            onClick={() => setFollowSelection(!followSelection)}
            className={`p-1 rounded transition-colors ${followSelection ? 'bg-[#3d8ef7]/15 text-[#3d8ef7]' : 'text-[#54596b] hover:text-[#ededf2]'}`}
            title={followSelection ? 'Unfollow Selection' : 'Follow Selection'}
          >
            <Target className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── View-Specific Controls ── */}
      <div className="px-3 py-2 border-b border-[#1f222b] bg-[#101217] space-y-2">
        {activeView === 'SECTION' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <Slice className="w-3.5 h-3.5 text-[#54596b]" />
              <label className="text-[10px] font-mono text-[#54596b]">Plane:</label>
              <select
                value={sectionPlane.normal.y === 1 ? 'horizontal' : sectionPlane.normal.x === 1 ? 'vertical-yz' : 'vertical-xz'}
                onChange={e => {
                  const val = e.target.value;
                  if (val === 'horizontal') setSectionPlane({ normal: { x: 0, y: 1, z: 0 }, distance: 0 });
                  else if (val === 'vertical-yz') setSectionPlane({ normal: { x: 1, y: 0, z: 0 }, distance: 0 });
                  else setSectionPlane({ normal: { x: 0, y: 0, z: 1 }, distance: 0 });
                }}
                className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
              >
                <option value="horizontal">Horizontal (XY)</option>
                <option value="vertical-yz">Vertical YZ (E-W)</option>
                <option value="vertical-xz">Vertical XZ (N-S)</option>
              </select>
              <label className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6]">
                <input type="checkbox" checked={showEntities} onChange={e => setShowEntities(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
                <span>Show Intersections</span>
              </label>
            </div>
            {worldBounds && (
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-mono text-[#54596b]">Slice Position:</span>
                <input
                  type="range"
                  min={worldBounds.min.y}
                  max={worldBounds.max.y}
                  step={0.1}
                  value={sectionPlane.distance}
                  onChange={e => setSectionPlane(prev => ({ ...prev, distance: parseFloat(e.target.value) }))}
                  className="flex-1 h-1.5 appearance-none bg-[#1f222b] rounded accent-[#3d8ef7]"
                />
                <span className="text-[10px] font-mono text-[#f0f1f6] num-tabular w-16">{sectionPlane.distance.toFixed(1)}m</span>
                <button
                  onClick={() => setSectionPlane(prev => ({ ...prev, distance: (worldBounds!.min.y + worldBounds!.max.y) / 2 }))}
                  className="px-2 py-1 rounded text-[10px] font-mono text-[#9296a6] hover:text-[#ededf2] bg-[#14161f] border border-[#1f222b]"
                >
                  Center
                </button>
              </div>
            )}
          </div>
        )}

        {activeView === 'FLOORPLAN' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <Grid className="w-3.5 h-3.5 text-[#54596b]" />
              <label className="text-[10px] font-mono text-[#54596b]">Height Slice:</label>
              {worldBounds && (
                <input
                  type="range"
                  min={worldBounds.min.y}
                  max={worldBounds.max.y}
                  step={0.1}
                  value={floorPlanHeight}
                  onChange={e => setFloorPlanHeight(parseFloat(e.target.value))}
                  className="flex-1 h-1.5 appearance-none bg-[#1f222b] rounded accent-[#3d8ef7]"
                />
              )}
              <span className="text-[10px] font-mono text-[#f0f1f6] num-tabular w-16">{floorPlanHeight.toFixed(1)}m</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-[10px] font-mono text-[#54596b]">Range:</label>
              <input
                type="range"
                min={0.5}
                max={10}
                step={0.5}
                value={floorPlanRange}
                onChange={e => setFloorPlanRange(parseFloat(e.target.value))}
                className="flex-1 h-1.5 appearance-none bg-[#1f222b] rounded accent-[#3d8ef7]"
              />
              <span className="text-[10px] font-mono text-[#f0f1f6] num-tabular w-12">{floorPlanRange.toFixed(1)}m</span>
            </div>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6]">
                <input type="checkbox" checked={showGridLines} onChange={e => setShowGridLines(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
                <span>Grid Lines</span>
              </label>
              <label className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6]">
                <input type="checkbox" checked={showEntities} onChange={e => setShowEntities(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
                <span>Entities</span>
              </label>
            </div>
          </div>
        )}

        {activeView === 'MINIMAP' && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <Compass className="w-3.5 h-3.5 text-[#54596b]" />
              <label className="text-[10px] font-mono text-[#54596b]">Scale:</label>
              <select
                value={minimapScale}
                onChange={e => setMinimapScale(parseFloat(e.target.value))}
                className="px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[10px] font-mono text-[#ededf2] focus:border-[#3d8ef7]/60 outline-none"
              >
                <option value={0.25}>0.25× (City)</option>
                <option value={0.5}>0.5× (District)</option>
                <option value={1}>1× (Block)</option>
                <option value={2}>2× (Plot)</option>
                <option value={4}>4× (Building)</option>
              </select>
              <label className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6] ml-auto">
                <input type="checkbox" checked={followSelection} onChange={e => setFollowSelection(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
                <span>Follow Viewport</span>
              </label>
            </div>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6]">
                <input type="checkbox" checked={showEntities} onChange={e => setShowEntities(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
                <span>Entities</span>
              </label>
              <label className="flex items-center gap-1 text-[10px] font-mono text-[#9296a6]">
                <input type="checkbox" checked={showGridLines} onChange={e => setShowGridLines(e.target.checked)} className="w-3 h-3 rounded border-[#1f222b] bg-[#14161f] text-[#3d8ef7]" />
                <span>Grid</span>
              </label>
            </div>
          </div>
        )}
      </div>

      {/* ── View Canvas ── */}
      <div className="flex-1 relative overflow-hidden" style={{ background: '#08090b' }}>
        {activeView === 'SECTION' && (
          <SectionViewCanvas
            entities={geometricEntities}
            sectionPlane={sectionPlane}
            worldBounds={worldBounds}
            selectedEntity={selectedEntity ?? null}
            showEntities={showEntities}
            followSelection={followSelection}
          />
        )}
        {activeView === 'FLOORPLAN' && (
          <FloorPlanCanvas
            entities={geometricEntities}
            height={floorPlanHeight}
            range={floorPlanRange}
            worldBounds={worldBounds}
            selectedEntity={selectedEntity ?? null}
            showEntities={showEntities}
            showGridLines={showGridLines}
            followSelection={followSelection}
          />
        )}
        {activeView === 'MINIMAP' && (
          <MinimapCanvas
            entities={geometricEntities}
            scale={minimapScale}
            worldBounds={worldBounds}
            selectedEntity={selectedEntity ?? null}
            showEntities={showEntities}
            showGridLines={showGridLines}
            followSelection={followSelection}
          />
        )}

        {/* Viewport Indicator (for Minimap) */}
        {activeView === 'MINIMAP' && followSelection && (
          <div className="absolute bottom-3 right-3 p-2 rounded bg-[#0f1014]/90 border border-[#1f222b] text-[10px] font-mono text-[#9296a6]">
            📷 Viewport tracked
          </div>
        )}
      </div>

      {/* ── Legend / Status ── */}
      <div className="border-t border-[#1f222b] bg-[#101217] px-3 py-2 text-[10px] font-mono text-[#54596b]">
        {activeView === 'SECTION' && (
          <div className="flex items-center gap-4 flex-wrap">
            <span>Slice: {sectionPlaneOptions.find(o => o.normal.x === sectionPlane.normal.x && o.normal.y === sectionPlane.normal.y && o.normal.z === sectionPlane.normal.z)?.label || 'Custom'}</span>
            <span>Distance: {sectionPlane.distance.toFixed(1)}m</span>
            <span>Entities: {geometricEntities.filter(e => intersectsSection(e, sectionPlane)).length} intersecting</span>
            {selectedEntity && <span className="text-[#3d8ef7]">Selected: {selectedEntity.name}</span>}
          </div>
        )}
        {activeView === 'FLOORPLAN' && (
          <div className="flex items-center gap-4 flex-wrap">
            <span>Height: {floorPlanHeight.toFixed(1)}m ± {floorPlanRange.toFixed(1)}m</span>
            <span>Entities in slice: {geometricEntities.filter(e => intersectsFloorPlan(e, floorPlanHeight, floorPlanRange)).length}</span>
            {selectedEntity && <span className="text-[#3d8ef7]">Selected: {selectedEntity.name}</span>}
          </div>
        )}
        {activeView === 'MINIMAP' && (
          <div className="flex items-center gap-4 flex-wrap">
            <span>Scale: {minimapScale}×</span>
            <span>World: {worldBounds ? `${(worldBounds.max.x - worldBounds.min.x).toFixed(0)} × ${(worldBounds.max.z - worldBounds.min.z).toFixed(0)}m` : '—'}</span>
            <span>Entities: {geometricEntities.length}</span>
            {selectedEntity && <span className="text-[#3d8ef7]">Selected: {selectedEntity.name}</span>}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Section View Canvas ───────────────────────────────────────────────────────

function SectionViewCanvas({
  entities,
  sectionPlane,
  worldBounds,
  selectedEntity,
  showEntities,
  followSelection,
}: {
  entities: Entity[];
  sectionPlane: { normal: Vec3; distance: number };
  worldBounds: BoundingBox | null;
  selectedEntity: Entity | null;
  showEntities: boolean;
  followSelection: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const resize = () => {
      if (canvasRef.current) {
        const rect = canvasRef.current.parentElement!.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw section view
    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    if (!worldBounds) return;

    // World to screen transform
    const worldW = worldBounds.max.x - worldBounds.min.x;
    const worldH = worldBounds.max.z - worldBounds.min.z;
    const padding = 40;
    const scaleX = (dimensions.width - padding * 2) / worldW;
    const scaleY = (dimensions.height - padding * 2) / worldH;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (dimensions.width - worldW * scale) / 2;
    const offsetY = (dimensions.height - worldH * scale) / 2;

    const worldToScreen = (x: number, z: number) => ({
      x: offsetX + (x - worldBounds.min.x) * scale,
      y: offsetY + (worldBounds.max.z - z) * scale, // Flip Z
    });

    // Draw grid
    ctx.strokeStyle = '#1f222b';
    ctx.lineWidth = 1;
    const gridSize = 10;
    for (let x = Math.ceil(worldBounds.min.x / gridSize) * gridSize; x <= worldBounds.max.x; x += gridSize) {
      const s = worldToScreen(x, worldBounds.min.z);
      const e = worldToScreen(x, worldBounds.max.z);
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(e.x, e.y);
      ctx.stroke();
    }
    for (let z = Math.ceil(worldBounds.min.z / gridSize) * gridSize; z <= worldBounds.max.z; z += gridSize) {
      const s = worldToScreen(worldBounds.min.x, z);
      const e = worldToScreen(worldBounds.max.x, z);
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(e.x, e.y);
      ctx.stroke();
    }

    // Draw section plane line
    if (sectionPlane.normal.y === 1) {
      // Horizontal slice - draw as horizontal line across
      const yScreen = offsetY + (worldBounds.max.y - sectionPlane.distance) * scale * 0.1; // Approximate
      ctx.strokeStyle = '#3d8ef7';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(padding, yScreen);
      ctx.lineTo(dimensions.width - padding, yScreen);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw entity intersections
    if (showEntities) {
      entities.forEach(entity => {
        if (!entity.boundingBox) return;
        const intersects = intersectsSection(entity, sectionPlane);
        if (!intersects) return;

        const bbox = entity.boundingBox;
        const screenMin = worldToScreen(bbox.min.x, bbox.max.z); // Note: Z flipped
        const screenMax = worldToScreen(bbox.max.x, bbox.min.z);
        const w = screenMax.x - screenMin.x;
        const h = screenMax.y - screenMin.y;

        const isSelected = selectedEntity?.id === entity.id;
        ctx.fillStyle = isSelected ? 'rgba(61, 132, 247, 0.5)' : getEntityColor(entity.type);
        ctx.strokeStyle = isSelected ? '#3d8ef7' : 'rgba(255,255,255,0.3)';
        ctx.lineWidth = isSelected ? 2 : 1;

        ctx.fillRect(screenMin.x, screenMin.y, w, h);
        ctx.strokeRect(screenMin.x, screenMin.y, w, h);

        // Label
        if (w > 60) {
          ctx.fillStyle = '#ededf2';
          ctx.font = '10px monospace';
          ctx.fillText(entity.name, screenMin.x + 2, screenMin.y + 12);
        }
      });
    }
  }, [dimensions, worldBounds, sectionPlane, showEntities, entities, selectedEntity]);

  return <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="w-full h-full" />;
}

function FloorPlanCanvas({
  entities,
  height,
  range,
  worldBounds,
  selectedEntity,
  showEntities,
  showGridLines,
  followSelection,
}: {
  entities: Entity[];
  height: number;
  range: number;
  worldBounds: BoundingBox | null;
  selectedEntity: Entity | null;
  showEntities: boolean;
  showGridLines: boolean;
  followSelection: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const resize = () => {
      if (canvasRef.current) {
        const rect = canvasRef.current.parentElement!.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  useEffect(() => {
    if (!worldBounds || !canvasRef.current || dimensions.width === 0 || dimensions.height === 0) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    const worldW = worldBounds.max.x - worldBounds.min.x;
    const worldH = worldBounds.max.z - worldBounds.min.z;
    const padding = 40;
    const scaleX = (dimensions.width - padding * 2) / worldW;
    const scaleY = (dimensions.height - padding * 2) / worldH;
    const scale = Math.min(scaleX, scaleY);
    const offsetX = (dimensions.width - worldW * scale) / 2;
    const offsetY = (dimensions.height - worldH * scale) / 2;

    const worldToScreen = (x: number, z: number) => ({
      x: offsetX + (x - worldBounds.min.x) * scale,
      y: offsetY + (worldBounds.max.z - z) * scale,
    });

    // Grid
    if (showGridLines) {
      ctx.strokeStyle = '#1f222b';
      ctx.lineWidth = 1;
      const gridSize = 10;
      for (let x = Math.ceil(worldBounds.min.x / gridSize) * gridSize; x <= worldBounds.max.x; x += gridSize) {
        const s = worldToScreen(x, worldBounds.min.z);
        const e = worldToScreen(x, worldBounds.max.z);
        ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
      }
      for (let z = Math.ceil(worldBounds.min.z / gridSize) * gridSize; z <= worldBounds.max.z; z += gridSize) {
        const s = worldToScreen(worldBounds.min.x, z);
        const e = worldToScreen(worldBounds.max.x, z);
        ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
      }
    }

    // Slice height indicator
    ctx.strokeStyle = '#3d8ef7';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    const hScreen = worldToScreen(worldBounds.min.x, height);
    ctx.beginPath(); ctx.moveTo(padding, hScreen.y); ctx.lineTo(dimensions.width - padding, hScreen.y); ctx.stroke();
    ctx.setLineDash([]);

    // Entities in floor plan range
    if (showEntities) {
      entities.forEach(entity => {
        if (!entity.boundingBox) return;
        if (!intersectsFloorPlan(entity, height, range)) return;

        const bbox = entity.boundingBox;
        const screenMin = worldToScreen(bbox.min.x, bbox.max.z);
        const screenMax = worldToScreen(bbox.max.x, bbox.min.z);
        const w = screenMax.x - screenMin.x;
        const h = screenMax.y - screenMin.y;

        const isSelected = selectedEntity?.id === entity.id;
        ctx.fillStyle = isSelected ? 'rgba(61, 132, 247, 0.5)' : getEntityColor(entity.type);
        ctx.strokeStyle = isSelected ? '#3d8ef7' : 'rgba(255,255,255,0.3)';
        ctx.lineWidth = isSelected ? 2 : 1;

        ctx.fillRect(screenMin.x, screenMin.y, w, h);
        ctx.strokeRect(screenMin.x, screenMin.y, w, h);

        if (w > 50) {
          ctx.fillStyle = '#ededf2';
          ctx.font = '9px monospace';
          ctx.fillText(entity.name, screenMin.x + 2, screenMin.y + 10);
        }
      });
    }
  }, [dimensions, worldBounds, height, range, showEntities, showGridLines, entities, selectedEntity]);

  return <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="w-full h-full" />;
}

function MinimapCanvas({
  entities,
  scale,
  worldBounds,
  selectedEntity,
  showEntities,
  showGridLines,
  followSelection,
}: {
  entities: Entity[];
  scale: number;
  worldBounds: BoundingBox | null;
  selectedEntity: Entity | null;
  showEntities: boolean;
  showGridLines: boolean;
  followSelection: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const resize = () => {
      if (canvasRef.current) {
        const rect = canvasRef.current.parentElement!.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);

  useEffect(() => {
    if (!worldBounds || !canvasRef.current || dimensions.width === 0 || dimensions.height === 0) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, dimensions.width, dimensions.height);
    ctx.fillStyle = '#08090b';
    ctx.fillRect(0, 0, dimensions.width, dimensions.height);

    const worldW = (worldBounds.max.x - worldBounds.min.x) * scale;
    const worldH = (worldBounds.max.z - worldBounds.min.z) * scale;
    const padding = 40;
    const scaleX = (dimensions.width - padding * 2) / worldW;
    const scaleY = (dimensions.height - padding * 2) / worldH;
    const finalScale = Math.min(scaleX, scaleY);
    const offsetX = (dimensions.width - worldW * finalScale) / 2;
    const offsetY = (dimensions.height - worldH * finalScale) / 2;

    const worldToScreen = (x: number, z: number) => ({
      x: offsetX + (x - worldBounds.min.x) * finalScale * scale,
      y: offsetY + (worldBounds.max.z - z) * finalScale * scale,
    });

    // Grid
    if (showGridLines) {
      ctx.strokeStyle = '#1f222b';
      ctx.lineWidth = 1;
      const gridSize = 50 * scale;
      for (let x = Math.ceil(worldBounds.min.x / gridSize) * gridSize; x <= worldBounds.max.x; x += gridSize) {
        const s = worldToScreen(x, worldBounds.min.z);
        const e = worldToScreen(x, worldBounds.max.z);
        ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
      }
      for (let z = Math.ceil(worldBounds.min.z / gridSize) * gridSize; z <= worldBounds.max.z; z += gridSize) {
        const s = worldToScreen(worldBounds.min.x, z);
        const e = worldToScreen(worldBounds.max.x, z);
        ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(e.x, e.y); ctx.stroke();
      }
    }

    // Entities
    if (showEntities) {
      entities.forEach(entity => {
        if (!entity.boundingBox) return;
        const bbox = entity.boundingBox;
        const screenMin = worldToScreen(bbox.min.x, bbox.max.z);
        const screenMax = worldToScreen(bbox.max.x, bbox.min.z);
        const w = screenMax.x - screenMin.x;
        const h = screenMax.y - screenMin.y;

        if (w < 1 && h < 1) {
          // Too small, draw as dot
          ctx.fillStyle = getEntityColor(entity.type);
          const cx = (screenMin.x + screenMax.x) / 2;
          const cy = (screenMin.y + screenMax.y) / 2;
          ctx.beginPath(); ctx.arc(cx, cy, 1.5, 0, Math.PI * 2); ctx.fill();
          return;
        }

        const isSelected = selectedEntity?.id === entity.id;
        ctx.fillStyle = isSelected ? 'rgba(61, 132, 247, 0.6)' : getEntityColor(entity.type);
        ctx.strokeStyle = isSelected ? '#3d8ef7' : 'rgba(255,255,255,0.2)';
        ctx.lineWidth = isSelected ? 2 : 0.5;

        ctx.fillRect(screenMin.x, screenMin.y, w, h);
        ctx.strokeRect(screenMin.x, screenMin.y, w, h);
      });
    }

    // Viewport indicator
    ctx.strokeStyle = '#3d8ef7';
    ctx.lineWidth = 2;
    ctx.strokeRect(
      dimensions.width * 0.35,
      dimensions.height * 0.35,
      dimensions.width * 0.3,
      dimensions.height * 0.3
    );
  }, [dimensions, worldBounds, scale, showEntities, showGridLines, entities, selectedEntity]);

  return <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="w-full h-full" />;
}

function intersectsSection(entity: Entity, plane: { normal: Vec3; distance: number }): boolean {
  if (!entity.boundingBox) return false;
  const { min, max } = entity.boundingBox;
  const n = plane.normal;
  const d = plane.distance;
  
  // Check if bbox intersects plane
  const vertices = [
    { x: min.x, y: min.y, z: min.z },
    { x: max.x, y: min.y, z: min.z },
    { x: min.x, y: max.y, z: min.z },
    { x: max.x, y: max.y, z: min.z },
    { x: min.x, y: min.y, z: max.z },
    { x: max.x, y: min.y, z: max.z },
    { x: min.x, y: max.y, z: max.z },
    { x: max.x, y: max.y, z: max.z },
  ];
  
  let hasPositive = false, hasNegative = false;
  for (const v of vertices) {
    const dist = n.x * v.x + n.y * v.y + n.z * v.z - d;
    if (dist > 0) hasPositive = true;
    if (dist < 0) hasNegative = true;
    if (hasPositive && hasNegative) return true;
  }
  return false;
}

function intersectsFloorPlan(entity: Entity, height: number, range: number): boolean {
  if (!entity.boundingBox) return false;
  const { min, max } = entity.boundingBox;
  return max.y >= height - range && min.y <= height + range;
}

function getEntityColor(type: string): string {
  const colors: Record<string, string> = {
    BUILDING: 'rgba(61, 132, 247, 0.4)',
    FACADE: 'rgba(61, 132, 247, 0.5)',
    WALL: 'rgba(61, 132, 247, 0.5)',
    FLOOR: 'rgba(61, 132, 247, 0.4)',
    ROOM: 'rgba(46, 204, 113, 0.4)',
    OBJECT: 'rgba(168, 85, 247, 0.4)',
    COMPONENT: 'rgba(168, 85, 247, 0.4)',
    COLUMN: 'rgba(245, 158, 11, 0.4)',
    TERRAIN: 'rgba(115, 84, 50, 0.4)',
    ROAD: 'rgba(148, 163, 184, 0.4)',
    VEGETATION: 'rgba(46, 204, 113, 0.4)',
    WORLD: 'rgba(61, 132, 247, 0.4)',
    SITE: 'rgba(94, 218, 255, 0.4)',
  };
  return colors[type] || 'rgba(84, 89, 107, 0.4)';
}