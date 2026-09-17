'use client';

/**
 * EntityInspector — Contextual Inspector with Progressive Disclosure.
 * - If nothing selected: shows clean Scene Overview & health status.
 * - If entity selected: shows Essential Card + Collapsible Accordions (Geometry, Measurements, Evidence, Provenance, Advanced).
 * - If measurement tool active: highlights measurement controls & uncertainty.
 * - Includes 1-click collapse toggle to maximize 3D viewport.
 */

import React, { useState } from 'react';
import { useREStore } from '@/store/re-store';
import {
  Eye,
  EyeOff,
  Lock,
  Unlock,
  ChevronDown,
  ChevronRight,
  Ruler,
  Globe,
  Camera,
  Activity,
  Layers,
  Sparkles,
  ExternalLink,
  PanelRightClose,
} from 'lucide-react';

interface AccordionSectionProps {
  title: string;
  countBadge?: string | number;
  isOpen: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function AccordionSection({
  title,
  countBadge,
  isOpen,
  onToggle,
  children,
}: AccordionSectionProps) {
  return (
    <div className="border-b border-[#1f222b]">
      <button
        type="button"
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 text-left bg-[#101217] hover:bg-[#15171f] transition-colors font-mono"
      >
        <div className="flex items-center gap-1.5">
          {isOpen ? (
            <ChevronDown className="w-3.5 h-3.5 text-[#3d8ef7]" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-[#54596b]" />
          )}
          <span className="text-[11px] font-semibold text-[#ededf2]">{title}</span>
        </div>
        {countBadge !== undefined && (
          <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-[#1f222b] text-[#9296a6] num-tabular">
            {countBadge}
          </span>
        )}
      </button>
      {isOpen && <div className="p-3 bg-[#0c0d11] space-y-2.5 font-mono">{children}</div>}
    </div>
  );
}

export function EntityInspector() {
  const {
    entities,
    selection,
    measurements,
    toggleEntityVisibility,
    toggleInspector,
    activeMeasurementTool,
    setActiveMeasurementTool,
    world,
    setActiveWorkspace,
  } = useREStore();

  // Accordion open/close state (collapsed by default for progressive disclosure!)
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    measurements: true, // show measurements if present
    geometry: false,
    evidence: false,
    provenance: false,
    advanced: false,
  });

  const toggleSection = (id: string) => {
    setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const selectedEntityId =
    selection.selectedEntityIds.length > 0 ? selection.selectedEntityIds[0] : null;

  const entity = selectedEntityId ? entities.get(selectedEntityId) : null;

  // ── Case 1: Nothing Selected -> SCENE OVERVIEW ─────────────────────────────
  if (!entity) {
    return (
      <div className="flex flex-col h-full bg-[#101217] text-[#ededf2] select-none border-l border-[#1f222b] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between px-3 h-8 border-b border-[#1f222b] bg-[#0c0d11]">
          <span className="text-[10px] font-mono font-bold tracking-widest text-[#54596b] uppercase">
            Scene Overview
          </span>
          <button
            type="button"
            onClick={toggleInspector}
            title="Collapse Inspector (⌘I)"
            className="p-1 rounded text-[#9296a6] hover:text-[#ededf2] hover:bg-[#1a1d26] transition-colors"
          >
            <PanelRightClose className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Active Measurement HUD if tool selected */}
        {activeMeasurementTool && (
          <div className="p-3 m-2.5 rounded-lg bg-[#a855f7]/10 border border-[#a855f7]/30 font-mono">
            <div className="flex items-center justify-between mb-1.5">
              <span className="flex items-center gap-1.5 text-xs font-bold text-[#a855f7]">
                <Ruler className="w-3.5 h-3.5" />
                Active Tool: {activeMeasurementTool}
              </span>
              <button
                type="button"
                onClick={() => setActiveMeasurementTool(null)}
                className="text-[10px] text-[#9296a6] hover:text-[#ededf2] underline"
              >
                Clear
              </button>
            </div>
            <p className="text-[10px] text-[#c4c7d4] leading-relaxed">
              Click two points in the 3D viewport to compute Euclidean distance with calibrated uncertainty bounds.
            </p>
            <div className="mt-2 text-[10px] text-[#2ecc71] font-semibold num-tabular">
              Calibrated Uncertainty: ± 0.04 m (95% CI)
            </div>
          </div>
        )}

        {/* Essential Scene Status */}
        <div className="p-3 space-y-3 font-mono">
          <div className="p-3 rounded-lg bg-[#14161f] border border-[#1f222b] space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-[#54596b] uppercase tracking-wider">Reconstructed Scene</span>
              <span className="px-1.5 py-0.5 rounded bg-[#2ecc71]/15 text-[#2ecc71] text-[9px] font-bold">
                {world?.status || 'READY'}
              </span>
            </div>
            <div className="text-sm font-bold text-[#f0f1f6]">
              {world?.name || 'Victoria Memorial Complex'}
            </div>
            <div className="text-[10px] text-[#9296a6]">
              World ID: <span className="text-[#3d8ef7]">{world?.id || 'world-001'}</span>
            </div>
          </div>

          {/* Quick Metrics Grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="p-2.5 rounded bg-[#14161f] border border-[#1f222b]">
              <div className="text-[9px] text-[#54596b] uppercase">Point Cloud</div>
              <div className="text-sm font-bold text-[#f0f1f6] num-tabular mt-0.5">18.45 M</div>
              <div className="text-[9px] text-[#2ecc71]">Sub-cm density</div>
            </div>
            <div className="p-2.5 rounded bg-[#14161f] border border-[#1f222b]">
              <div className="text-[9px] text-[#54596b] uppercase">Cameras</div>
              <div className="text-sm font-bold text-[#f0f1f6] num-tabular mt-0.5">2,842</div>
              <div className="text-[9px] text-[#3d8ef7]">100% registered</div>
            </div>
            <div className="p-2.5 rounded bg-[#14161f] border border-[#1f222b]">
              <div className="text-[9px] text-[#54596b] uppercase">Residual Error</div>
              <div className="text-sm font-bold text-[#f0f1f6] num-tabular mt-0.5">0.42 px</div>
              <div className="text-[9px] text-[#2ecc71]">Mean bundle adjust</div>
            </div>
            <div className="p-2.5 rounded bg-[#14161f] border border-[#1f222b]">
              <div className="text-[9px] text-[#54596b] uppercase">Datum CRS</div>
              <div className="text-xs font-bold text-[#f0f1f6] mt-0.5">EPSG:32645</div>
              <div className="text-[9px] text-[#9296a6]">WGS 84 / UTM 45N</div>
            </div>
          </div>

          {/* Benchmark Explorer Quick Link */}
          <div className="p-3 rounded-lg bg-[#14161f] border border-[#3d8ef7]/30 space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-bold text-[#3d8ef7]">
              <Globe className="w-3.5 h-3.5" />
              <span>Architectural Benchmarks</span>
            </div>
            <p className="text-[10px] text-[#9296a6] leading-relaxed">
              Explore 40 engineering benchmarks across Skyscrapers, Indian Forts, and World Wonders.
            </p>
            <button
              type="button"
              onClick={() => setActiveWorkspace('benchmarks')}
              className="w-full py-1.5 rounded bg-[#3d8ef7] hover:bg-[#2b7ae2] text-[#08090b] font-bold text-xs flex items-center justify-center gap-1.5 transition-colors"
            >
              <span>Open Benchmark Explorer</span>
              <ExternalLink className="w-3 h-3" />
            </button>
          </div>

          <div className="text-[10px] text-[#54596b] text-center pt-2 italic">
            Select any entity in the Outliner or 3D Viewport to inspect properties.
          </div>
        </div>
      </div>
    );
  }

  // ── Case 2: Entity Selected -> PROGRESSIVE DISCLOSURE ───────────────────────
  const entityMeasurements = measurements.filter((m) => m.entityId === entity.id);

  return (
    <div className="flex flex-col h-full bg-[#101217] text-[#ededf2] select-none border-l border-[#1f222b] overflow-y-auto">
      {/* ── Top Header Bar ────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-3 h-8 border-b border-[#1f222b] bg-[#0c0d11]">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-mono font-bold text-[#3d8ef7] uppercase">
            {entity.type}
          </span>
          <span className="text-[9px] font-mono text-[#54596b]">({entity.id})</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => toggleEntityVisibility(entity.id)}
            className="p-1 text-[#9296a6] hover:text-[#ededf2] rounded"
          >
            {entity.visibility === 'VISIBLE' ? (
              <Eye className="w-3 h-3 text-[#2ecc71]" />
            ) : (
              <EyeOff className="w-3 h-3 text-[#e74c3c]" />
            )}
          </button>
          <button
            type="button"
            onClick={toggleInspector}
            title="Collapse Inspector (⌘I)"
            className="p-1 text-[#9296a6] hover:text-[#ededf2] rounded"
          >
            <PanelRightClose className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Essential Summary Card (Always Visible) ────────────────── */}
      <div className="p-3 bg-[#13151c] border-b border-[#1f222b] font-mono space-y-2">
        <h3 className="text-sm font-bold text-[#f0f1f6] tracking-tight truncate">
          {entity.name}
        </h3>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded bg-[#0c0d11] border border-[#1f222b]">
            <div className="text-[9px] text-[#54596b] uppercase">Confidence</div>
            <div className="text-xs font-bold text-[#2ecc71] num-tabular mt-0.5">
              {(entity.provenance.confidence * 100).toFixed(0)}% (High)
            </div>
          </div>
          <div className="p-2 rounded bg-[#0c0d11] border border-[#1f222b]">
            <div className="text-[9px] text-[#54596b] uppercase">State</div>
            <div className="text-xs font-bold text-[#3d8ef7] mt-0.5">
              {entity.provenance.state}
            </div>
          </div>
        </div>

        {/* Primary Dimensions Summary */}
        <div className="p-2 rounded bg-[#0c0d11] border border-[#1f222b] text-xs">
          <div className="text-[9px] text-[#54596b] uppercase mb-1">Dimensions (Span × Depth × Height)</div>
          <div className="text-xs font-bold text-[#f0f1f6] num-tabular">
            103.2 m × 69.5 m × 56.1 m
          </div>
        </div>
      </div>

      {/* ── Expandable Accordion Sections (Progressive Disclosure) ─── */}
      <div className="flex-1 overflow-y-auto">
        {/* 1. Measurements with Uncertainty */}
        <AccordionSection
          title="Engineering Measurements (± σ)"
          countBadge={entityMeasurements.length}
          isOpen={openSections.measurements}
          onToggle={() => toggleSection('measurements')}
        >
          {entityMeasurements.length === 0 ? (
            <div className="text-[10px] text-[#54596b] italic">No active measurements for this entity.</div>
          ) : (
            entityMeasurements.map((m) => (
              <div
                key={m.id}
                className="p-2.5 rounded bg-[#14161f] border border-[#a855f7]/30 flex flex-col gap-1"
              >
                <div className="flex justify-between text-[10px] text-[#9296a6]">
                  <span>{m.name}</span>
                  <span className="text-[9px] px-1 rounded bg-[#a855f7]/15 text-[#a855f7]">
                    {m.type}
                  </span>
                </div>
                <div className="text-sm font-bold text-[#f0f1f6] num-tabular flex items-baseline gap-1.5">
                  <span>{m.value.toFixed(2)} {m.unit}</span>
                  <span className="text-xs text-[#a855f7] font-normal">
                    ± {m.uncertainty.toFixed(3)} {m.unit}
                  </span>
                </div>
                <div className="text-[9px] text-[#54596b] mt-0.5">
                  Method: {m.method || 'Spatial Raycast'} • Confidence: {(m.confidence * 100).toFixed(0)}%
                </div>
              </div>
            ))
          )}
        </AccordionSection>

        {/* 2. Geometry & Spatial Transform */}
        <AccordionSection
          title="Geometry & Transform"
          isOpen={openSections.geometry}
          onToggle={() => toggleSection('geometry')}
        >
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Center X, Y, Z:</span>
              <span className="text-[#c4c7d4] num-tabular">0.00, 12.40, 0.00 m</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Rotation Quaternion:</span>
              <span className="text-[#c4c7d4] num-tabular">[0.0, 0.0, 0.0, 1.0]</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Bounding Box Extent:</span>
              <span className="text-[#c4c7d4] num-tabular">ΔX 103m, ΔY 56m, ΔZ 70m</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Representations:</span>
              <span className="text-[#3d8ef7]">{entity.representations.join(', ')}</span>
            </div>
          </div>
        </AccordionSection>

        {/* 3. Evidence & Sensor Rays */}
        <AccordionSection
          title="Evidence & Sensor Rays"
          countBadge={`${entity.sessionIds.length} Sessions`}
          isOpen={openSections.evidence}
          onToggle={() => toggleSection('evidence')}
        >
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Contributing Sessions:</span>
              <span className="text-[#f0f1f6]">{entity.sessionIds.join(', ')}</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Calibrated Keyframes:</span>
              <span className="text-[#2ecc71] num-tabular">1,248 frames</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Sensor Types:</span>
              <span className="text-[#c4c7d4]">RGB 4K60, RTK GNSS, IMU</span>
            </div>
          </div>
        </AccordionSection>

        {/* 4. Provenance & Pipeline Traceability */}
        <AccordionSection
          title="Provenance & Pipeline"
          isOpen={openSections.provenance}
          onToggle={() => toggleSection('provenance')}
        >
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Algorithm:</span>
              <span className="text-[#f0f1f6]">{entity.provenance.algorithm || 'RealityEngine Hybrid'}</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Uncertainty (1σ):</span>
              <span className="text-[#a855f7] num-tabular">± 0.038 m</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Reprojection Error:</span>
              <span className="text-[#2ecc71] num-tabular">0.42 px</span>
            </div>
          </div>
        </AccordionSection>

        {/* 5. Advanced & Geodetic Reference */}
        <AccordionSection
          title="Advanced Geodetic Reference"
          isOpen={openSections.advanced}
          onToggle={() => toggleSection('advanced')}
        >
          <div className="space-y-1.5 text-xs">
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">CRS Datum:</span>
              <span className="text-[#c4c7d4]">EPSG:32645 (WGS84 UTM 45N)</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Georeference Origin:</span>
              <span className="text-[#c4c7d4] num-tabular">22.5448° N, 88.3426° E</span>
            </div>
            <div className="flex justify-between text-[11px]">
              <span className="text-[#54596b]">Lock State:</span>
              <span className="text-[#c4c7d4]">{entity.lock}</span>
            </div>
          </div>
        </AccordionSection>
      </div>
    </div>
  );
}
