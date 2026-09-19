'use client';

import React, { useState } from 'react';
import { useREStore } from '@/store/re-store';
import { ConfidenceGauge } from '@/components/ui/confidence-gauge';
import { EmptyState } from '@/components/ui/empty-state';
import {
  PanelRightClose,
  Box,
  Layers,
  Sparkles,
  Camera,
  Activity,
  History,
  Network,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  ShieldCheck,
} from 'lucide-react';

export default function ContextInspectorPanel() {
  const selection = useREStore((s) => s.selection);
  const entities = useREStore((s) => s.entities);
  const toggleInspector = useREStore((s) => s.toggleInspector);
  const spatialScale = useREStore((s) => s.spatialScale);

  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    geometry: true,
    semantics: true,
    evidence: true,
    provenance: true,
    uncertainty: true,
    temporal: true,
    relationships: true,
  });

  const toggleSection = (key: string) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const selectedEntityId = selection.selectedEntityIds[0] ?? selection.focusedEntityId;
  const entity = selectedEntityId ? entities.get(selectedEntityId) : undefined;

  // Fallback realistic inspection data (when an entity or placeholder is selected)
  const inspectionData = {
    name: entity?.name ?? 'Central Street Light #14',
    type: entity?.type ?? 'Street_Light',
    meshCount: 1880,
    triangles: 2880,
    vertices: 133,
    lods: '8.5m / LOD 2',
    material: 'Galvanized Steel & High-Pressure Sodium',
    height: '8.5 m',
    sourceSession: 'Laser Scan [2024-05-15]',
    imageFrames: 'Frame 345, 346, 350',
    resolution: '1.8 mm/px',
    algorithm: 'SfM MVS → Lidar Fusion → Mesh Gen',
    processedBy: 'Reality Engine Daemon v7.3',
    geometricErrorCm: 2.1,
    semanticConfidencePct: 98.2,
    temporalHistory: [
      { version: 'V6.3', event: 'Initial Spatial Anchoring' },
      { version: 'V7.1', event: 'Luminaire Replacement' },
      { version: 'V7.3', event: 'Current Reconstructed State', isCurrent: true },
    ],
    relationships: [
      { relation: 'Connected To', target: 'Power Grid Node 12' },
      { relation: 'Located On', target: 'Main Street Sidewalk #4' },
      { relation: 'Occluded By', target: 'Northern Oak Canopy' },
    ],
  };

  return (
    <aside
      className="flex flex-col w-80 h-full bg-[#101217] border-l border-[#1f222b] select-none text-[#ededf2] font-sans shrink-0 overflow-hidden"
      aria-label="Context Inspector Panel"
    >
      {/* ── Header ── */}
      <div className="h-10 px-3 border-b border-[#1f222b] flex items-center justify-between shrink-0 bg-[#0d0e12]">
        <div className="flex items-center gap-1.5 truncate">
          <span className="text-[11px] font-mono font-bold tracking-wider text-[#9296a6] uppercase">
            Context Inspector
          </span>
          <span className="text-[9px] font-mono px-1 rounded bg-[#161a24] text-[#00e5ff] border border-[#00e5ff]/30">
            {spatialScale}
          </span>
        </div>

        <button
          type="button"
          onClick={toggleInspector}
          className="p-1 rounded text-[#54596b] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
          title="Collapse Panel (⌘I)"
        >
          <PanelRightClose className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* ── Entity Title / Badge ── */}
      <div className="p-3 border-b border-[#1f222b] bg-[#0f1116] shrink-0">
        <div className="flex items-center justify-between gap-1">
          <div className="flex items-center gap-1.5 truncate">
            <div className="w-2 h-2 rounded-full bg-[#00e5ff] shadow-[0_0_6px_#00e5ff]" />
            <h3 className="text-xs font-bold text-[#f0f1f6] truncate font-sans">
              {inspectionData.name}
            </h3>
          </div>
          <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-white/5 text-[#9296a6]">
            {inspectionData.type}
          </span>
        </div>
        <div className="flex items-center gap-2 mt-1 text-[10px] font-mono text-[#54596b]">
          <span>ID: {selectedEntityId ?? 'ent-active-01'}</span>
          <span>•</span>
          <span className="text-[#2ecc71] font-semibold">VERIFIED</span>
        </div>
      </div>

      {/* ── Scrollable Inspector Sections ── */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-3 font-mono text-xs">
        {/* 1. GEOMETRY */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('geometry')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Box className="w-3.5 h-3.5 text-[#38bdf8]" />
              <span className="text-[#f0f1f6]">GEOMETRY</span>
            </span>
            {openSections.geometry ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.geometry && (
            <div className="p-2.5 pt-0 grid grid-cols-2 gap-2 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
              <div>
                <div className="text-[#54596b]">Mesh Count</div>
                <div className="text-[#f0f1f6] font-bold font-num">{inspectionData.meshCount}</div>
              </div>
              <div>
                <div className="text-[#54596b]">Triangles</div>
                <div className="text-[#f0f1f6] font-bold font-num">{inspectionData.triangles.toLocaleString()}</div>
              </div>
              <div>
                <div className="text-[#54596b]">Vertices</div>
                <div className="text-[#f0f1f6] font-bold font-num">{inspectionData.vertices}</div>
              </div>
              <div>
                <div className="text-[#54596b]">LOD Resolution</div>
                <div className="text-[#00e5ff] font-bold font-num">{inspectionData.lods}</div>
              </div>
            </div>
          )}
        </div>

        {/* 2. SEMANTICS */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('semantics')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-[#a855f7]" />
              <span className="text-[#f0f1f6]">SEMANTICS</span>
            </span>
            {openSections.semantics ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.semantics && (
            <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
              <div className="flex justify-between">
                <span className="text-[#54596b]">Class:</span>
                <span className="text-[#f0f1f6] font-semibold">{inspectionData.type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#54596b]">Material:</span>
                <span className="text-[#f0f1f6] font-semibold">{inspectionData.material}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#54596b]">Height:</span>
                <span className="text-[#00e5ff] font-bold">{inspectionData.height}</span>
              </div>
            </div>
          )}
        </div>

        {/* 3. EVIDENCE */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('evidence')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Camera className="w-3.5 h-3.5 text-[#2ecc71]" />
              <span className="text-[#f0f1f6]">EVIDENCE</span>
            </span>
            {openSections.evidence ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.evidence && (
            <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
              <div className="flex justify-between">
                <span className="text-[#54596b]">Source:</span>
                <span className="text-[#f0f1f6] truncate max-w-[150px]">{inspectionData.sourceSession}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#54596b]">Image Frame:</span>
                <span className="text-[#f0f1f6]">{inspectionData.imageFrames}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#54596b]">Resolution:</span>
                <span className="text-[#2ecc71] font-bold">{inspectionData.resolution}</span>
              </div>
            </div>
          )}
        </div>

        {/* 4. PROVENANCE */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('provenance')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#ec4899]" />
              <span className="text-[#f0f1f6]">PROVENANCE</span>
            </span>
            {openSections.provenance ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.provenance && (
            <div className="p-2.5 pt-0 space-y-1.5 text-[10px] text-[#9296a6] border-t border-white/5 mt-1">
              <div>
                <div className="text-[#54596b]">Algorithm:</div>
                <div className="text-[#f0f1f6] text-[9px] mt-0.5 leading-relaxed">{inspectionData.algorithm}</div>
              </div>
              <div>
                <div className="text-[#54596b]">Processed By:</div>
                <div className="text-[#9296a6] text-[9px]">{inspectionData.processedBy}</div>
              </div>
            </div>
          )}
        </div>

        {/* 5. UNCERTAINTY & CONFIDENCE */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('uncertainty')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-[#f5a623]" />
              <span className="text-[#f0f1f6]">UNCERTAINTY</span>
            </span>
            {openSections.uncertainty ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.uncertainty && (
            <div className="p-2.5 pt-0 space-y-2.5 border-t border-white/5 mt-1">
              <ConfidenceGauge
                label="Geometric Error"
                value={42}
                displayValue={`${inspectionData.geometricErrorCm} cm`}
                variant="error"
              />
              <ConfidenceGauge
                label="Semantic Confidence"
                value={inspectionData.semanticConfidencePct}
                variant="confidence"
              />
            </div>
          )}
        </div>

        {/* 6. TEMPORAL HISTORY */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('temporal')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <History className="w-3.5 h-3.5 text-[#38bdf8]" />
              <span className="text-[#f0f1f6]">TEMPORAL HISTORY</span>
            </span>
            {openSections.temporal ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.temporal && (
            <div className="p-2.5 pt-0 space-y-1.5 text-[9px] border-t border-white/5 mt-1">
              {inspectionData.temporalHistory.map((item) => (
                <div
                  key={item.version}
                  className={`flex items-center justify-between p-1.5 rounded ${
                    item.isCurrent ? 'bg-[#00e5ff]/10 text-[#00e5ff] font-bold' : 'text-[#9296a6]'
                  }`}
                >
                  <span>{item.version}:</span>
                  <span>{item.event}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 7. RELATIONSHIPS */}
        <div className="rounded-lg bg-[#14161f] border border-[#1f222b] overflow-hidden">
          <button
            type="button"
            onClick={() => toggleSection('relationships')}
            className="w-full flex items-center justify-between p-2.5 hover:bg-white/5 text-[10px] font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Network className="w-3.5 h-3.5 text-[#00e5ff]" />
              <span className="text-[#f0f1f6]">RELATIONSHIPS</span>
            </span>
            {openSections.relationships ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.relationships && (
            <div className="p-2.5 pt-0 space-y-1.5 text-[9px] text-[#9296a6] border-t border-white/5 mt-1">
              {inspectionData.relationships.map((rel, i) => (
                <div key={i} className="flex justify-between items-center p-1 rounded hover:bg-white/5">
                  <span className="text-[#54596b]">{rel.relation}:</span>
                  <span className="text-[#f0f1f6] font-semibold">{rel.target}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
