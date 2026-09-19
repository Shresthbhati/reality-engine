'use client';

/**
 * ViewportToolbar — Streamlined primary actions toolbar.
 * - Reduced button clutter: compact projection and shading dropdowns
 * - Grouped essential layers pill (Grid, Mesh, Cameras)
 * - Measurement tool trigger with live precision indicator
 * - Panel visibility toggles (Outliner ⌘B, Inspector ⌘I, Bottom Drawer ⌘J, Commands ⌘K)
 */

import React, { useState } from 'react';
import { useREStore, MeasurementType } from '@/store/re-store';
import {
  ChevronDown,
  Layers,
  Ruler,
  PanelLeft,
  PanelRight,
  PanelBottom,
  Command,
  Grid3x3,
  Camera,
  Box,
} from 'lucide-react';

export function ViewportToolbar() {
  const {
    viewportMode,
    setViewportMode,
    shadingMode,
    setShadingMode,
    showGrid,
    showCameras,
    showMesh,
    toggleViewportOption,
    activeMeasurementTool,
    setActiveMeasurementTool,
    outlinerCollapsed,
    toggleOutliner,
    inspectorCollapsed,
    toggleInspector,
    bottomDrawerOpen,
    toggleBottomDrawer,
    setCommandPaletteOpen,
  } = useREStore();

  const [measurementDropdownOpen, setMeasurementDropdownOpen] = useState(false);
  const [shadingDropdownOpen, setShadingDropdownOpen] = useState(false);
  const [projectionDropdownOpen, setProjectionDropdownOpen] = useState(false);

  const measurementTools: Array<{ type: MeasurementType; label: string }> = [
    { type: 'POINT_TO_POINT', label: 'Point-to-Point Distance' },
    { type: 'HEIGHT', label: 'Vertical Height' },
    { type: 'WIDTH', label: 'Horizontal Width / Span' },
    { type: 'DEPTH', label: 'Depth Dimension' },
    { type: 'WALL_THICKNESS', label: 'Wall Thickness' },
    { type: 'AREA', label: 'Surface / Floor Area' },
    { type: 'VOLUME', label: 'Volumetric Boundary' },
  ];

  const shadingModes: Array<{ id: typeof shadingMode; label: string; desc: string }> = [
    { id: 'RGB', label: 'True Color (RGB)', desc: 'Calibrated radiometric color' },
    { id: 'CONFIDENCE', label: 'Confidence Heatmap', desc: 'Green (high) to red (low)' },
    { id: 'COVERAGE', label: 'Coverage Density', desc: 'Observation ray density' },
    { id: 'NORMALS', label: 'Surface Normals', desc: 'Geometric vector direction' },
    { id: 'DEPTH', label: 'Depth Gradient', desc: 'Z-depth range ramp' },
    { id: 'POINT_CLOUD', label: 'Dense Point Cloud', desc: 'Sub-millimeter point samples' },
    { id: 'WIREFRAME', label: 'Structural Wireframe', desc: 'Tessellated face edges' },
  ];

  const projectionModes: Array<{ id: typeof viewportMode; label: string }> = [
    { id: 'PERSPECTIVE', label: 'Perspective (3D)' },
    { id: 'ORTHOGRAPHIC', label: 'Orthographic' },
    { id: 'TOP', label: 'Top View (Plan)' },
    { id: 'FRONT', label: 'Front View (Elevation)' },
    { id: 'SIDE', label: 'Side View' },
  ];

  return (
    <div className="h-8 px-2.5 flex items-center justify-between border-b border-[#1f222b] bg-[#0c0d11] shrink-0 text-xs select-none font-mono">
      {/* ── Left: Projections & Shading (Compact Dropdowns) ────────── */}
      <div className="flex items-center gap-1.5">
        {/* Projection Mode */}
        <div className="relative">
          <button
            type="button"
            onClick={() => {
              setProjectionDropdownOpen(!projectionDropdownOpen);
              setShadingDropdownOpen(false);
              setMeasurementDropdownOpen(false);
            }}
            className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] hover:border-[#3d8ef7]/40 text-[#ededf2] text-[11px] transition-colors"
          >
            <span className="text-[#54596b]">View:</span>
            <span className="font-bold text-[#3d8ef7]">{viewportMode}</span>
            <ChevronDown className="w-3 h-3 text-[#54596b]" />
          </button>

          {projectionDropdownOpen && (
            <div className="absolute top-full left-0 mt-1 w-44 bg-[#0f1014] border border-[#1f222b] rounded-lg shadow-2xl py-1 z-50 text-[11px]">
              {projectionModes.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => {
                    setViewportMode(p.id);
                    setProjectionDropdownOpen(false);
                  }}
                  className={`w-full text-left px-3 py-1.5 hover:bg-[#181b23] flex items-center justify-between ${
                    viewportMode === p.id ? 'text-[#3d8ef7] font-bold bg-[#141720]' : 'text-[#c4c7d4]'
                  }`}
                >
                  <span>{p.label}</span>
                  {viewportMode === p.id && <span className="text-[10px] text-[#3d8ef7]">✓</span>}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Shading Mode */}
        <div className="relative">
          <button
            type="button"
            onClick={() => {
              setShadingDropdownOpen(!shadingDropdownOpen);
              setProjectionDropdownOpen(false);
              setMeasurementDropdownOpen(false);
            }}
            className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#14161f] border border-[#1f222b] hover:border-[#3d8ef7]/40 text-[#ededf2] text-[11px] transition-colors"
          >
            <Layers className="w-3 h-3 text-[#2ecc71]" />
            <span className="text-[#54596b]">Shading:</span>
            <span>{shadingMode}</span>
            <ChevronDown className="w-3 h-3 text-[#54596b]" />
          </button>

          {shadingDropdownOpen && (
            <div className="absolute top-full left-0 mt-1 w-52 bg-[#0f1014] border border-[#1f222b] rounded-lg shadow-2xl py-1 z-50 text-[11px]">
              {shadingModes.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => {
                    setShadingMode(m.id);
                    setShadingDropdownOpen(false);
                  }}
                  className={`w-full text-left px-3 py-1.5 hover:bg-[#181b23] flex flex-col ${
                    shadingMode === m.id ? 'text-[#3d8ef7] font-bold bg-[#141720]' : 'text-[#c4c7d4]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span>{m.label}</span>
                    {shadingMode === m.id && <span className="text-[10px] text-[#3d8ef7]">✓</span>}
                  </div>
                  <span className="text-[9px] text-[#54596b]">{m.desc}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Center: Essential Layers Pill ──────────────────────────── */}
      <div className="flex items-center bg-[#14161f] rounded border border-[#1f222b] p-0.5 text-[10px]">
        <button
          type="button"
          onClick={() => toggleViewportOption('showGrid')}
          className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
            showGrid ? 'bg-[#1f222b] text-[#f0f1f6] font-semibold' : 'text-[#54596b] hover:text-[#9296a6]'
          }`}
          title="Toggle Grid Ground Plane"
        >
          <Grid3x3 className="w-2.5 h-2.5" />
          <span>Grid</span>
        </button>
        <button
          type="button"
          onClick={() => toggleViewportOption('showMesh')}
          className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
            showMesh ? 'bg-[#1f222b] text-[#f0f1f6] font-semibold' : 'text-[#54596b] hover:text-[#9296a6]'
          }`}
          title="Toggle Reconstructed Surface Mesh"
        >
          <Box className="w-2.5 h-2.5" />
          <span>Mesh</span>
        </button>
        <button
          type="button"
          onClick={() => toggleViewportOption('showCameras')}
          className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
            showCameras ? 'bg-[#1f222b] text-[#f0f1f6] font-semibold' : 'text-[#54596b] hover:text-[#9296a6]'
          }`}
          title="Toggle Orbital Camera Frustums"
        >
          <Camera className="w-2.5 h-2.5" />
          <span>Cameras</span>
        </button>
      </div>

      {/* ── Right: Measurement Tool & Panel Controls ──────────────── */}
      <div className="flex items-center gap-1.5">
        {/* Measurement Tool Dropdown */}
        <div className="relative">
          <button
            type="button"
            onClick={() => {
              setMeasurementDropdownOpen(!measurementDropdownOpen);
              setShadingDropdownOpen(false);
              setProjectionDropdownOpen(false);
            }}
            className={`flex items-center gap-1 px-2 py-1 rounded text-[11px] border transition-colors ${
              activeMeasurementTool
                ? 'bg-[#a855f7]/15 border-[#a855f7]/40 text-[#a855f7] font-bold'
                : 'bg-[#14161f] border-[#1f222b] text-[#c4c7d4] hover:text-[#ededf2]'
            }`}
          >
            <Ruler className="w-3 h-3" />
            <span>
              {activeMeasurementTool ? `${activeMeasurementTool} (±0.04m)` : 'Measure'}
            </span>
            <ChevronDown className="w-2.5 h-2.5 text-[#54596b]" />
          </button>

          {measurementDropdownOpen && (
            <div className="absolute top-full right-0 mt-1 w-56 bg-[#0f1014] border border-[#1f222b] rounded-lg shadow-2xl py-1 z-50 text-[11px]">
              <div className="px-3 py-1 text-[9px] text-[#54596b] uppercase border-b border-[#1f222b]">
                Select Spatial Measurement Tool
              </div>
              {measurementTools.map((tool) => (
                <button
                  key={tool.type}
                  type="button"
                  onClick={() => {
                    setActiveMeasurementTool(
                      activeMeasurementTool === tool.type ? null : tool.type
                    );
                    setMeasurementDropdownOpen(false);
                  }}
                  className={`w-full text-left px-3 py-1.5 hover:bg-[#181b23] flex items-center justify-between ${
                    activeMeasurementTool === tool.type
                      ? 'text-[#a855f7] font-bold bg-[#181b23]'
                      : 'text-[#ededf2]'
                  }`}
                >
                  <span>{tool.label}</span>
                  {activeMeasurementTool === tool.type && (
                    <span className="text-[10px] text-[#a855f7]">Active</span>
                  )}
                </button>
              ))}
              {activeMeasurementTool && (
                <button
                  type="button"
                  onClick={() => {
                    setActiveMeasurementTool(null);
                    setMeasurementDropdownOpen(false);
                  }}
                  className="w-full text-center px-3 py-1.5 text-[10px] text-[#e74c3c] hover:bg-[#1f1515] border-t border-[#1f222b]"
                >
                  Clear Active Measurement Tool
                </button>
              )}
            </div>
          )}
        </div>

        <span className="text-[#1f222b]">|</span>

        {/* Panel Toggles */}
        <div className="flex items-center gap-0.5 bg-[#14161f] p-0.5 rounded border border-[#1f222b]">
          <button
            type="button"
            onClick={toggleOutliner}
            title={outlinerCollapsed ? 'Show Outliner (⌘B)' : 'Hide Outliner (⌘B)'}
            className={`p-1 rounded transition-colors ${
              !outlinerCollapsed ? 'bg-[#1f222b] text-[#3d8ef7]' : 'text-[#54596b] hover:text-[#ededf2]'
            }`}
          >
            <PanelLeft className="w-3 h-3" />
          </button>
          <button
            type="button"
            onClick={() => toggleBottomDrawer()}
            title={bottomDrawerOpen ? 'Close Bottom Drawer (⌘J)' : 'Open Bottom Drawer (⌘J)'}
            className={`p-1 rounded transition-colors ${
              bottomDrawerOpen ? 'bg-[#1f222b] text-[#3d8ef7]' : 'text-[#54596b] hover:text-[#ededf2]'
            }`}
          >
            <PanelBottom className="w-3 h-3" />
          </button>
          <button
            type="button"
            onClick={toggleInspector}
            title={inspectorCollapsed ? 'Show Inspector (⌘I)' : 'Hide Inspector (⌘I)'}
            className={`p-1 rounded transition-colors ${
              !inspectorCollapsed ? 'bg-[#1f222b] text-[#3d8ef7]' : 'text-[#54596b] hover:text-[#ededf2]'
            }`}
          >
            <PanelRight className="w-3 h-3" />
          </button>
        </div>

        {/* Command Palette Trigger */}
        <button
          type="button"
          onClick={() => setCommandPaletteOpen(true)}
          title="Command Palette (⌘K)"
          className="flex items-center gap-1 px-1.5 py-1 rounded bg-[#14161f] border border-[#1f222b] text-[#9296a6] hover:text-[#ededf2] transition-colors"
        >
          <Command className="w-2.5 h-2.5" />
          <span className="text-[10px]">⌘K</span>
        </button>
      </div>
    </div>
  );
}
