'use client';

import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Compass, Layers, Crosshair } from 'lucide-react';
import type { CityTool } from './CityToolPalette';

interface CityMapViewProps {
  activeTool: CityTool;
}

export function CityMapView({ activeTool }: CityMapViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState(1);
  const [showSessionCoverage, setShowSessionCoverage] = useState(true);
  const [showZoning, setShowZoning] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 600);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
      draw();
    };
    window.addEventListener('resize', handleResize);

    const draw = () => {
      ctx.save();
      ctx.fillStyle = '#0d0d0f';
      ctx.fillRect(0, 0, width, height);

      // Apply zoom & pan translation
      ctx.translate(width / 2, height / 2);
      ctx.scale(zoom, zoom);
      ctx.translate(-width / 2, -height / 2);

      // 1. Coordinate Grid
      ctx.strokeStyle = '#181822';
      ctx.lineWidth = 1;
      const step = 30;
      for (let x = -width; x < width * 2; x += step) {
        ctx.beginPath();
        ctx.moveTo(x, -height);
        ctx.lineTo(x, height * 2);
        ctx.stroke();
      }
      for (let y = -height; y < height * 2; y += step) {
        ctx.beginPath();
        ctx.moveTo(-width, y);
        ctx.lineTo(width * 2, y);
        ctx.stroke();
      }

      // 2. Multi-Session Coverage Polygons
      if (showSessionCoverage) {
        // Drone 001 Coverage Footprint
        ctx.fillStyle = 'rgba(61, 142, 247, 0.08)';
        ctx.strokeStyle = 'rgba(61, 142, 247, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.rect(width * 0.15, height * 0.15, width * 0.7, height * 0.65);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#3d8ef7';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText('COVERAGE: Drone_Session_001 (4,821 Imgs)', width * 0.16, height * 0.18);

        // Phone 002 Coverage Footprint (Localized street level)
        ctx.fillStyle = 'rgba(52, 199, 111, 0.1)';
        ctx.strokeStyle = 'rgba(52, 199, 111, 0.5)';
        ctx.beginPath();
        ctx.rect(width * 0.2, height * 0.35, width * 0.35, height * 0.3);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#34c76f';
        ctx.fillText('COVERAGE: Phone_Session_002 (1,239 Imgs)', width * 0.21, height * 0.38);
      }

      // 3. Zoning Parcels
      if (showZoning) {
        const parcels = [
          { x: width * 0.22, y: height * 0.22, w: 180, h: 140, zone: 'M-1 INDUSTRIAL' },
          { x: width * 0.55, y: height * 0.22, w: 220, h: 180, zone: 'C-2 COMMERCIAL' },
          { x: width * 0.55, y: height * 0.52, w: 190, h: 140, zone: 'U-3 UTILITY / LOGISTICS' },
        ];

        parcels.forEach((p) => {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.02)';
          ctx.strokeStyle = '#272733';
          ctx.lineWidth = 1;
          ctx.fillRect(p.x, p.y, p.w, p.h);
          ctx.strokeRect(p.x, p.y, p.w, p.h);

          ctx.fillStyle = '#5c5c78';
          ctx.font = '9px JetBrains Mono, monospace';
          ctx.fillText(p.zone, p.x + 6, p.y + 14);
        });
      }

      // 4. Roads Network
      ctx.strokeStyle = '#22222c';
      ctx.lineWidth = 22;
      ctx.beginPath();
      ctx.moveTo(0, height * 0.48);
      ctx.lineTo(width, height * 0.48);
      ctx.moveTo(width * 0.48, 0);
      ctx.lineTo(width * 0.48, height);
      ctx.stroke();

      // Road curbs & centerlines
      ctx.strokeStyle = '#34344a';
      ctx.lineWidth = 2;
      ctx.setLineDash([10, 8]);
      ctx.beginPath();
      ctx.moveTo(0, height * 0.48);
      ctx.lineTo(width, height * 0.48);
      ctx.moveTo(width * 0.48, 0);
      ctx.lineTo(width * 0.48, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // 5. Buildings 2.5D Footprints
      const buildings = [
        { x: width * 0.25, y: height * 0.25, w: 130, h: 90, h3d: 28, name: 'Building A (Main Admin)' },
        { x: width * 0.58, y: height * 0.25, w: 160, h: 120, h3d: 14, name: 'Warehouse B' },
        { x: width * 0.26, y: height * 0.55, w: 100, h: 80, h3d: 18, name: 'Substation 04' },
      ];

      buildings.forEach((b) => {
        // Shadow/depth extrusion
        ctx.fillStyle = '#121215';
        ctx.fillRect(b.x + 6, b.y + 6, b.w, b.h);

        // Building top footprint
        ctx.fillStyle = '#17171c';
        ctx.strokeStyle = '#3d8ef7';
        ctx.lineWidth = 1.5;
        ctx.fillRect(b.x, b.y, b.w, b.h);
        ctx.strokeRect(b.x, b.y, b.w, b.h);

        ctx.fillStyle = '#e8e8f0';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(b.name, b.x + 8, b.y + 22);

        ctx.fillStyle = '#9898b0';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.fillText(`Height: ${b.h3d}m | LOD2 Mesh`, b.x + 8, b.y + 38);
      });

      // 6. Active Tool Visual Indicator
      ctx.fillStyle = '#3d8ef7';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText(`TOOL ACTIVE: [${activeTool}]`, 20, height - 20);

      ctx.restore();
    };

    draw();

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [zoom, showSessionCoverage, showZoning, activeTool]);

  return (
    <div className="relative w-full h-full bg-[#0d0d0f] overflow-hidden select-none">
      <canvas ref={canvasRef} className="block w-full h-full cursor-crosshair" />

      {/* Top Left Compass & Coordinate Info */}
      <div className="absolute top-3 left-3 bg-[#121215]/85 backdrop-blur-sm border border-[#272733] rounded px-3 py-2 flex items-center gap-3 text-xs font-mono">
        <Compass className="w-5 h-5 text-[#3d8ef7]" />
        <div>
          <div className="text-[#e8e8f0] font-semibold">KOLKATA METROPOLITAN WORLD</div>
          <div className="text-[10px] text-[#5c5c78]">CRS: EPSG:32645 (WGS 84 / UTM Zone 45N)</div>
        </div>
      </div>

      {/* Top Right Zoom Controls */}
      <div className="absolute top-3 right-3 flex items-center gap-1 bg-[#121215]/90 backdrop-blur-sm border border-[#272733] p-1 rounded">
        <button
          onClick={() => setZoom((z) => Math.min(3, z + 0.2))}
          className="p-1.5 hover:bg-[#22222c] rounded text-[#9898b0] hover:text-[#e8e8f0]"
          title="Zoom In"
        >
          <ZoomIn className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(0.4, z - 0.2))}
          className="p-1.5 hover:bg-[#22222c] rounded text-[#9898b0] hover:text-[#e8e8f0]"
          title="Zoom Out"
        >
          <ZoomOut className="w-4 h-4" />
        </button>
        <button
          onClick={() => setZoom(1)}
          className="p-1.5 hover:bg-[#22222c] rounded text-[#9898b0] hover:text-[#e8e8f0]"
          title="Fit Extents"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Bottom Floating Layer Toggles */}
      <div className="absolute bottom-3 right-3 flex items-center gap-2 bg-[#121215]/90 backdrop-blur-sm border border-[#272733] px-2 py-1 rounded text-[10px] font-mono">
        <label className="flex items-center gap-1.5 cursor-pointer text-[#9898b0] hover:text-[#e8e8f0]">
          <input
            type="checkbox"
            checked={showSessionCoverage}
            onChange={(e) => setShowSessionCoverage(e.target.checked)}
            className="accent-[#3d8ef7]"
          />
          Session Footprints
        </label>
        <span className="text-[#272733]">|</span>
        <label className="flex items-center gap-1.5 cursor-pointer text-[#9898b0] hover:text-[#e8e8f0]">
          <input
            type="checkbox"
            checked={showZoning}
            onChange={(e) => setShowZoning(e.target.checked)}
            className="accent-[#3d8ef7]"
          />
          Parcels
        </label>
      </div>
    </div>
  );
}
