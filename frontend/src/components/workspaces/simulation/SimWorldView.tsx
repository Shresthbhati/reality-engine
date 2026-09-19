'use client';

import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Layers, Compass } from 'lucide-react';

interface Agent {
  x: number;
  y: number;
  vx: number;
  vy: number;
  status: 'safe' | 'warning' | 'danger';
}

export function SimWorldView() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [showAgents, setShowAgents] = useState(true);
  const [showFlood, setShowFlood] = useState(true);
  const [showGrid, setShowGrid] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = canvas.parentElement?.clientWidth || 800);
    let height = (canvas.height = canvas.parentElement?.clientHeight || 600);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.clientWidth;
      height = canvas.height = canvas.parentElement.clientHeight;
    };
    window.addEventListener('resize', handleResize);

    // Generate 60 dynamic agents
    const agents: Agent[] = Array.from({ length: 60 }).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.8,
      vy: (Math.random() - 0.5) * 0.8,
      status: Math.random() > 0.7 ? 'warning' : Math.random() > 0.9 ? 'danger' : 'safe',
    }));

    let floodRadius = 80;

    const render = () => {
      ctx.fillStyle = '#0d0d0f';
      ctx.fillRect(0, 0, width, height);

      // 1. Grid
      if (showGrid) {
        ctx.strokeStyle = '#1a1a24';
        ctx.lineWidth = 1;
        const step = 40;
        for (let x = 0; x < width; x += step) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }
        for (let y = 0; y < height; y += step) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }
      }

      // 2. City Road Network Outline
      ctx.strokeStyle = '#272733';
      ctx.lineWidth = 16;
      ctx.beginPath();
      // Main arterial
      ctx.moveTo(0, height * 0.45);
      ctx.lineTo(width, height * 0.45);
      // Secondary cross road
      ctx.moveTo(width * 0.35, 0);
      ctx.lineTo(width * 0.35, height);
      ctx.stroke();

      // Road markings
      ctx.strokeStyle = '#34344a';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.beginPath();
      ctx.moveTo(0, height * 0.45);
      ctx.lineTo(width, height * 0.45);
      ctx.moveTo(width * 0.35, 0);
      ctx.lineTo(width * 0.35, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // 3. Buildings Blocks
      const buildings = [
        { x: width * 0.1, y: height * 0.15, w: 120, h: 90, name: 'Building A' },
        { x: width * 0.42, y: height * 0.12, w: 160, h: 100, name: 'Warehouse B' },
        { x: width * 0.42, y: height * 0.55, w: 140, h: 120, name: 'Substation 4' },
        { x: width * 0.12, y: height * 0.58, w: 110, h: 80, name: 'Control Room' },
      ];

      buildings.forEach((b) => {
        ctx.fillStyle = '#17171c';
        ctx.strokeStyle = '#34344a';
        ctx.lineWidth = 1.5;
        ctx.fillRect(b.x, b.y, b.w, b.h);
        ctx.strokeRect(b.x, b.y, b.w, b.h);

        ctx.fillStyle = '#9898b0';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText(b.name, b.x + 8, b.y + 18);
      });

      // 4. Hydrodynamic Inundation Spread
      if (showFlood) {
        floodRadius = 90 + Math.sin(Date.now() * 0.001) * 15;
        const floodGradient = ctx.createRadialGradient(
          width * 0.8,
          height * 0.3,
          10,
          width * 0.8,
          height * 0.3,
          floodRadius + 140
        );
        floodGradient.addColorStop(0, 'rgba(61, 142, 247, 0.55)');
        floodGradient.addColorStop(0.6, 'rgba(61, 142, 247, 0.25)');
        floodGradient.addColorStop(1, 'rgba(61, 142, 247, 0)');

        ctx.fillStyle = floodGradient;
        ctx.beginPath();
        ctx.arc(width * 0.8, height * 0.3, floodRadius + 140, 0, Math.PI * 2);
        ctx.fill();

        // High water contour line
        ctx.strokeStyle = '#5edaff';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(width * 0.8, height * 0.3, floodRadius + 90, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 5. Dynamic Moving Agents
      if (showAgents) {
        agents.forEach((agent) => {
          agent.x += agent.vx;
          agent.y += agent.vy;

          if (agent.x < 10 || agent.x > width - 10) agent.vx *= -1;
          if (agent.y < 10 || agent.y > height - 10) agent.vy *= -1;

          ctx.beginPath();
          ctx.arc(agent.x, agent.y, 3, 0, Math.PI * 2);

          if (agent.status === 'danger') {
            ctx.fillStyle = '#e54d4d';
            ctx.shadowColor = '#e54d4d';
            ctx.shadowBlur = 6;
          } else if (agent.status === 'warning') {
            ctx.fillStyle = '#f5a623';
            ctx.shadowColor = '#f5a623';
            ctx.shadowBlur = 4;
          } else {
            ctx.fillStyle = '#34c76f';
            ctx.shadowColor = '#34c76f';
            ctx.shadowBlur = 2;
          }

          ctx.fill();
          ctx.shadowBlur = 0; // reset
        });
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, [showAgents, showFlood, showGrid]);

  return (
    <div className="relative w-full h-full overflow-hidden select-none bg-[#0d0d0f]">
      <canvas ref={canvasRef} className="block w-full h-full" />

      {/* Top Left Viewport Stats Overlay */}
      <div className="absolute top-3 left-3 bg-[#121215]/85 backdrop-blur-sm border border-[#272733] rounded px-3 py-2 flex flex-col gap-1 text-[11px] font-mono">
        <div className="flex items-center gap-2 text-[#e8e8f0]">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <span>SIMULATION VIEWPORT: 2D TOP-DOWN</span>
        </div>
        <div className="text-[10px] text-[#9898b0]">
          Grid resolution: 2.5m | Projection: EPSG:3857 (WGS 84 / Pseudo-Mercator)
        </div>
      </div>

      {/* Floating View Toggles */}
      <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-[#121215]/90 backdrop-blur-sm border border-[#272733] p-1 rounded">
        <button
          onClick={() => setShowFlood(!showFlood)}
          className={`px-2 py-1 text-[10px] font-mono rounded ${
            showFlood ? 'bg-[#3d8ef7] text-white' : 'text-[#9898b0] hover:text-[#e8e8f0]'
          }`}
        >
          Flood Layer
        </button>
        <button
          onClick={() => setShowAgents(!showAgents)}
          className={`px-2 py-1 text-[10px] font-mono rounded ${
            showAgents ? 'bg-[#3d8ef7] text-white' : 'text-[#9898b0] hover:text-[#e8e8f0]'
          }`}
        >
          Agents
        </button>
        <button
          onClick={() => setShowGrid(!showGrid)}
          className={`px-2 py-1 text-[10px] font-mono rounded ${
            showGrid ? 'bg-[#3d8ef7] text-white' : 'text-[#9898b0] hover:text-[#e8e8f0]'
          }`}
        >
          Grid
        </button>
      </div>

      {/* Bottom Floating Legend */}
      <div className="absolute bottom-3 left-3 bg-[#121215]/90 backdrop-blur-sm border border-[#272733] px-3 py-1.5 rounded flex items-center gap-4 text-[10px] font-mono">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#34c76f]" />
          <span className="text-[#9898b0]">Safe Agents (44)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#f5a623]" />
          <span className="text-[#9898b0]">Evacuation Route (12)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#e54d4d]" />
          <span className="text-[#9898b0]">Inundated Zone (4)</span>
        </div>
      </div>
    </div>
  );
}
