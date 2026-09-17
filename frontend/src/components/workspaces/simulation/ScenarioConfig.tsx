'use client';

import React, { useState } from 'react';
import { Sliders, Droplets, Wind, Users, Car, CheckCircle2, Play } from 'lucide-react';

export function ScenarioConfig() {
  const [rainfall, setRainfall] = useState(180);
  const [duration, setDuration] = useState(4);
  const [riverLevel, setRiverLevel] = useState(3.2);
  const [permeability, setPermeability] = useState(0.42);
  const [population, setPopulation] = useState(12402);
  const [vehicles, setVehicles] = useState(3891);

  return (
    <div className="flex flex-col h-full text-[#e8e8f0] p-3 gap-4 select-none">
      <div className="flex items-center justify-between pb-2 border-b border-[#1e1e28]">
        <div className="flex items-center gap-2">
          <Sliders className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span className="text-[11px] font-semibold tracking-wider text-[#e8e8f0] uppercase">
            SCENARIO PARAMETERS
          </span>
        </div>
        <span className="text-[10px] text-[#34c76f] bg-green-500/10 px-1.5 py-0.5 rounded border border-green-500/20 font-mono">
          VALIDATED
        </span>
      </div>

      {/* Hydro-physical parameters */}
      <div className="space-y-3">
        <span className="text-[10px] font-semibold text-[#9898b0] uppercase tracking-wider flex items-center gap-1.5">
          <Droplets className="w-3 h-3 text-[#5edaff]" />
          Hydrodynamic Boundary Conditions
        </span>

        <div className="space-y-2 bg-[#17171c] p-2.5 rounded border border-[#272733]">
          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-[#9898b0]">Precipitation Rate</span>
              <span className="font-mono text-[#3d8ef7]">{rainfall} mm/h</span>
            </div>
            <input
              type="range"
              min="0"
              max="350"
              value={rainfall}
              onChange={(e) => setRainfall(Number(e.target.value))}
              className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-[#9898b0]">Storm Duration</span>
              <span className="font-mono text-[#3d8ef7]">{duration} Hours</span>
            </div>
            <input
              type="range"
              min="1"
              max="24"
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-[#9898b0]">River Crest Height</span>
              <span className="font-mono text-[#3d8ef7]">+{riverLevel.toFixed(1)} m MSL</span>
            </div>
            <input
              type="range"
              min="0"
              max="10"
              step="0.1"
              value={riverLevel}
              onChange={(e) => setRiverLevel(Number(e.target.value))}
              className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
            />
          </div>

          <div>
            <div className="flex justify-between text-[11px] mb-1">
              <span className="text-[#9898b0]">Soil Infiltration (Runoff)</span>
              <span className="font-mono text-[#3d8ef7]">{permeability}</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={permeability}
              onChange={(e) => setPermeability(Number(e.target.value))}
              className="w-full accent-[#3d8ef7] h-1.5 bg-[#272733] rounded cursor-pointer"
            />
          </div>
        </div>
      </div>

      {/* Agents & Evacuation Capacity */}
      <div className="space-y-3">
        <span className="text-[10px] font-semibold text-[#9898b0] uppercase tracking-wider flex items-center gap-1.5">
          <Users className="w-3 h-3 text-[#c77df5]" />
          Agent Population Dynamics
        </span>

        <div className="grid grid-cols-2 gap-2 bg-[#17171c] p-2.5 rounded border border-[#272733]">
          <div className="space-y-1">
            <span className="text-[10px] text-[#5c5c78]">Resident Agents</span>
            <input 
              type="number" 
              value={population}
              onChange={(e) => setPopulation(Number(e.target.value))}
              className="w-full bg-[#0d0d0f] border border-[#272733] rounded px-2 py-1 text-xs font-mono text-[#e8e8f0]" 
            />
          </div>
          <div className="space-y-1">
            <span className="text-[10px] text-[#5c5c78]">Active Vehicles</span>
            <input 
              type="number" 
              value={vehicles}
              onChange={(e) => setVehicles(Number(e.target.value))}
              className="w-full bg-[#0d0d0f] border border-[#272733] rounded px-2 py-1 text-xs font-mono text-[#e8e8f0]" 
            />
          </div>
        </div>
      </div>

      {/* Physics Engine Backend Target */}
      <div className="bg-[#17171c] p-2.5 rounded border border-[#272733] space-y-1.5">
        <div className="text-[10px] text-[#5c5c78] uppercase">Solver Backend</div>
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-[#3d8ef7]">RTE Fluid-Dynamic V11</span>
          <span className="text-green-400 flex items-center gap-1 text-[10px]">
            <CheckCircle2 className="w-3 h-3" /> Ready
          </span>
        </div>
        <div className="text-[9px] text-[#5c5c78]">
          Mesh: Kolkata_Site_LOD2 | Cell Grid: 2.5m x 2.5m
        </div>
      </div>

      {/* Actions */}
      <div className="pt-2">
        <button className="w-full py-2 bg-[#3d8ef7] hover:bg-[#5ba3fa] text-white rounded text-xs font-medium flex items-center justify-center gap-2 transition-colors shadow-[0_0_12px_rgba(61,142,247,0.3)]">
          <Play className="w-3.5 h-3.5 fill-current" />
          Re-Run Coupled Simulation
        </button>
        <span className="block text-center text-[9px] text-[#5c5c78] mt-1.5 font-mono">
          Estimated solve time: 18.2s on CUDA RTX
        </span>
      </div>
    </div>
  );
}
