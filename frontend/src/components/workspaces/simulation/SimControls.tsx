'use client';

import React, { useState } from 'react';
import { Play, Pause, RotateCcw, StepForward, Activity, Settings2, Plus } from 'lucide-react';

export function SimControls() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState('1x');
  const [scenario, setScenario] = useState('Flood Event 1 - 100yr Inundation');

  return (
    <div className="flex items-center justify-between px-4 h-[44px] bg-[#121215] border-b border-[#1e1e28] select-none">
      {/* Left: Simulation State & Scenario Selector */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-2 py-1 rounded bg-[#17171c] border border-[#272733]">
          <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-[#34c76f] animate-pulse' : 'bg-[#f5a623]'}`} />
          <span className="text-[11px] font-mono tracking-wider text-[#e8e8f0]">
            {isPlaying ? 'SIMULATING' : 'PAUSED'}
          </span>
        </div>

        <select
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          className="bg-[#17171c] border border-[#272733] text-[#e8e8f0] text-[11px] rounded px-2.5 py-1 focus:outline-none focus:border-[#3d8ef7] cursor-pointer"
        >
          <option>Flood Event 1 - 100yr Inundation</option>
          <option>Peak Urban Traffic Dispersion</option>
          <option>Seismic Dynamic Load Analysis</option>
          <option>Thermal Microclimate Island Effect</option>
        </select>

        <button className="flex items-center gap-1 text-[11px] text-[#9898b0] hover:text-[#e8e8f0] px-2 py-1 bg-[#17171c] hover:bg-[#22222c] rounded border border-[#272733] transition-colors">
          <Plus className="w-3 h-3" />
          New Branch
        </button>
      </div>

      {/* Center: Playback Transport Controls */}
      <div className="flex items-center gap-1.5 bg-[#17171c] p-1 rounded border border-[#272733]">
        <button 
          onClick={() => setIsPlaying(false)}
          className="p-1.5 hover:bg-[#22222c] rounded text-[#9898b0] hover:text-[#e8e8f0] transition-colors"
          title="Reset to T=0"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        <button 
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-3 py-1 bg-[#3d8ef7] hover:bg-[#5ba3fa] text-white rounded text-[11px] font-medium flex items-center gap-1.5 transition-colors"
        >
          {isPlaying ? (
            <>
              <Pause className="w-3 h-3 fill-current" />
              Pause
            </>
          ) : (
            <>
              <Play className="w-3 h-3 fill-current" />
              Simulate
            </>
          )}
        </button>

        <button 
          className="p-1.5 hover:bg-[#22222c] rounded text-[#9898b0] hover:text-[#e8e8f0] transition-colors"
          title="Step Single Tick (+100ms)"
        >
          <StepForward className="w-3.5 h-3.5" />
        </button>

        {/* Speed multiplier buttons */}
        <div className="flex items-center pl-2 ml-1 border-l border-[#272733] gap-1">
          {['0.5x', '1x', '2x', '5x'].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${
                speed === s ? 'bg-[#2a2a36] text-[#3d8ef7] font-bold' : 'text-[#5c5c78] hover:text-[#9898b0]'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Right: Simulation Time Clock & Status */}
      <div className="flex items-center gap-4 text-xs font-mono">
        <div className="flex flex-col items-end">
          <span className="text-[10px] text-[#5c5c78] uppercase">Tick / Simulated Clock</span>
          <span className="text-[#3d8ef7] font-semibold tracking-wider">T+02:45:12.400 / 04:00:00</span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-1 bg-[#17171c] rounded border border-[#272733] text-[10px] text-[#9898b0]">
          <Activity className="w-3 h-3 text-[#34c76f]" />
          <span>60 Hz Solver</span>
        </div>
      </div>
    </div>
  );
}
