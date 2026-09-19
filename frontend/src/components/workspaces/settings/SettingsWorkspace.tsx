'use client';

import React, { useState } from 'react';
import { Settings, Globe, Cpu, Sliders, Database, HardDrive, Shield, Check, Save } from 'lucide-react';

export function SettingsWorkspace() {
  const [crs, setCrs] = useState('EPSG:32645 (WGS 84 / UTM Zone 45N)');
  const [units, setUnits] = useState('SI Metric (Meters, Radians, Kilograms)');
  const [gpuDevice, setGpuDevice] = useState('NVIDIA GeForce RTX 4090 (24GB VRAM)');
  const [precision, setPrecision] = useState('Float64 (High Numerical Fidelity)');
  const [maxVoxelResolution, setMaxVoxelResolution] = useState('0.02 m (2 cm)');
  const [compilerTarget, setCompilerTarget] = useState('OpenUSD + CityGML LOD2');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#0d0d0f] text-[#e8e8f0] select-none overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 bg-[#121215] border-b border-[#1e1e28]">
        <div className="flex items-center gap-3">
          <Settings className="w-5 h-5 text-[#3d8ef7]" />
          <div>
            <h1 className="text-sm font-semibold tracking-wider uppercase text-[#e8e8f0]">
              Reality Engine Configuration & Standards
            </h1>
            <p className="text-xs text-[#9898b0]">
              Global parameters, coordinate reference frames, numerical solver guards, and export compilers
            </p>
          </div>
        </div>

        <button
          onClick={handleSave}
          className="flex items-center gap-2 px-4 py-2 bg-[#3d8ef7] hover:bg-[#5ba3fa] text-white rounded text-xs font-medium transition-colors shadow-[0_0_12px_rgba(61,142,247,0.3)]"
        >
          {saved ? <Check className="w-4 h-4 text-white" /> : <Save className="w-4 h-4" />}
          <span>{saved ? 'Preferences Saved' : 'Save Preferences'}</span>
        </button>
      </div>

      {/* Main Settings Body */}
      <div className="max-w-4xl p-6 space-y-6">
        {/* Section 1: Geospatial & Coordinate Systems */}
        <div className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#9898b0] flex items-center gap-2">
            <Globe className="w-4 h-4 text-[#3d8ef7]" />
            Geospatial Reference Frame (CRS)
          </h2>

          <div className="bg-[#121215] p-4 rounded border border-[#1e1e28] space-y-3">
            <div>
              <label className="block text-xs font-medium text-[#e8e8f0] mb-1">
                Project Coordinate Reference System
              </label>
              <select
                value={crs}
                onChange={(e) => setCrs(e.target.value)}
                className="w-full bg-[#17171c] border border-[#272733] text-xs font-mono rounded px-3 py-2 text-[#e8e8f0] focus:outline-none focus:border-[#3d8ef7]"
              >
                <option>EPSG:32645 (WGS 84 / UTM Zone 45N) - Eastern India</option>
                <option>EPSG:3857 (WGS 84 / Pseudo-Mercator - WebGIS)</option>
                <option>EPSG:4326 (WGS 84 Geodetic - Latitude / Longitude)</option>
                <option>LOCAL_ENU (Local East-North-Up Tangent Plane)</option>
                <option>ECEF (Earth-Centered Earth-Fixed Cartesian)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#e8e8f0] mb-1">
                Physical Units & Dimensional Enforcement
              </label>
              <select
                value={units}
                onChange={(e) => setUnits(e.target.value)}
                className="w-full bg-[#17171c] border border-[#272733] text-xs font-mono rounded px-3 py-2 text-[#e8e8f0] focus:outline-none focus:border-[#3d8ef7]"
              >
                <option>SI Metric (Meters, Radians, Kilograms, Pascals)</option>
                <option>US Survey (Feet, Degrees, Pounds)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 2: Compute Acceleration & Hardware */}
        <div className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#9898b0] flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#34c76f]" />
            Compute & Photogrammetry Backends
          </h2>

          <div className="bg-[#121215] p-4 rounded border border-[#1e1e28] space-y-3">
            <div>
              <label className="block text-xs font-medium text-[#e8e8f0] mb-1">
                Primary CUDA / Metal Compute Device
              </label>
              <select
                value={gpuDevice}
                onChange={(e) => setGpuDevice(e.target.value)}
                className="w-full bg-[#17171c] border border-[#272733] text-xs font-mono rounded px-3 py-2 text-[#e8e8f0] focus:outline-none focus:border-[#3d8ef7]"
              >
                <option>NVIDIA GeForce RTX 4090 (24GB VRAM) — Direct Compute</option>
                <option>Apple Metal / MPS — Unified Memory 64GB</option>
                <option>Vulkan Compute Headless (CPU Fallback)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-[#e8e8f0] mb-1">
                Spatial Discretization / Voxel Grid
              </label>
              <select
                value={maxVoxelResolution}
                onChange={(e) => setMaxVoxelResolution(e.target.value)}
                className="w-full bg-[#17171c] border border-[#272733] text-xs font-mono rounded px-3 py-2 text-[#e8e8f0] focus:outline-none focus:border-[#3d8ef7]"
              >
                <option>0.01 m (1 cm - Ultra High Fidelity / Architectural Detail)</option>
                <option>0.02 m (2 cm - Production Default)</option>
                <option>0.05 m (5 cm - City Scale / Rapid Survey)</option>
                <option>0.10 m (10 cm - Regional Terrain)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 3: Export Compilers & Interoperability */}
        <div className="space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#9898b0] flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-[#f5a623]" />
            Reality Compilers & Downstream Pipelines
          </h2>

          <div className="bg-[#121215] p-4 rounded border border-[#1e1e28] space-y-3">
            <div>
              <label className="block text-xs font-medium text-[#e8e8f0] mb-1">
                Default WorldIR Output Format
              </label>
              <select
                value={compilerTarget}
                onChange={(e) => setCompilerTarget(e.target.value)}
                className="w-full bg-[#17171c] border border-[#272733] text-xs font-mono rounded px-3 py-2 text-[#e8e8f0] focus:outline-none focus:border-[#3d8ef7]"
              >
                <option>OpenUSD (Pixar USD Universal Scene Description)</option>
                <option>IFC 4x3 (Industry Foundation Classes - BIM / AEC)</option>
                <option>glTF 2.0 (Khronos Binary with KHR_draco_mesh_compression)</option>
                <option>Unreal Engine 5 Nanite Mesh Package</option>
                <option>ROS 2 URDF / SDF Autonomous Robotics Package</option>
                <option>SUMO Urban Mobility Simulation Network</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SettingsWorkspace;
