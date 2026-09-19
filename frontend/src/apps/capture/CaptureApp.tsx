'use client';

/**
 * CaptureApp — Reality Engine Mobile Reality-Capture Instrument.
 * Designed with iPhone Camera-level elegance + engineering-grade survey precision.
 * 
 * Features:
 * - Multi-Pass Capture Guidance:
 *   - SITE PASS (Wide perimeter, GNSS geotagging, context loop)
 *   - STRUCTURE PASS (Full elevation orbits, oblique angles)
 *   - FACADE PASS (Orthogonal grid, 70-80% overlap)
 *   - DETAIL PASS (Close-range ring around capitals, moldings, reliefs)
 *   - MICRO-DETAIL PASS (Photomacrography, cracks, surface weathering)
 * - Live Density & Overlap Guidance Cues:
 *   - "Your current capture resolves the facade, but not the decorative capitals."
 *   - "Move closer. Maintain 60-80% overlap."
 * - Precision Telemetry HUD:
 *   - RTK Fixed Carrier-Phase GNSS (± 0.016 m)
 *   - Real-time IMU Artificial Horizon (Pitch, Roll, Heading)
 *   - Thermal envelope, battery, storage budget & frame drop monitor
 * - Multi-lens switching (13mm / 24mm / 77mm telephoto/macro)
 * - Device frame toggle (Simulated iPhone Pro Handset vs Full-Screen Viewport)
 */

import React, { useState, useEffect } from 'react';
import {
  Camera,
  Compass,
  Radio,
  Clock,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  ChevronRight,
  RotateCcw,
  Smartphone,
  Eye,
  Sliders,
  Play,
  Square,
  ArrowLeft,
  Layers,
  MapPin,
  UploadCloud,
  Check,
  Zap,
  Crosshair,
  Maximize2,
  Info,
  ChevronDown,
} from 'lucide-react';
import { useREStore, CapturePassType } from '@/store/re-store';

export default function CaptureApp() {
  const {
    captureScreen,
    setCaptureScreen,
    mobileSensors,
    liveCaptureQuality,
    cameraLens,
    setCameraLens,
    isRecording,
    toggleRecording,
    triggerPhotoCapture,
    simulateMobileDevice,
    toggleSimulateMobileDevice,
    showCoverageHud,
    toggleCoverageHud,
    showQualityDetails,
    toggleQualityDetails,
    activeCapturePass,
    setActiveCapturePass,
    setActiveWorkspace,
  } = useREStore();

  const [simulatedHeading, setSimulatedHeading] = useState(148.4);
  const [levelPitch, setLevelPitch] = useState(0.4);
  const [levelRoll, setLevelRoll] = useState(-0.2);
  const [shutterSpeed, setShutterSpeed] = useState('1/1200s');
  const [isoVal, setIsoVal] = useState('ISO 100');
  const [focusDistanceM, setFocusDistanceM] = useState('1.2m');
  const [wbTemp, setWbTemp] = useState('5200K');

  // Multi-pass guidance cues dynamically reacting to current pass and camera
  const PASS_DESCRIPTIONS: Record<CapturePassType, { title: string; hint: string; recommendedLens: string; requiredOverlap: string }> = {
    SITE_PASS: {
      title: 'Site Pass (Perimeter Context)',
      hint: 'Maintain 30-50m standoff distance. Complete full perimeter circle with 60% overlap.',
      recommendedLens: '0.5x (13mm)',
      requiredOverlap: '60% Forward / 50% Side',
    },
    STRUCTURE_PASS: {
      title: 'Structure Pass (Elevation Orbits)',
      hint: 'Orbit building at 45° oblique angle. Capture transition from plinth to main roof line.',
      recommendedLens: '1x (24mm)',
      requiredOverlap: '70% Forward / 65% Side',
    },
    FACADE_PASS: {
      title: 'Facade Pass (Orthogonal Grid)',
      hint: 'Keep camera axis orthogonal to wall plane. Maintain constant distance across lateral sweeps.',
      recommendedLens: '1x (24mm)',
      requiredOverlap: '75-80% Forward / 70% Side',
    },
    DETAIL_PASS: {
      title: 'Detail Pass (Capitals & Moldings)',
      hint: 'Current capture resolves facade massing, but not capitals. Move to 1.5m and sweep full 180° arc.',
      recommendedLens: '3x (77mm)',
      requiredOverlap: '85% Overlap / 12-16 angles',
    },
    MICRO_DETAIL_PASS: {
      title: 'Micro-Detail Pass (Relief & Weathering)',
      hint: 'Macro standoff < 0.3m. Ensure razor-sharp focus on carved marble relief and surface weathering.',
      recommendedLens: '3x (77mm Macro)',
      requiredOverlap: '90% Overlap / Fixed distance',
    },
  };

  // Sensor telemetry fluctuations for realistic instrument response
  useEffect(() => {
    const interval = setInterval(() => {
      setSimulatedHeading((prev) => (prev + 0.15) % 360);
      setLevelPitch((prev) => prev + (Math.random() - 0.5) * 0.08);
      setLevelRoll((prev) => prev + (Math.random() - 0.5) * 0.06);
    }, 150);
    return () => clearInterval(interval);
  }, []);

  const currentPassInfo = PASS_DESCRIPTIONS[activeCapturePass];

  return (
    <div className="flex flex-col w-full h-full bg-[#08090b] text-[#f0f1f6] overflow-hidden select-none font-sans">
      {/* ── Top Instrument Chrome (36px) ─────────────────────────────── */}
      <header className="h-9 px-3.5 flex items-center justify-between border-b border-[#1f222b] bg-[#0d0e12] shrink-0 text-xs">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold text-[#3d8ef7] tracking-wider">
            <Camera className="w-3.5 h-3.5" />
            <span>REALITY CAPTURE INSTRUMENT</span>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/30 font-mono">
            RTK FIXED ±0.016m
          </span>
          <span className="text-[10px] font-mono text-[#9296a6] hidden sm:inline">
            EPSG:32645 (WGS84 UTM 45N)
          </span>
        </div>

        <div className="flex items-center gap-2 text-[11px] font-mono text-[#9296a6]">
          <button
            type="button"
            onClick={toggleSimulateMobileDevice}
            className={`px-2 py-1 rounded border text-[10px] flex items-center gap-1 transition-colors ${
              simulateMobileDevice
                ? 'bg-[#3d8ef7]/20 border-[#3d8ef7] text-[#3d8ef7]'
                : 'bg-[#14161f] border-[#262a36] text-[#9296a6] hover:text-[#f0f1f6]'
            }`}
          >
            <Smartphone className="w-3 h-3" />
            <span>{simulateMobileDevice ? 'Handset Frame' : 'Full Canvas'}</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveWorkspace('loader')}
            className="px-2 py-1 rounded bg-[#1b1e27] hover:bg-[#252936] text-[#ededf2] border border-[#2b3040] text-[10px] flex items-center gap-1.5 transition-colors"
          >
            <UploadCloud className="w-3 h-3 text-[#2ecc71]" />
            <span>Offload to Loader</span>
          </button>
        </div>
      </header>

      {/* ── Pass Selector Ribbon (34px) ──────────────────────────────── */}
      <div className="h-8.5 px-3 bg-[#111318] border-b border-[#1f222b] flex items-center justify-between overflow-x-auto text-[11px] font-mono shrink-0">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] uppercase text-[#54596b] mr-1">Capture Pass:</span>
          {(['SITE_PASS', 'STRUCTURE_PASS', 'FACADE_PASS', 'DETAIL_PASS', 'MICRO_DETAIL_PASS'] as CapturePassType[]).map(
            (pass) => {
              const active = activeCapturePass === pass;
              return (
                <button
                  key={pass}
                  type="button"
                  onClick={() => setActiveCapturePass(pass)}
                  className={`px-2 py-0.5 rounded text-[10px] font-bold transition-all ${
                    active
                      ? 'bg-[#3d8ef7] text-[#08090b] shadow-sm'
                      : 'bg-[#171922] text-[#9296a6] hover:text-[#f0f1f6] border border-[#222633]'
                  }`}
                >
                  {pass.replace('_PASS', '')}
                </button>
              );
            }
          )}
        </div>

        <div className="flex items-center gap-2 text-[10px] text-[#2ecc71]">
          <CheckCircle2 className="w-3 h-3" />
          <span>IMU Calibration Valid</span>
        </div>
      </div>

      {/* ── Main Instrument Body ──────────────────────────────────────── */}
      <div className="flex-1 min-h-0 w-full flex items-center justify-center p-2 sm:p-4 bg-[#050608] overflow-hidden">
        {/* Device Frame Wrapper */}
        <div
          className={`relative transition-all duration-300 flex flex-col overflow-hidden shadow-2xl ${
            simulateMobileDevice
              ? 'w-[390px] h-[780px] max-h-full rounded-[44px] border-[10px] border-[#22242c] bg-black ring-1 ring-white/10'
              : 'w-full h-full rounded-lg border border-[#1f222b] bg-black'
          }`}
        >
          {/* Simulated Dynamic Island (if phone frame enabled) */}
          {simulateMobileDevice && (
            <div className="absolute top-2 left-1/2 -translate-x-1/2 w-28 h-6 bg-black rounded-full z-40 flex items-center justify-between px-2.5 border border-white/10">
              <span className="w-2.5 h-2.5 rounded-full bg-[#111] border border-white/20" />
              <span className="text-[9px] font-mono text-[#2ecc71] flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-[#2ecc71] animate-pulse" />
                REC
              </span>
            </div>
          )}

          {/* ── Camera Viewfinder (Simulated Reality Sensor) ──────────── */}
          <div className="relative flex-1 w-full bg-[#0a0c10] overflow-hidden flex items-center justify-center">
            {/* Viewfinder Background Imagery / Grid Simulation */}
            <div className="absolute inset-0 opacity-25 bg-[radial-gradient(#3d8ef7_1px,transparent_1px)] [background-size:24px_24px]" />

            {/* Simulated Architecture Silhouette (Victoria Memorial facade) */}
            <div className="relative w-full h-full flex flex-col items-center justify-center pointer-events-none opacity-40">
              <div className="w-48 h-32 border-2 border-dashed border-[#3d8ef7]/40 rounded-t-xl flex flex-col items-center justify-center">
                <div className="text-[10px] font-mono text-[#3d8ef7] uppercase tracking-widest text-center">
                  Target: Victoria Memorial
                </div>
                <div className="text-[9px] font-mono text-[#9296a6] mt-1">
                  Active Region: North Portico Colonnade
                </div>
              </div>
            </div>

            {/* Crosshairs & Reticle Overlay */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <Crosshair className="w-12 h-12 text-[#3d8ef7]/60 stroke-[1.2]" />
              <div className="absolute w-24 h-24 border border-white/15 rounded-lg" />
            </div>

            {/* Real-time Electronic Level (Pitch & Roll HUD) */}
            <div className="absolute inset-x-8 top-1/2 -translate-y-1/2 pointer-events-none flex items-center justify-between">
              <div
                className="w-12 h-0.5 bg-[#2ecc71] shadow-[0_0_8px_#2ecc71] transition-transform duration-100"
                style={{ transform: `rotate(${levelRoll * 8}deg)` }}
              />
              <div className="text-[9px] font-mono text-[#2ecc71] bg-black/60 px-1.5 py-0.5 rounded border border-[#2ecc71]/40">
                PITCH: {levelPitch.toFixed(1)}° • ROLL: {levelRoll.toFixed(1)}°
              </div>
              <div
                className="w-12 h-0.5 bg-[#2ecc71] shadow-[0_0_8px_#2ecc71] transition-transform duration-100"
                style={{ transform: `rotate(${-levelRoll * 8}deg)` }}
              />
            </div>

            {/* Top Viewfinder Telemetry Banner */}
            <div className="absolute top-10 inset-x-3 flex items-center justify-between text-[10px] font-mono bg-black/70 backdrop-blur-md px-2.5 py-1.5 rounded-lg border border-white/10 z-20">
              <div className="flex items-center gap-2">
                <span className="text-[#3d8ef7] font-bold">{shutterSpeed}</span>
                <span className="text-[#9296a6]">|</span>
                <span className="text-[#3d8ef7]">{isoVal}</span>
                <span className="text-[#9296a6]">|</span>
                <span className="text-[#f0f1f6]">{wbTemp}</span>
              </div>
              <div className="flex items-center gap-2 text-right">
                <span className="text-[#2ecc71]">HDG: {simulatedHeading.toFixed(1)}°</span>
                <span className="text-[#9296a6]">|</span>
                <span className="text-[#f0f1f6]">{liveCaptureQuality.fps} FPS</span>
              </div>
            </div>

            {/* Dynamic Live Guidance Cue Prompt */}
            <div className="absolute top-22 inset-x-3 z-20">
              <div className="bg-[#12151f]/90 backdrop-blur-md border border-[#3d8ef7]/40 rounded-xl p-2.5 shadow-xl flex items-start gap-2.5">
                <div className="p-1 rounded bg-[#3d8ef7]/20 text-[#3d8ef7] shrink-0 mt-0.5">
                  <Info className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-bold text-[#f0f1f6] font-mono">
                    {currentPassInfo.title}
                  </div>
                  <div className="text-[10px] text-[#c5c8d6] mt-0.5 leading-relaxed">
                    {currentPassInfo.hint}
                  </div>
                  <div className="mt-1 flex items-center gap-3 text-[9px] font-mono text-[#9296a6]">
                    <span>Required Overlap: <strong className="text-[#3d8ef7]">{currentPassInfo.requiredOverlap}</strong></span>
                    <span>Rec. Lens: <strong className="text-[#2ecc71]">{currentPassInfo.recommendedLens}</strong></span>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Viewfinder HUD Controls: Lens Selector (0.5x, 1x, 3x) */}
            <div className="absolute bottom-20 inset-x-0 flex items-center justify-center gap-2 z-20">
              {(['0.5x', '1x', '3x'] as const).map((lens) => (
                <button
                  key={lens}
                  type="button"
                  onClick={() => setCameraLens(lens)}
                  className={`w-9 h-9 rounded-full font-mono text-[11px] font-bold flex items-center justify-center transition-all ${
                    cameraLens === lens
                      ? 'bg-[#3d8ef7] text-[#08090b] ring-2 ring-white/30 scale-110 shadow-lg'
                      : 'bg-black/60 backdrop-blur-md border border-white/20 text-[#f0f1f6] hover:bg-black/80'
                  }`}
                >
                  {lens}
                </button>
              ))}
            </div>

            {/* Bottom Left Quality Overlay Pill */}
            <div className="absolute bottom-4 left-3 z-20">
              <div className="bg-black/75 backdrop-blur-md border border-white/10 rounded-lg p-2 text-[9px] font-mono space-y-1">
                <div className="flex justify-between gap-2 text-[#9296a6]">
                  <span>Frames:</span>
                  <span className="text-[#f0f1f6] font-bold">{liveCaptureQuality.capturedFrames}</span>
                </div>
                <div className="flex justify-between gap-2 text-[#9296a6]">
                  <span>Coverage:</span>
                  <span className="text-[#2ecc71] font-bold">{liveCaptureQuality.coveragePercentage.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between gap-2 text-[#9296a6]">
                  <span>Storage:</span>
                  <span className="text-[#f0f1f6] font-bold">{mobileSensors.storageAvailableGb.toFixed(0)} GB left</span>
                </div>
              </div>
            </div>

            {/* Bottom Right Focus / Shutter Info */}
            <div className="absolute bottom-4 right-3 z-20">
              <div className="bg-black/75 backdrop-blur-md border border-white/10 rounded-lg p-2 text-[9px] font-mono space-y-1 text-right">
                <div className="text-[#9296a6]">FOCUS: <strong className="text-[#3d8ef7]">{focusDistanceM}</strong></div>
                <div className="text-[#2ecc71] flex items-center justify-end gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#2ecc71]" />
                  <span>PTP SYNCED</span>
                </div>
              </div>
            </div>
          </div>

          {/* ── Physical Shutter & Controls Dock (80px) ────────────────── */}
          <div className="h-20 bg-[#0d0e12] border-t border-[#1f222b] px-6 flex items-center justify-between shrink-0 z-30">
            {/* Gallery Thumbnail Preview */}
            <button
              type="button"
              onClick={() => setActiveWorkspace('loader')}
              title="View Ingested Sessions in Loader"
              className="w-11 h-11 rounded-lg border border-[#2b3040] bg-[#171922] flex flex-col items-center justify-center text-[9px] font-mono text-[#9296a6] hover:border-[#3d8ef7]"
            >
              <Layers className="w-4 h-4 text-[#3d8ef7] mb-0.5" />
              <span>{liveCaptureQuality.capturedFrames}</span>
            </button>

            {/* Main Shutter Trigger Button */}
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={triggerPhotoCapture}
                title="Trigger Single Precision Frame"
                className="w-15 h-15 rounded-full border-4 border-white/80 p-0.5 flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-xl"
              >
                <span className="w-full h-full rounded-full bg-white active:bg-[#e0e0e0] block" />
              </button>

              <button
                type="button"
                onClick={toggleRecording}
                title={isRecording ? 'Stop Sweep' : 'Continuous Spatial Sweep'}
                className={`w-11 h-11 rounded-full border border-white/20 flex items-center justify-center transition-all ${
                  isRecording
                    ? 'bg-[#e74c3c] text-white animate-pulse'
                    : 'bg-[#171922] text-[#f0f1f6] hover:bg-[#222530]'
                }`}
              >
                {isRecording ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
              </button>
            </div>

            {/* Settings & Manual Sliders Trigger */}
            <button
              type="button"
              onClick={toggleQualityDetails}
              title="Camera Parameters & Calibration"
              className="w-11 h-11 rounded-lg border border-[#2b3040] bg-[#171922] flex items-center justify-center text-[#9296a6] hover:text-[#f0f1f6]"
            >
              <Sliders className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
