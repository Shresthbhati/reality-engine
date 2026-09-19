'use client';

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
  Maximize2,
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
} from 'lucide-react';
import { useREStore } from '@/store/re-store';

export default function CaptureWorkspace() {
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
    setActiveWorkspace,
  } = useREStore();

  const [simulatedAngle, setSimulatedAngle] = useState(42);
  const [levelPitch, setLevelPitch] = useState(0.2);
  const [levelRoll, setLevelRoll] = useState(-0.1);

  // Periodic simulated sensor fluctuation for realistic instrument feel
  useEffect(() => {
    const interval = setInterval(() => {
      setSimulatedAngle((prev) => (prev + 0.4) % 360);
      setLevelPitch(Math.sin(Date.now() / 1500) * 0.8);
      setLevelRoll(Math.cos(Date.now() / 2000) * 0.5);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col w-full h-full bg-[#08090b] text-[#f0f1f6] overflow-hidden select-none">
      {/* ── Top Instrument Chrome ────────────────────────────────────────── */}
      <header className="h-9 px-3 flex items-center justify-between border-b border-[#1f222b] bg-[#0f1014] shrink-0 text-xs">
        <div className="flex items-center gap-2.5">
          <span className="flex items-center gap-1.5 font-mono text-[11px] font-semibold text-[#3d8ef7] tracking-wider">
            <Camera className="w-3.5 h-3.5" />
            REALITY CAPTURE
          </span>
          <span className="text-[#54596b]">/</span>
          <span className="text-[#9296a6] font-mono text-[10px] uppercase">
            APPLICATION 1 — MOBILE FIELD INSTRUMENT
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* View mode toggle: Simulated Device vs Direct Viewport */}
          <button
            type="button"
            onClick={toggleSimulateMobileDevice}
            className="flex items-center gap-1.5 px-2 py-0.5 rounded border border-[#1f222b] hover:border-[#3d8ef7]/40 bg-[#15171d] text-[11px] text-[#9296a6] hover:text-[#f0f1f6] transition-colors"
          >
            {simulateMobileDevice ? (
              <>
                <Maximize2 className="w-3 h-3 text-[#3d8ef7]" />
                <span>Responsive View</span>
              </>
            ) : (
              <>
                <Smartphone className="w-3 h-3 text-[#3d8ef7]" />
                <span>Phone Frame</span>
              </>
            )}
          </button>

          {/* Quick Stage Navigators for testing workflows */}
          <div className="flex items-center bg-[#15171d] rounded border border-[#1f222b] p-0.5 text-[10px] font-mono">
            {(['home', 'project', 'prep', 'camera', 'review'] as const).map((scr) => (
              <button
                key={scr}
                type="button"
                onClick={() => setCaptureScreen(scr)}
                className={`px-2 py-0.5 rounded transition-colors uppercase ${
                  captureScreen === scr
                    ? 'bg-[#3d8ef7] text-[#08090b] font-semibold'
                    : 'text-[#9296a6] hover:text-[#f0f1f6]'
                }`}
              >
                {scr}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* ── Main Canvas Viewport ────────────────────────────────────────── */}
      <div className="flex-1 flex items-center justify-center p-3 overflow-hidden bg-[#08090b]">
        {simulateMobileDevice ? (
          /* Phone Frame Container */
          <div
            className="relative flex flex-col overflow-hidden bg-black shadow-2xl border-[4px] border-[#2d323f] rounded-[44px]"
            style={{
              width: '390px',
              height: '810px',
              boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.9), 0 0 0 1px rgba(255, 255, 255, 0.05)',
            }}
          >
            {/* Dynamic Island / Speaker Notch */}
            <div className="absolute top-2 left-1/2 -translate-x-1/2 z-50 w-28 h-6 bg-black rounded-full flex items-center justify-between px-3 border border-[#2d323f]/50">
              <div className="w-2.5 h-2.5 rounded-full bg-[#15171d] border border-blue-500/30 flex items-center justify-center">
                <div className="w-1 h-1 rounded-full bg-blue-500" />
              </div>
              <div className="w-2.5 h-2.5 rounded-full bg-[#15171d]" />
            </div>

            {/* Screen Content */}
            <div className="flex-1 flex flex-col pt-7 pb-4 px-0 overflow-hidden relative">
              <MobileScreenRouter />
            </div>

            {/* iOS Home Indicator Bar */}
            <div className="absolute bottom-1.5 left-1/2 -translate-x-1/2 w-32 h-1 bg-white/30 rounded-full z-50 pointer-events-none" />
          </div>
        ) : (
          /* Responsive Direct Viewport Container */
          <div className="w-full h-full max-w-4xl bg-black rounded-xl border border-[#1f222b] overflow-hidden flex flex-col relative shadow-xl">
            <MobileScreenRouter />
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Screen Router for Mobile Stages ─────────────────────────────────────────

function MobileScreenRouter() {
  const { captureScreen } = useREStore();

  switch (captureScreen) {
    case 'home':
      return <CaptureHomeScreen />;
    case 'project':
      return <CaptureProjectScreen />;
    case 'prep':
      return <CapturePrepScreen />;
    case 'camera':
      return <CaptureCameraScreen />;
    case 'review':
      return <CaptureReviewScreen />;
    default:
      return <CaptureCameraScreen />;
  }
}

// ─── Stage 1: Capture Home Screen ────────────────────────────────────────────

function CaptureHomeScreen() {
  const { setCaptureScreen } = useREStore();

  return (
    <div className="flex-1 flex flex-col p-5 bg-[#09090b] text-[#ededf2] overflow-y-auto">
      {/* Home Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[#f0f1f6]">Reality Capture</h1>
          <p className="text-[11px] text-[#9296a6]">Precision Spatial Acquisition</p>
        </div>
        <div className="w-8 h-8 rounded-full bg-[#15171d] border border-[#1f222b] flex items-center justify-center text-[#3d8ef7]">
          <Radio className="w-4 h-4 text-[#2ecc71] animate-pulse" />
        </div>
      </div>

      {/* Primary Action Button */}
      <button
        type="button"
        onClick={() => setCaptureScreen('prep')}
        className="w-full py-4 mb-6 rounded-2xl bg-[#3d8ef7] hover:bg-[#5ca2f9] text-[#08090b] font-semibold flex items-center justify-center gap-2.5 shadow-lg shadow-blue-500/20 active:scale-[0.99] transition-all"
      >
        <Camera className="w-5 h-5" />
        <span className="text-sm tracking-wide">NEW CAPTURE SESSION</span>
      </button>

      {/* System Status Summary */}
      <div className="bg-[#121215] border border-[#1f222b] rounded-2xl p-4 mb-6">
        <div className="flex items-center justify-between pb-3 border-b border-[#1f222b] text-xs">
          <span className="text-[#9296a6]">Sensor Subsystem</span>
          <span className="flex items-center gap-1.5 text-[#2ecc71] font-medium text-[11px]">
            <CheckCircle2 className="w-3.5 h-3.5" /> Nominal (RTK Fix)
          </span>
        </div>
        <div className="grid grid-cols-2 gap-3 pt-3 text-[11px] font-mono">
          <div>
            <span className="text-[#54596b] block">STORAGE</span>
            <span className="text-[#f0f1f6] font-semibold">194.5 GB Free</span>
          </div>
          <div>
            <span className="text-[#54596b] block">LOADER SYNC</span>
            <span className="text-[#2ecc71] font-semibold">Live (LAN/5G)</span>
          </div>
        </div>
      </div>

      {/* Recent Projects List */}
      <div className="flex items-center justify-between mb-3 text-xs">
        <span className="font-semibold text-[#9296a6] uppercase tracking-wider text-[10px]">Active Projects</span>
        <button
          type="button"
          onClick={() => setCaptureScreen('project')}
          className="text-[#3d8ef7] text-[11px] hover:underline"
        >
          View Details
        </button>
      </div>

      <div className="space-y-2.5 mb-6">
        <div
          onClick={() => setCaptureScreen('project')}
          className="p-3.5 rounded-xl bg-[#15171d] border border-[#1f222b] hover:border-[#3d8ef7]/40 cursor-pointer transition-colors"
        >
          <div className="flex items-start justify-between mb-1">
            <span className="font-semibold text-sm text-[#f0f1f6]">Victoria Memorial Complex</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#3d8ef7]/15 text-[#3d8ef7] border border-[#3d8ef7]/30">
              74% COVERAGE
            </span>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-[#9296a6]">
            <MapPin className="w-3 h-3 text-[#e74c3c]" />
            <span>Kolkata, India</span>
            <span>•</span>
            <span>3 Sessions</span>
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-[#15171d] border border-[#1f222b] opacity-75">
          <div className="flex items-start justify-between mb-1">
            <span className="font-semibold text-sm text-[#f0f1f6]">Amber Fort Outer Ramparts</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30">
              STAGED
            </span>
          </div>
          <div className="flex items-center gap-2 text-[11px] text-[#9296a6]">
            <MapPin className="w-3 h-3 text-[#e74c3c]" />
            <span>Jaipur, India</span>
            <span>•</span>
            <span>Drone Complete</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Stage 2: Project Screen ─────────────────────────────────────────────────

function CaptureProjectScreen() {
  const { setCaptureScreen } = useREStore();

  return (
    <div className="flex-1 flex flex-col p-5 bg-[#09090b] text-[#ededf2] overflow-y-auto">
      <button
        type="button"
        onClick={() => setCaptureScreen('home')}
        className="flex items-center gap-1.5 text-xs text-[#9296a6] hover:text-[#f0f1f6] mb-4 transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" /> Back to Home
      </button>

      <div className="mb-5">
        <h1 className="text-xl font-bold text-[#f0f1f6] mb-1">Victoria Memorial Monument</h1>
        <div className="flex items-center gap-2 text-xs text-[#9296a6]">
          <MapPin className="w-3.5 h-3.5 text-[#e74c3c]" />
          <span>22.5448° N, 88.3426° E • WGS84</span>
        </div>
      </div>

      {/* Coverage Status Card */}
      <div className="bg-[#121215] border border-[#1f222b] rounded-2xl p-4 mb-5">
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs text-[#9296a6]">Reconstruction Coverage</span>
          <span className="font-mono text-sm font-bold text-[#2ecc71]">74.2%</span>
        </div>
        <div className="w-full h-2 bg-[#1f222b] rounded-full overflow-hidden mb-3">
          <div className="h-full bg-gradient-to-r from-blue-500 to-[#2ecc71] w-[74%]" />
        </div>
        <div className="grid grid-cols-3 gap-2 text-center text-[10px] font-mono">
          <div className="p-2 rounded bg-[#15171d] border border-[#1f222b]">
            <span className="text-[#54596b] block">PHOTOS</span>
            <span className="text-[#f0f1f6] font-semibold">7,661</span>
          </div>
          <div className="p-2 rounded bg-[#15171d] border border-[#1f222b]">
            <span className="text-[#54596b] block">LIDAR</span>
            <span className="text-[#f0f1f6] font-semibold">18.5M pts</span>
          </div>
          <div className="p-2 rounded bg-[#15171d] border border-[#1f222b]">
            <span className="text-[#54596b] block">FACADES</span>
            <span className="text-[#f0f1f6] font-semibold">3 / 4 Done</span>
          </div>
        </div>
      </div>

      {/* Past Sessions List */}
      <span className="font-semibold text-[#9296a6] uppercase tracking-wider text-[10px] mb-2.5 block">
        Project Sessions
      </span>
      <div className="space-y-2 mb-6">
        <div className="p-3 rounded-xl bg-[#15171d] border border-[#1f222b] text-xs">
          <div className="flex justify-between items-center mb-1">
            <span className="font-mono font-medium text-[#f0f1f6]">Drone_Aerial_Orbit_001</span>
            <span className="text-[10px] text-[#2ecc71] font-medium">Reconstructed</span>
          </div>
          <span className="text-[11px] text-[#54596b]">4,821 aerial frames • RTK GNSS 10Hz</span>
        </div>

        <div className="p-3 rounded-xl bg-[#15171d] border border-[#1f222b] text-xs">
          <div className="flex justify-between items-center mb-1">
            <span className="font-mono font-medium text-[#f0f1f6]">iPhone_Pro_Walkthrough_002</span>
            <span className="text-[10px] text-[#2ecc71] font-medium">Reconstructed</span>
          </div>
          <span className="text-[11px] text-[#54596b]">2,840 handheld ground frames • IMU 200Hz</span>
        </div>
      </div>

      <button
        type="button"
        onClick={() => setCaptureScreen('prep')}
        className="w-full py-3.5 rounded-xl bg-[#3d8ef7] hover:bg-[#5ca2f9] text-[#08090b] font-semibold text-xs tracking-wider uppercase transition-colors"
      >
        Prepare Next Capture Pass
      </button>
    </div>
  );
}

// ─── Stage 3: Capture Prep Screen ────────────────────────────────────────────

function CapturePrepScreen() {
  const { setCaptureScreen, mobileSensors } = useREStore();

  const checks = [
    { label: 'Camera Hardware', detail: '4K @ 60fps ProRes / RAW', ready: mobileSensors.cameraReady },
    { label: 'GNSS Satellite Fix', detail: `RTK Fixed (±${mobileSensors.gnssAccuracyM}m, ${mobileSensors.gnssHz}Hz)`, ready: mobileSensors.gnssFix },
    { label: 'IMU Spatial Calibration', detail: `Gravitational Allan Variance converged (${mobileSensors.imuHz}Hz)`, ready: mobileSensors.imuCalibrated },
    { label: 'PTP Time Synchronization', detail: `Drift < ${mobileSensors.ptpDriftMs} ms with Master Clock`, ready: mobileSensors.ptpTimeSynced },
    { label: 'Flash Storage Allocation', detail: `${mobileSensors.storageAvailableGb} GB available on NVMe`, ready: true },
    { label: 'Lens Distortion Model', detail: 'Brown-Conrady 8-param profile verified', ready: mobileSensors.lensCalibrated },
  ];

  return (
    <div className="flex-1 flex flex-col p-5 bg-[#09090b] text-[#ededf2] overflow-y-auto">
      <button
        type="button"
        onClick={() => setCaptureScreen('home')}
        className="flex items-center gap-1.5 text-xs text-[#9296a6] hover:text-[#f0f1f6] mb-4 transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" /> Back
      </button>

      <h1 className="text-xl font-bold text-[#f0f1f6] mb-1">Pre-Capture Calibration</h1>
      <p className="text-xs text-[#9296a6] mb-5">Validating hardware sensors before acquiring reality evidence.</p>

      {/* Sensor Readiness Matrix */}
      <div className="space-y-2.5 mb-6">
        {checks.map((chk, i) => (
          <div
            key={i}
            className="flex items-center justify-between p-3 rounded-xl bg-[#121215] border border-[#1f222b]"
          >
            <div>
              <span className="font-semibold text-xs text-[#f0f1f6] block">{chk.label}</span>
              <span className="text-[11px] font-mono text-[#9296a6]">{chk.detail}</span>
            </div>
            <div className="w-5 h-5 rounded-full bg-[#2ecc71]/15 border border-[#2ecc71]/40 flex items-center justify-center text-[#2ecc71]">
              <Check className="w-3 h-3" />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-auto">
        <button
          type="button"
          onClick={() => setCaptureScreen('camera')}
          className="w-full py-4 rounded-2xl bg-[#3d8ef7] hover:bg-[#5ca2f9] text-[#08090b] font-bold text-sm tracking-wider uppercase shadow-lg shadow-blue-500/25 active:scale-[0.99] transition-all flex items-center justify-center gap-2"
        >
          <Camera className="w-4 h-4" />
          <span>INITIALIZE CAMERA VIEWPORT</span>
        </button>
      </div>
    </div>
  );
}

// ─── Stage 4: Full-Screen Camera Viewport ────────────────────────────────────

function CaptureCameraScreen() {
  const {
    setCaptureScreen,
    cameraLens,
    setCameraLens,
    isRecording,
    toggleRecording,
    triggerPhotoCapture,
    liveCaptureQuality,
    mobileSensors,
    showCoverageHud,
    toggleCoverageHud,
    showQualityDetails,
    toggleQualityDetails,
  } = useREStore();

  const [shutterFlash, setShutterFlash] = useState(false);

  const handleCapture = () => {
    triggerPhotoCapture();
    setShutterFlash(true);
    setTimeout(() => setShutterFlash(false), 120);
  };

  return (
    <div className="flex-1 flex flex-col relative bg-[#050507] overflow-hidden">
      {/* ── Shutter Flash Feedback */}
      {shutterFlash && <div className="absolute inset-0 bg-white/40 z-40 pointer-events-none" />}

      {/* ── Simulated Live Video Feed & Scene ───────────────────────────── */}
      <div className="absolute inset-0 overflow-hidden">
        {/* Background gradient simulating outdoor monument scene */}
        <div className="w-full h-full bg-gradient-to-b from-[#111827] via-[#0c121e] to-[#05070a] flex items-center justify-center relative">
          {/* Simulated 3D structure wireframe contour in viewfinder */}
          <svg className="w-full h-full opacity-35" viewBox="0 0 400 600">
            {/* Monument Dome Silhouette */}
            <path
              d="M 120 380 L 120 280 Q 200 160 280 280 L 280 380 Z"
              fill="none"
              stroke="#3d8ef7"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
            {/* Colonnade Pillars */}
            <line x1="140" y1="380" x2="140" y2="460" stroke="#3d8ef7" strokeWidth="1.5" />
            <line x1="170" y1="380" x2="170" y2="460" stroke="#3d8ef7" strokeWidth="1.5" />
            <line x1="200" y1="380" x2="200" y2="460" stroke="#3d8ef7" strokeWidth="1.5" />
            <line x1="230" y1="380" x2="230" y2="460" stroke="#3d8ef7" strokeWidth="1.5" />
            <line x1="260" y1="380" x2="260" y2="460" stroke="#3d8ef7" strokeWidth="1.5" />
            <line x1="80" y1="460" x2="320" y2="460" stroke="#3d8ef7" strokeWidth="2" />
          </svg>

          {/* Viewfinder Reticle (Rule of thirds grid) */}
          <div className="absolute inset-x-8 inset-y-16 pointer-events-none border border-white/10 grid grid-cols-3 grid-rows-3">
            <div className="border-r border-b border-white/10" />
            <div className="border-r border-b border-white/10" />
            <div className="border-b border-white/10" />
            <div className="border-r border-b border-white/10" />
            <div className="border-r border-b border-white/10 flex items-center justify-center">
              {/* Center focus indicator */}
              <div className="w-10 h-10 border border-[#2ecc71]/80 rounded-sm relative flex items-center justify-center">
                <div className="w-1.5 h-1.5 bg-[#2ecc71] rounded-full" />
              </div>
            </div>
            <div className="border-b border-white/10" />
            <div className="border-r border-white/10" />
            <div className="border-r border-white/10" />
            <div />
          </div>

          {/* Level Horizon Gyro Indicator */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 flex items-center justify-between pointer-events-none opacity-60">
            <div className="w-10 h-[1.5px] bg-[#2ecc71]" />
            <span className="text-[9px] font-mono text-[#2ecc71]">0.0°</span>
            <div className="w-10 h-[1.5px] bg-[#2ecc71]" />
          </div>
        </div>
      </div>

      {/* ── Top HUD: Sensor Status & Timer ──────────────────────────────── */}
      <div className="relative z-30 pt-2 px-3 flex items-center justify-between text-[11px] font-mono bg-gradient-to-b from-black/80 to-transparent">
        <button
          type="button"
          onClick={() => setCaptureScreen('review')}
          className="px-2 py-1 rounded bg-black/60 backdrop-blur border border-white/15 text-[#f0f1f6] hover:bg-white/10 transition-colors"
        >
          Finish Session
        </button>

        <div className="flex items-center gap-2">
          {/* RTK Fix */}
          <span className="px-1.5 py-0.5 rounded bg-black/60 backdrop-blur border border-[#2ecc71]/40 text-[#2ecc71] text-[10px] flex items-center gap-1">
            <Radio className="w-2.5 h-2.5 animate-pulse" /> RTK 10Hz
          </span>
          {/* Frame count */}
          <span className="px-1.5 py-0.5 rounded bg-black/60 backdrop-blur border border-white/15 text-[#f0f1f6] text-[10px] num-tabular">
            {liveCaptureQuality.capturedFrames} FRAMES
          </span>
        </div>
      </div>

      {/* ── Progressive Quality Warning Banner ──────────────────────────── */}
      <div className="relative z-30 px-3 mt-2">
        <div
          onClick={toggleQualityDetails}
          className="p-2 rounded-xl bg-black/70 backdrop-blur border border-white/15 cursor-pointer flex flex-col gap-1 transition-all"
        >
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#2ecc71] animate-pulse" />
              <span className="font-semibold text-[11px] text-[#f0f1f6]">Capture Quality: Nominal</span>
            </div>
            <span className="text-[10px] font-mono text-[#3d8ef7]">
              {showQualityDetails ? 'Close Details ▲' : 'Inspect ▼'}
            </span>
          </div>

          {/* Progressive Disclosure: Tap to expand */}
          {showQualityDetails && (
            <div className="mt-2 pt-2 border-t border-white/10 grid grid-cols-3 gap-2 text-[10px] font-mono">
              <div>
                <span className="text-[#9296a6] block">BLUR RESIDUAL</span>
                <span className="text-[#2ecc71]">0.03 (Very Low)</span>
              </div>
              <div>
                <span className="text-[#9296a6] block">LIGHT RANGE</span>
                <span className="text-[#f0f1f6]">13.2 EV</span>
              </div>
              <div>
                <span className="text-[#9296a6] block">IMU STABILITY</span>
                <span className="text-[#2ecc71]">200 Hz Valid</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Lightweight Coverage HUD Radar ──────────────────────────────── */}
      {showCoverageHud && (
        <div className="absolute top-24 right-3 z-30 w-28 p-2 rounded-xl bg-black/75 backdrop-blur border border-white/15 text-[10px] font-mono flex flex-col items-center">
          <div className="flex justify-between w-full mb-1 text-[#9296a6]">
            <span>COVERAGE</span>
            <span className="text-[#2ecc71]">{liveCaptureQuality.coveragePercentage}%</span>
          </div>
          {/* Mini 2D Radar Frustum View */}
          <div className="w-16 h-16 rounded-full border border-white/20 relative flex items-center justify-center bg-black/50">
            {/* Monument Target center */}
            <div className="w-3 h-3 bg-[#3d8ef7] rounded-sm" />
            {/* Camera Frustum Cone */}
            <div
              className="absolute w-12 h-12 pointer-events-none"
              style={{ transform: 'rotate(45deg)' }}
            >
              <div className="w-0 h-0 border-l-[12px] border-l-transparent border-r-[12px] border-r-transparent border-b-[24px] border-b-[#2ecc71]/40 mx-auto" />
            </div>
          </div>
          <span className="text-[9px] text-[#9296a6] mt-1">North Elevation</span>
        </div>
      )}

      {/* ── Bottom Controls: Shutter & Lens Selector ─────────────────────── */}
      <div className="mt-auto relative z-30 pb-5 pt-3 bg-gradient-to-t from-black/90 via-black/60 to-transparent flex flex-col items-center gap-3">
        {/* Focal Length Lens Switcher */}
        <div className="flex items-center gap-3 bg-black/60 backdrop-blur border border-white/15 px-3 py-1 rounded-full text-xs font-mono">
          {(['0.5x', '1x', '3x'] as const).map((lens) => (
            <button
              key={lens}
              type="button"
              onClick={() => setCameraLens(lens)}
              className={`px-2 py-0.5 rounded-full transition-colors ${
                cameraLens === lens
                  ? 'bg-[#3d8ef7] text-[#08090b] font-bold'
                  : 'text-[#9296a6] hover:text-[#f0f1f6]'
              }`}
            >
              {lens}
            </button>
          ))}
        </div>

        {/* Shutter / Capture Trigger */}
        <div className="w-full flex items-center justify-around px-8">
          {/* Coverage Overlay Toggle */}
          <button
            type="button"
            onClick={toggleCoverageHud}
            className={`w-10 h-10 rounded-full border flex items-center justify-center transition-colors ${
              showCoverageHud
                ? 'border-[#3d8ef7] bg-[#3d8ef7]/20 text-[#3d8ef7]'
                : 'border-white/20 bg-black/50 text-[#9296a6]'
            }`}
          >
            <Compass className="w-5 h-5" />
          </button>

          {/* Primary Shutter Button */}
          <button
            type="button"
            onClick={handleCapture}
            className="w-18 h-18 rounded-full border-[4px] border-white p-1 flex items-center justify-center active:scale-95 transition-transform shadow-2xl"
          >
            <div className="w-full h-full rounded-full bg-white active:bg-[#3d8ef7] transition-colors" />
          </button>

          {/* Continuous Record Toggle */}
          <button
            type="button"
            onClick={toggleRecording}
            className={`w-10 h-10 rounded-full border flex items-center justify-center transition-colors ${
              isRecording
                ? 'border-red-500 bg-red-500/20 text-red-500 animate-pulse'
                : 'border-white/20 bg-black/50 text-[#9296a6]'
            }`}
          >
            {isRecording ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Stage 5: Session Review & Ingestion Trigger ─────────────────────────────

function CaptureReviewScreen() {
  const { setCaptureScreen, setActiveWorkspace, liveCaptureQuality } = useREStore();

  return (
    <div className="flex-1 flex flex-col p-5 bg-[#09090b] text-[#ededf2] overflow-y-auto">
      <h1 className="text-xl font-bold text-[#f0f1f6] mb-1">Session Complete</h1>
      <p className="text-xs text-[#9296a6] mb-5">Review capture telemetry before transferring to Desktop Loader.</p>

      {/* Session Metrics Card */}
      <div className="bg-[#121215] border border-[#1f222b] rounded-2xl p-4 mb-5">
        <div className="flex justify-between items-center pb-3 border-b border-[#1f222b]">
          <span className="text-xs text-[#9296a6]">Session Identifier</span>
          <span className="text-xs font-mono text-[#3d8ef7]">SESS-2026-09-16-004</span>
        </div>

        <div className="grid grid-cols-2 gap-3 py-3 border-b border-[#1f222b] text-xs font-mono">
          <div>
            <span className="text-[#54596b] block">TOTAL FRAMES</span>
            <span className="text-sm font-bold text-[#f0f1f6] num-tabular">
              {liveCaptureQuality.capturedFrames}
            </span>
          </div>
          <div>
            <span className="text-[#54596b] block">DURATION</span>
            <span className="text-sm font-bold text-[#f0f1f6] num-tabular">02m 48s</span>
          </div>
          <div>
            <span className="text-[#54596b] block">SURFACE COVERAGE</span>
            <span className="text-sm font-bold text-[#2ecc71] num-tabular">
              {liveCaptureQuality.coveragePercentage}%
            </span>
          </div>
          <div>
            <span className="text-[#54596b] block">GNSS INTEGRITY</span>
            <span className="text-sm font-bold text-[#2ecc71] num-tabular">100% Fix</span>
          </div>
        </div>

        <div className="pt-3 text-[11px] text-[#9296a6]">
          <span className="text-[#54596b] block mb-1">PROVENANCE READY</span>
          <span>Encoded with camera intrinsics, IMU gravity alignment, and ISO 8601 GPS timestamps.</span>
        </div>
      </div>

      <div className="mt-auto space-y-2.5">
        <button
          type="button"
          onClick={() => {
            // Hand over session to Desktop Loader application
            setActiveWorkspace('loader');
          }}
          className="w-full py-4 rounded-2xl bg-[#3d8ef7] hover:bg-[#5ca2f9] text-[#08090b] font-bold text-sm tracking-wider uppercase flex items-center justify-center gap-2 shadow-lg shadow-blue-500/25 active:scale-[0.99] transition-all"
        >
          <UploadCloud className="w-4 h-4" />
          <span>TRANSFER TO LOADER WORKSPACE</span>
        </button>

        <button
          type="button"
          onClick={() => setCaptureScreen('home')}
          className="w-full py-2.5 rounded-xl border border-[#1f222b] hover:bg-white/5 text-xs text-[#9296a6] transition-colors"
        >
          Return to Projects
        </button>
      </div>
    </div>
  );
}
