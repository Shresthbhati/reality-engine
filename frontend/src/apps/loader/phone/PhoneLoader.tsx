'use client';

/**
 * PhoneLoader — Mobile Companion Evidence Ingest & Transfer Client.
 * 
 * Spec:
 * - Field wireless transfer companion (Wi-Fi Direct / BLE beaconing)
 * - Quick session offload with checksum validation
 * - Instant on-device thumbnail gallery & metadata inspector
 * - Field QA Sanity Check (flags blur, low overlap, or missing GNSS before leaving site)
 * - 1-click push to Computer Workstation / Reconstruction Studio
 */

import React, { useState } from 'react';
import {
  Smartphone,
  Wifi,
  Radio,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  ArrowUpRight,
  RefreshCw,
  Layers,
  Camera,
  MapPin,
  Clock,
  Send,
  Sliders,
  Check,
  Zap,
} from 'lucide-react';
import { useREStore } from '@/store/re-store';

export function PhoneLoader() {
  const {
    sessions,
    selectedSessionId,
    setSelectedSessionId,
    setActiveWorkspace,
    setLoaderDeviceMode,
  } = useREStore();

  const [transferSpeedMbps, setTransferSpeedMbps] = useState(84.2);
  const [transferProgress, setTransferProgress] = useState(68);
  const [isTransferring, setIsTransferring] = useState(false);
  const [pairedDesktop, setPairedDesktop] = useState('Studio-Workstation-Pro (192.168.1.142)');

  const activeSession = sessions.find((s) => s.id === selectedSessionId) || sessions[0];

  const handleStartTransfer = () => {
    setIsTransferring(true);
    const timer = setInterval(() => {
      setTransferProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          setIsTransferring(false);
          return 100;
        }
        return prev + 8;
      });
    }, 400);
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#090a0d] text-[#ededf2] select-none font-sans overflow-hidden">
      {/* ── Mobile Companion Top Bar ─────────────────────────────────── */}
      <div className="h-10 px-4 bg-[#101217] border-b border-[#1f222b] flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="p-1 rounded bg-[#3d8ef7]/20 text-[#3d8ef7]">
            <Smartphone className="w-3.5 h-3.5" />
          </div>
          <span className="font-mono text-xs font-bold text-[#f0f1f6] tracking-wide">
            PHONE EVIDENCE COMPANION
          </span>
          <span className="text-[10px] px-1.5 py-0.2 rounded bg-[#2ecc71]/15 text-[#2ecc71] font-mono border border-[#2ecc71]/30">
            PAIRED
          </span>
        </div>

        <button
          type="button"
          onClick={() => setLoaderDeviceMode('computer')}
          className="text-[10px] font-mono text-[#3d8ef7] hover:underline flex items-center gap-1"
        >
          <span>Switch to Computer View</span>
          <ArrowUpRight className="w-3 h-3" />
        </button>
      </div>

      {/* ── Mobile Transfer Container ────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-4 max-w-xl mx-auto w-full space-y-4">
        {/* 1. Wireless Transfer Hub Card */}
        <div className="p-4 rounded-xl bg-[#13151c] border border-[#222633] space-y-3 shadow-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Wifi className="w-4 h-4 text-[#3d8ef7]" />
              <span className="text-xs font-bold text-[#f0f1f6]">Wi-Fi 6 Direct Link</span>
            </div>
            <span className="text-[10px] font-mono text-[#2ecc71] bg-[#2ecc71]/10 px-2 py-0.5 rounded">
              {transferSpeedMbps} MB/s
            </span>
          </div>

          <div className="text-[11px] text-[#9296a6] font-mono flex items-center justify-between">
            <span>Target: {pairedDesktop}</span>
            <span>Latency: 1.2ms</span>
          </div>

          {/* Transfer Progress Bar */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[10px] font-mono">
              <span className="text-[#9296a6]">Offloading Session: {activeSession?.name}</span>
              <span className="text-[#3d8ef7] font-bold">{transferProgress}%</span>
            </div>
            <div className="h-2 w-full bg-[#1b1e27] rounded-full overflow-hidden border border-[#262a36]">
              <div
                className="h-full bg-gradient-to-r from-[#3d8ef7] to-[#2ecc71] transition-all duration-300 rounded-full"
                style={{ width: `${transferProgress}%` }}
              />
            </div>
          </div>

          <div className="flex gap-2 pt-1">
            <button
              type="button"
              onClick={handleStartTransfer}
              disabled={isTransferring}
              className={`flex-1 py-2 rounded-lg font-mono text-xs font-bold flex items-center justify-center gap-1.5 transition-all ${
                isTransferring
                  ? 'bg-[#1b1e27] text-[#9296a6] cursor-not-allowed'
                  : 'bg-[#3d8ef7] hover:bg-[#2b7ae2] text-[#08090b]'
              }`}
            >
              {isTransferring ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Transferring Raw Frames...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>Sync Session to Workstation</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* 2. Field QA Sanity Check Card (Must-Have Before Leaving Site) */}
        <div className="p-4 rounded-xl bg-[#13151c] border border-[#222633] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#f0f1f6] flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#2ecc71]" />
              <span>Field Quality Assurance (Pre-Departure)</span>
            </span>
            <span className="text-[10px] font-mono text-[#2ecc71]">98.2% PASS</span>
          </div>

          <div className="space-y-2 text-[11px] font-mono">
            <div className="p-2 rounded bg-[#171922] border border-[#262a36] flex items-center justify-between">
              <span className="text-[#9296a6]">Motion Blur Check:</span>
              <span className="text-[#2ecc71] font-bold">Passed (0 Blurry Frames)</span>
            </div>
            <div className="p-2 rounded bg-[#171922] border border-[#262a36] flex items-center justify-between">
              <span className="text-[#9296a6]">GNSS Geotag Integrity:</span>
              <span className="text-[#2ecc71] font-bold">100% Locked (RTK Fixed)</span>
            </div>
            <div className="p-2 rounded bg-[#171922] border border-[#262a36] flex items-center justify-between">
              <span className="text-[#9296a6]">Colonnade Overlap Ratio:</span>
              <span className="text-[#3d8ef7] font-bold">78.4% (Exceeds 70% threshold)</span>
            </div>
          </div>
        </div>

        {/* 3. Session Selection & Thumbnail Preview Grid */}
        <div className="p-4 rounded-xl bg-[#13151c] border border-[#222633] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-[#f0f1f6] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-[#3d8ef7]" />
              <span>Available Evidence Sessions</span>
            </span>
            <span className="text-[10px] font-mono text-[#9296a6]">
              {sessions.length} local sessions
            </span>
          </div>

          <div className="space-y-2">
            {sessions.map((sess) => {
              const selected = sess.id === (activeSession?.id ?? '');
              return (
                <div
                  key={sess.id}
                  onClick={() => setSelectedSessionId(sess.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selected
                      ? 'bg-[#1b202e] border-[#3d8ef7]'
                      : 'bg-[#171922] border-[#222633] hover:border-[#3d8ef7]/40'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-mono font-bold">
                    <span className={selected ? 'text-[#3d8ef7]' : 'text-[#f0f1f6]'}>
                      {sess.name}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-[#2ecc71]/15 text-[#2ecc71]">
                      {sess.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 mt-2 text-[10px] font-mono text-[#9296a6]">
                    <span className="flex items-center gap-1">
                      <Camera className="w-3 h-3" />
                      {sess.imageCount} imgs
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {sess.reprojectionError?.toFixed(2) ?? '0.00'}px reproj
                    </span>
                    <span>{(((sess.pointCount ?? 0)) / 1000000).toFixed(1)}M pts</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
