'use client';

import React, { useState } from 'react';
import { useREStore } from '@/store/re-store';
import { TimelinePrimitive, TimelineStage } from '@/components/ui/timeline-primitive';
import {
  Clock,
  Cpu,
  Camera,
  Layers,
  ChevronDown,
  ChevronUp,
  Maximize2,
  Minimize2,
  AlertTriangle,
  CheckCircle2,
  Play,
  Pause,
} from 'lucide-react';

export default function ContextualBottomDrawer() {
  const bottomDrawerOpen = useREStore((s) => s.bottomDrawerOpen);
  const toggleBottomDrawer = useREStore((s) => s.toggleBottomDrawer);
  const addNotification = useREStore((s) => s.addNotification);

  const [activeStageId, setActiveStageId] = useState('stage-reconstruction');
  const [isPlaying, setIsPlaying] = useState(false);

  const stages: TimelineStage[] = [
    { id: 'stage-evidence', label: 'Evidence Ingest', status: 'COMPLETED' },
    { id: 'stage-session', label: 'Session Alignment', status: 'COMPLETED' },
    { id: 'stage-reconstruction', label: 'Reconstruction', status: 'IN_PROGRESS' },
    { id: 'stage-quality', label: 'Quality Check', status: 'PENDING' },
    { id: 'stage-world', label: 'World Update', status: 'PENDING' },
    { id: 'stage-query', label: 'Query / Review', status: 'PENDING' },
  ];

  const processingJobs = [
    { name: 'Lidar Point Cloud Processing', pct: 87, status: 'Active (87%)', color: 'bg-[#00e5ff]' },
    { name: 'Semantic Segmentation', pct: 100, status: 'Complete', color: 'bg-[#2ecc71]' },
    { name: 'Mesh Refinement', pct: 0, status: 'Pending queue', color: 'bg-[#54596b]' },
  ];

  const captureRecommendations = [
    { id: 'cap-1', title: 'Corner of 5th & Oak', type: 'Gap Analysis', urgency: 'HIGH' },
    { id: 'cap-2', title: 'Schedule Drone Flight for Riverside', type: 'Occlusion', urgency: 'MED' },
    { id: 'cap-3', title: 'North Facade Lower Cornice', type: 'Low Density', urgency: 'LOW' },
  ];

  const evidenceThumbnails = [
    { id: 'th-1', label: 'RGB Photo #345', modality: 'RGB', color: 'from-blue-900 to-indigo-900' },
    { id: 'th-2', label: 'Depth Frame', modality: 'DEPTH', color: 'from-emerald-900 to-teal-900' },
    { id: 'th-3', label: 'Point Cloud Slice', modality: 'PLY', color: 'from-amber-900 to-orange-900' },
    { id: 'th-4', label: 'Surface Normals', modality: 'NRM', color: 'from-purple-900 to-fuchsia-900' },
  ];

  if (!bottomDrawerOpen) return null;

  return (
    <div
      className="flex flex-col h-48 w-full bg-[#0e1015] border-t border-[#1f222b] select-none text-[#ededf2] font-mono shrink-0 overflow-hidden shadow-2xl transition-all"
      aria-label="Contextual Bottom Drawer"
    >
      {/* ── Top Drawer Header Bar ── */}
      <div className="h-7 px-3 bg-[#0a0b0e] border-b border-[#1f222b] flex items-center justify-between shrink-0 text-[10px]">
        <div className="flex items-center gap-2">
          <span className="font-bold tracking-wider text-[#9296a6] uppercase">
            Contextual Timeline / Processing / Capture / Evidence
          </span>
          <span className="text-[#54596b]">•</span>
          <span className="text-[#00e5ff] font-semibold">WORLD CONSTRUCTED (2.4 km² urban area)</span>
        </div>

        <div className="flex items-center gap-2 text-[#54596b]">
          <span className="hidden md:inline">⌘J: Toggle Dock</span>
          <button
            type="button"
            onClick={() => toggleBottomDrawer()}
            className="p-0.5 rounded hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
            title="Close Drawer (⌘J)"
          >
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── 4-Column Workstation Grid ── */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-[#1f222b] overflow-hidden text-xs">
        {/* COLUMN 1: TIMELINE */}
        <div className="flex flex-col p-2.5 overflow-hidden">
          <div className="flex items-center justify-between pb-1 mb-1 border-b border-white/5">
            <span className="text-[10px] font-bold text-[#9296a6] uppercase flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-[#38bdf8]" />
              <span>TIMELINE</span>
            </span>

            <button
              type="button"
              onClick={() => setIsPlaying(!isPlaying)}
              className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-[#161a24] text-[#00e5ff] hover:bg-[#00e5ff]/20 text-[9px]"
            >
              {isPlaying ? <Pause className="w-2.5 h-2.5" /> : <Play className="w-2.5 h-2.5" />}
              <span>{isPlaying ? 'PAUSE' : 'PLAY'}</span>
            </button>
          </div>

          <div className="flex-1 flex flex-col justify-center">
            <TimelinePrimitive
              stages={stages}
              activeStageId={activeStageId}
              onSelectStage={(id) => {
                setActiveStageId(id);
                addNotification({
                  type: 'info',
                  title: 'Timeline Stage Selected',
                  message: `Switched view to stage: ${id.replace('stage-', '')}`,
                });
              }}
              showTicks={true}
            />
          </div>
        </div>

        {/* COLUMN 2: PROCESSING */}
        <div className="flex flex-col p-2.5 overflow-hidden">
          <div className="flex items-center justify-between pb-1 mb-1.5 border-b border-white/5">
            <span className="text-[10px] font-bold text-[#9296a6] uppercase flex items-center gap-1.5">
              <Cpu className="w-3 h-3 text-[#2ecc71]" />
              <span>PROCESSING</span>
            </span>
            <span className="text-[9px] text-[#2ecc71] font-bold animate-pulse">● LIVE</span>
          </div>

          <div className="flex-1 space-y-2 overflow-y-auto pr-1">
            {processingJobs.map((job) => (
              <div key={job.name} className="space-y-1">
                <div className="flex items-center justify-between text-[10px]">
                  <span className="text-[#f0f1f6] truncate max-w-[140px]">{job.name}</span>
                  <span className="text-[#9296a6] text-[9px]">{job.status}</span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-[#151821] overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${job.color}`}
                    style={{ width: `${job.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* COLUMN 3: CAPTURE RECOMMENDATIONS */}
        <div className="flex flex-col p-2.5 overflow-hidden">
          <div className="flex items-center justify-between pb-1 mb-1.5 border-b border-white/5">
            <span className="text-[10px] font-bold text-[#9296a6] uppercase flex items-center gap-1.5">
              <Camera className="w-3 h-3 text-[#f5a623]" />
              <span>CAPTURE GUIDANCE</span>
            </span>
            <span className="text-[9px] text-[#54596b] font-mono">3 Tasks</span>
          </div>

          <div className="flex-1 space-y-1.5 overflow-y-auto pr-1">
            {captureRecommendations.map((rec) => (
              <div
                key={rec.id}
                className="p-1.5 rounded bg-[#14161f] border border-[#1f222b] flex items-center justify-between hover:border-[#f5a623]/40 cursor-pointer transition-colors"
                onClick={() => {
                  addNotification({
                    type: 'warning',
                    title: 'Capture Mission Created',
                    message: `Dispatched capture cue: ${rec.title}`,
                  });
                }}
              >
                <div className="truncate pr-1">
                  <div className="text-[10px] font-bold text-[#f0f1f6] truncate">{rec.title}</div>
                  <div className="text-[8px] text-[#9296a6]">{rec.type}</div>
                </div>
                <span
                  className={`text-[8px] font-mono px-1 rounded shrink-0 ${
                    rec.urgency === 'HIGH'
                      ? 'bg-[#e54d4d]/20 text-[#e54d4d]'
                      : 'bg-[#f5a623]/20 text-[#f5a623]'
                  }`}
                >
                  {rec.urgency}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* COLUMN 4: EVIDENCE THUMBNAILS */}
        <div className="flex flex-col p-2.5 overflow-hidden">
          <div className="flex items-center justify-between pb-1 mb-1.5 border-b border-white/5">
            <span className="text-[10px] font-bold text-[#9296a6] uppercase flex items-center gap-1.5">
              <Layers className="w-3 h-3 text-[#00e5ff]" />
              <span>EVIDENCE SOURCES</span>
            </span>
            <span className="text-[9px] text-[#54596b]">4 Keyframes</span>
          </div>

          <div className="flex-1 grid grid-cols-2 gap-1.5 overflow-y-auto">
            {evidenceThumbnails.map((th) => (
              <div
                key={th.id}
                className={`relative h-14 rounded bg-gradient-to-br ${th.color} border border-white/10 p-1 flex flex-col justify-between hover:border-[#00e5ff] cursor-pointer transition-all`}
                onClick={() => {
                  addNotification({
                    type: 'info',
                    title: 'Evidence Inspected',
                    message: `Opened source frame: ${th.label}`,
                  });
                }}
              >
                <span className="text-[8px] font-bold px-1 rounded bg-black/60 text-white w-max">
                  {th.modality}
                </span>
                <span className="text-[9px] text-white/90 truncate font-sans">
                  {th.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
