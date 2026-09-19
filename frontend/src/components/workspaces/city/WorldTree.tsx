'use client';

import React, { useState } from 'react';
import { useREStore } from '@/store/re-store';
import { 
  FolderTree, 
  Layers, 
  ChevronRight, 
  ChevronDown, 
  Eye, 
  EyeOff, 
  CheckSquare, 
  Square, 
  Plus, 
  Building, 
  Compass, 
  Trees, 
  Zap, 
  MapPin, 
  FolderGit2 
} from 'lucide-react';

interface CityNode {
  id: string;
  name: string;
  count: string;
  icon: React.ComponentType<{ className?: string }>;
  visible: boolean;
  children?: { id: string; name: string; type: string }[];
}

export function WorldTree() {
  const { sessions, project } = useREStore();
  const [expandedNodes, setExpandedNodes] = useState<Record<string, boolean>>({
    buildings: true,
    sessions: true,
    roads: true,
  });

  const [activeSessions, setActiveSessions] = useState<Record<string, boolean>>({
    'sess-001': true,
    'sess-002': true,
    'sess-003': false,
  });

  const toggleExpand = (id: string) => {
    setExpandedNodes((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleSession = (id: string) => {
    setActiveSessions((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const cityLayers: CityNode[] = [
    { id: 'terrain', name: 'Terrain & Elevation', count: '1 DEM', icon: Compass, visible: true },
    { id: 'roads', name: 'Roads & Transit', count: '14 Segments', icon: MapPin, visible: true, children: [
      { id: 'r1', name: 'National Arterial 12', type: 'Highway 4-Lane' },
      { id: 'r2', name: 'Industrial Access Spur', type: 'Local 2-Lane' },
    ]},
    { id: 'buildings', name: 'Buildings & Parcels', count: '8 Structures', icon: Building, visible: true, children: [
      { id: 'b1', name: 'Building A (Main Admin)', type: 'LOD2 Mesh' },
      { id: 'b2', name: 'Warehouse Block B', type: 'Procedural Box' },
      { id: 'b3', name: 'Substation Control Room', type: 'Observed BIM' },
    ]},
    { id: 'infrastructure', name: 'Utilities & Power', count: '24 Nodes', icon: Zap, visible: true },
    { id: 'vegetation', name: 'Canopy & Greenery', count: '340 Trees', icon: Trees, visible: true },
  ];

  return (
    <div className="flex flex-col h-full bg-[#121215] text-[#e8e8f0] select-none text-xs">
      {/* Panel Header */}
      <div className="flex items-center justify-between px-3 h-[32px] bg-[#17171c] border-b border-[#1e1e28]">
        <div className="flex items-center gap-1.5 font-semibold text-[11px] uppercase tracking-wider text-[#9898b0]">
          <FolderTree className="w-3.5 h-3.5 text-[#3d8ef7]" />
          <span>City Assembly Outliner</span>
        </div>
        <span className="text-[10px] font-mono text-[#5c5c78]">Kolkata City</span>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-4">
        {/* Section 1: Multi-Session Compositor */}
        <div className="space-y-1">
          <div 
            onClick={() => toggleExpand('sessions')}
            className="flex items-center justify-between py-1 px-1.5 rounded hover:bg-[#1c1c23] cursor-pointer text-[#9898b0] hover:text-[#e8e8f0]"
          >
            <div className="flex items-center gap-1.5 font-semibold text-[10px] uppercase tracking-wider">
              {expandedNodes['sessions'] ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <FolderGit2 className="w-3.5 h-3.5 text-[#3d8ef7]" />
              <span>Contributing Sessions ({sessions.length})</span>
            </div>
            <button 
              title="Add Session from Ingestion Store"
              className="p-0.5 hover:bg-[#272733] rounded text-[#3d8ef7]"
            >
              <Plus className="w-3 h-3" />
            </button>
          </div>

          {expandedNodes['sessions'] && (
            <div className="space-y-1 pl-3 pt-1">
              {sessions.map((sess) => {
                const isFused = activeSessions[sess.id] ?? false;
                return (
                  <div
                    key={sess.id}
                    onClick={() => toggleSession(sess.id)}
                    className={`flex items-center justify-between p-1.5 rounded border transition-colors cursor-pointer ${
                      isFused ? 'bg-[#17171c] border-[#272733]' : 'bg-[#121215] border-[#1e1e28] opacity-50'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      {isFused ? (
                        <CheckSquare className="w-3.5 h-3.5 text-[#34c76f] flex-shrink-0" />
                      ) : (
                        <Square className="w-3.5 h-3.5 text-[#5c5c78] flex-shrink-0" />
                      )}
                      <div className="truncate">
                        <div className="text-[11px] font-medium text-[#e8e8f0] truncate">{sess.name}</div>
                        <div className="text-[9px] font-mono text-[#5c5c78]">
                          {sess.imageCount > 0 ? `${sess.imageCount} imgs` : 'LiDAR Cloud'} • {sess.status}
                        </div>
                      </div>
                    </div>

                    <span className="text-[9px] font-mono text-[#3d8ef7] px-1.5 py-0.5 bg-[#22222c] rounded">
                      {isFused ? 'FUSED' : 'MUTED'}
                    </span>
                  </div>
                );
              })}

              <div className="text-[9px] text-[#5c5c78] font-mono italic px-1 pt-0.5">
                {"// WIRE: POST /api/worlds/compose {sessionIds}"}
              </div>
            </div>
          )}
        </div>

        <div className="w-full h-[1px] bg-[#1e1e28]" />

        {/* Section 2: City Semantic Layers */}
        <div className="space-y-1">
          <div className="text-[10px] font-semibold text-[#5c5c78] uppercase tracking-wider px-1 mb-1">
            WorldIR Semantic Hierarchy
          </div>

          {cityLayers.map((layer) => {
            const Icon = layer.icon;
            const isExp = expandedNodes[layer.id];
            return (
              <div key={layer.id} className="space-y-0.5">
                <div 
                  onClick={() => layer.children && toggleExpand(layer.id)}
                  className="flex items-center justify-between py-1 px-1.5 rounded hover:bg-[#1c1c23] cursor-pointer group"
                >
                  <div className="flex items-center gap-2">
                    {layer.children ? (
                      isExp ? <ChevronDown className="w-3 h-3 text-[#5c5c78]" /> : <ChevronRight className="w-3 h-3 text-[#5c5c78]" />
                    ) : (
                      <span className="w-3" />
                    )}
                    <Icon className="w-3.5 h-3.5 text-[#9898b0] group-hover:text-[#3d8ef7]" />
                    <span className="text-[11px] font-medium text-[#e8e8f0]">{layer.name}</span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-[9px] font-mono text-[#5c5c78]">{layer.count}</span>
                    <button className="text-[#5c5c78] hover:text-[#e8e8f0]">
                      <Eye className="w-3 h-3" />
                    </button>
                  </div>
                </div>

                {isExp && layer.children && (
                  <div className="pl-6 space-y-0.5">
                    {layer.children.map((ch) => (
                      <div 
                        key={ch.id}
                        className="flex items-center justify-between py-1 px-1.5 rounded hover:bg-[#17171c] cursor-pointer text-[#9898b0] hover:text-[#3d8ef7]"
                      >
                        <span className="text-[11px] truncate">{ch.name}</span>
                        <span className="text-[9px] font-mono text-[#5c5c78]">{ch.type}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
