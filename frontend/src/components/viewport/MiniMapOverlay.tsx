"use client";

import React, { useState, useMemo } from "react";
import { MapPin, ChevronDown, ChevronUp, Compass } from "lucide-react";
import type { NavigationData, RoomWaypoint, SpatialBookmark } from "@/lib/api/navigation";

interface MiniMapOverlayProps {
  navData: NavigationData | null;
  cameraPos: [number, number, number];
  cameraYaw: number;
  activeRoomName?: string;
  onNavigateToPoint?: (x: number, z: number) => void;
  onSelectBookmark?: (bmk: SpatialBookmark) => void;
}

export default function MiniMapOverlay({
  navData,
  cameraPos,
  cameraYaw,
  activeRoomName,
  onNavigateToPoint,
  onSelectBookmark,
}: MiniMapOverlayProps) {
  const [collapsed, setCollapsed] = useState(false);

  // Map coordinate conversion helpers
  const bounds = navData?.bounds || { min_x: -10, max_x: 10, min_z: -10, max_z: 10 };
  const widthM = Math.max(4, bounds.max_x - bounds.min_x);
  const depthM = Math.max(4, bounds.max_z - bounds.min_z);

  // SVG dimensions
  const svgSize = 160;
  const padding = 16;
  const drawSize = svgSize - padding * 2;

  const toSvgX = (x: number) => {
    return padding + ((x - bounds.min_x) / widthM) * drawSize;
  };

  const toSvgY = (z: number) => {
    return padding + ((z - bounds.min_z) / depthM) * drawSize;
  };

  const toWorldCoords = (svgX: number, svgY: number): [number, number] => {
    const normX = (svgX - padding) / drawSize;
    const normZ = (svgY - padding) / drawSize;
    return [
      bounds.min_x + normX * widthM,
      bounds.min_z + normZ * depthM,
    ];
  };

  const playerSvgX = toSvgX(cameraPos[0]);
  const playerSvgY = toSvgY(cameraPos[2]);

  // Compute vision cone triangle in SVG coordinates
  const coneLength = 22;
  const fovHalfRad = (35 * Math.PI) / 180;
  // Yaw in three.js: -sin(yaw) is forward X, -cos(yaw) is forward Z
  const dirAngle = -cameraYaw - Math.PI / 2;
  const leftAngle = dirAngle - fovHalfRad;
  const rightAngle = dirAngle + fovHalfRad;

  const coneP1 = `${playerSvgX + Math.cos(leftAngle) * coneLength},${
    playerSvgY + Math.sin(leftAngle) * coneLength
  }`;
  const coneP2 = `${playerSvgX + Math.cos(rightAngle) * coneLength},${
    playerSvgY + Math.sin(rightAngle) * coneLength
  }`;

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const [worldX, worldZ] = toWorldCoords(clickX, clickY);
    onNavigateToPoint?.(worldX, worldZ);
  };

  if (!navData || navData.rooms.length === 0) {
    return null;
  }

  return (
    <div
      data-testid="minimap-overlay"
      className="absolute bottom-4 right-4 z-20 flex flex-col items-end pointer-events-auto select-none"
    >
      <div className="bg-[var(--bg-surface)] border border-[var(--border)] rounded-lg shadow-xl overflow-hidden backdrop-blur-md">
        {/* Header bar */}
        <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-[var(--border-subtle)] bg-[var(--bg-elevated)] gap-3 text-xs">
          <div className="flex items-center gap-1.5 text-[var(--text-secondary)] font-medium">
            <Compass className="w-3.5 h-3.5 text-[var(--accent)]" />
            <span className="truncate max-w-[120px] font-semibold text-[var(--text-primary)]">
              {activeRoomName || "Floor Plan"}
            </span>
          </div>
          <button
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? "Expand minimap" : "Collapse minimap"}
            className="p-0.5 text-[var(--text-tertiary)] hover:text-[var(--text-primary)] rounded transition-colors cursor-pointer"
          >
            {collapsed ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Map Canvas */}
        {!collapsed && (
          <div className="p-2 relative bg-[var(--bg-base)]">
            <svg
              width={svgSize}
              height={svgSize}
              onClick={handleSvgClick}
              className="cursor-crosshair rounded bg-[var(--bg-surface)] border border-[var(--border-subtle)]"
            >
              {/* Rooms */}
              {navData.rooms.map((room) => {
                const rx = toSvgX(room.bounds_min[0]);
                const ry = toSvgY(room.bounds_min[2]);
                const rw = Math.max(12, toSvgX(room.bounds_max[0]) - rx);
                const rh = Math.max(12, toSvgY(room.bounds_max[2]) - ry);
                const isCurrent = room.name === activeRoomName;

                return (
                  <g key={room.room_id}>
                    <rect
                      x={rx}
                      y={ry}
                      width={rw}
                      height={rh}
                      fill={isCurrent ? "var(--accent-subtle)" : "var(--bg-elevated)"}
                      stroke={isCurrent ? "var(--accent)" : "var(--border)"}
                      strokeWidth={isCurrent ? "1.5" : "1"}
                      rx="3"
                    />
                    <text
                      x={rx + rw / 2}
                      y={ry + rh / 2 + 3}
                      textAnchor="middle"
                      fill={isCurrent ? "var(--text-primary)" : "var(--text-tertiary)"}
                      fontSize="8"
                      fontFamily="var(--font-sans)"
                      fontWeight={isCurrent ? "600" : "400"}
                      className="pointer-events-none select-none"
                    >
                      {room.name.length > 10 ? `${room.name.slice(0, 8)}…` : room.name}
                    </text>
                  </g>
                );
              })}

              {/* Doors */}
              {navData.doors.map((door) => {
                const dx = toSvgX(door.position[0]);
                const dy = toSvgY(door.position[2]);
                return (
                  <circle
                    key={door.door_id}
                    cx={dx}
                    cy={dy}
                    r="2.5"
                    fill="var(--bg-surface)"
                    stroke="var(--warning)"
                    strokeWidth="1"
                  />
                );
              })}

              {/* Waypoint path trace */}
              {navData.recommended_tour.waypoint_path.length > 1 && (
                <polyline
                  points={navData.recommended_tour.waypoint_path
                    .map((pt) => `${toSvgX(pt[0])},${toSvgY(pt[2])}`)
                    .join(" ")}
                  fill="none"
                  stroke="var(--accent)"
                  strokeWidth="1.2"
                  strokeDasharray="2,2"
                  opacity="0.65"
                />
              )}

              {/* Spatial Bookmark pins */}
              {navData.suggested_bookmarks.map((bmk) => {
                const bx = toSvgX(bmk.position[0]);
                const by = toSvgY(bmk.position[2]);
                return (
                  <g
                    key={bmk.bookmark_id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectBookmark?.(bmk);
                    }}
                    className="cursor-pointer"
                  >
                    <circle cx={bx} cy={by} r="3" fill="var(--success)" opacity="0.8" />
                  </g>
                );
              })}

              {/* Player Frustum Cone & Position */}
              <polygon
                points={`${playerSvgX},${playerSvgY} ${coneP1} ${coneP2}`}
                fill="var(--accent-subtle)"
                stroke="var(--accent)"
                strokeWidth="1"
                opacity="0.7"
              />
              <circle
                cx={playerSvgX}
                cy={playerSvgY}
                r="3.5"
                fill="var(--accent)"
                stroke="var(--bg-surface)"
                strokeWidth="1"
              />
            </svg>

            {/* Quick action legend / help */}
            <div className="flex items-center justify-between mt-1.5 text-[9px] text-[var(--text-tertiary)]">
              <span>Click map to teleport</span>
              <span className="flex items-center gap-1 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" /> Player
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
