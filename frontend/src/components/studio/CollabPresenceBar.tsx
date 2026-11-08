"use client";

import React from "react";
import {
  Eye,
  MapPin,
  MessageSquare,
  Users,
} from "lucide-react";
import type { PeerPresence } from "@/lib/studio/collab-presence-model";

export interface CollabPresenceBarProps {
  peers: PeerPresence[];
  activeMarkupCount: number;
  followedClientId: string | null;
  onFollowPeer: (peer: PeerPresence | null) => void;
  onOpenMarkups: () => void;
  onAddPin: () => void;
  isPinModeActive?: boolean;
}

export function CollabPresenceBar({
  peers,
  activeMarkupCount,
  followedClientId,
  onFollowPeer,
  onOpenMarkups,
  onAddPin,
  isPinModeActive = false,
}: CollabPresenceBarProps) {
  return (
    <div
      role="region"
      aria-label="Collaborative Telepresence Bar"
      className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-studio-surface/90 backdrop-blur-md border border-studio-border/80 shadow-lg pointer-events-auto"
    >
      {/* Presence count */}
      <div className="flex items-center gap-1.5 pr-2 border-r border-studio-border/60">
        <Users className="w-3.5 h-3.5 text-studio-muted" />
        <span className="text-xs font-mono font-medium text-studio-text">
          {peers.length}
        </span>
        <span className="text-[10px] text-studio-muted hidden sm:inline">online</span>
      </div>

      {/* Peer Avatars */}
      <div className="flex items-center -space-x-1 overflow-x-auto max-w-[180px] sm:max-w-[260px] py-0.5">
        {peers.map((peer) => {
          const isFollowed = followedClientId === peer.client_id;
          const initials = peer.user_name
            .split(" ")
            .map((n) => n[0])
            .join("")
            .slice(0, 2)
            .toUpperCase() || "CL";

          return (
            <button
              key={peer.client_id}
              type="button"
              onClick={() => onFollowPeer(isFollowed ? null : peer)}
              title={`${peer.user_name} (${peer.role}) - ${isFollowed ? "Click to stop following" : "Click to follow camera"}`}
              className={`relative inline-flex items-center justify-center w-6 h-6 rounded-full text-[10px] font-bold border transition-transform hover:scale-110 focus:outline-none ${
                isFollowed
                  ? "ring-2 ring-studio-accent border-studio-surface scale-105 z-10"
                  : "border-studio-surface bg-studio-panel text-studio-text"
              }`}
              style={{
                backgroundColor: `${peer.color}20`,
                borderColor: isFollowed ? undefined : peer.color,
                color: peer.color,
              }}
            >
              {initials}
              {isFollowed && (
                <span className="absolute -bottom-1 -right-1 w-2.5 h-2.5 rounded-full bg-studio-accent border border-studio-surface flex items-center justify-center">
                  <Eye className="w-1.5 h-1.5 text-studio-surface" />
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1.5 pl-1.5 border-l border-studio-border/60">
        {/* Drop Pin Mode Toggle */}
        <button
          type="button"
          onClick={onAddPin}
          title={isPinModeActive ? "Click 3D surface to place pin" : "Drop Spatial Pin"}
          className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border transition-colors ${
            isPinModeActive
              ? "bg-studio-accent text-studio-surface border-studio-accent shadow-sm"
              : "bg-studio-panel/80 hover:bg-studio-border/30 text-studio-text border-studio-border/60"
          }`}
        >
          <MapPin className="w-3 h-3" />
          <span className="hidden md:inline">
            {isPinModeActive ? "Placing Pin..." : "Add Pin"}
          </span>
        </button>

        {/* Markups List Button */}
        <button
          type="button"
          onClick={onOpenMarkups}
          title="Open Spatial Markups & BCF Issues"
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium bg-studio-panel/80 hover:bg-studio-border/30 text-studio-text border border-studio-border/60 transition-colors"
        >
          <MessageSquare className="w-3 h-3 text-studio-muted" />
          <span className="hidden sm:inline">Markups</span>
          {activeMarkupCount > 0 && (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-studio-border text-studio-text">
              {activeMarkupCount}
            </span>
          )}
        </button>
      </div>
    </div>
  );
}
