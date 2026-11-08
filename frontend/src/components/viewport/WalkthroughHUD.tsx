"use client";

import React, { useState } from "react";
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Bookmark,
  BookmarkPlus,
  Footprints,
  Compass,
  X,
  Volume2,
} from "lucide-react";
import type { NavigationData, SpatialBookmark, TourKeyframe } from "@/lib/api/navigation";
import { formatTourTime } from "@/lib/viewport/tour-model";

interface WalkthroughHUDProps {
  navData: NavigationData | null;
  activeRoomName: string;
  isTourPlaying: boolean;
  tourElapsedSeconds: number;
  tourTotalSeconds: number;
  tourNarration: string;
  tourSpeed: number;
  onToggleTourPlay: () => void;
  onNextStop: () => void;
  onPrevStop: () => void;
  onSetTourSpeed: (speed: number) => void;
  onSelectBookmark: (bmk: SpatialBookmark) => void;
  onAddBookmark: (name: string) => Promise<void>;
  onExitWalkMode: () => void;
}

export default function WalkthroughHUD({
  navData,
  activeRoomName,
  isTourPlaying,
  tourElapsedSeconds,
  tourTotalSeconds,
  tourNarration,
  tourSpeed,
  onToggleTourPlay,
  onNextStop,
  onPrevStop,
  onSetTourSpeed,
  onSelectBookmark,
  onAddBookmark,
  onExitWalkMode,
}: WalkthroughHUDProps) {
  const [showAddBookmark, setShowAddBookmark] = useState(false);
  const [newBookmarkName, setNewBookmarkName] = useState("");
  const [savingBookmark, setSavingBookmark] = useState(false);

  const handleSaveBookmark = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newBookmarkName.trim()) return;
    setSavingBookmark(true);
    try {
      await onAddBookmark(newBookmarkName.trim());
      setNewBookmarkName("");
      setShowAddBookmark(false);
    } finally {
      setSavingBookmark(false);
    }
  };

  const progressPct =
    tourTotalSeconds > 0
      ? Math.min(100, Math.max(0, (tourElapsedSeconds / tourTotalSeconds) * 100))
      : 0;

  return (
    <div
      data-testid="walkthrough-hud"
      className="absolute top-4 left-1/2 -translate-x-1/2 z-20 flex flex-col items-center gap-2 max-w-xl w-[90%] pointer-events-auto select-none"
    >
      {/* Top Main Control Bar */}
      <div
        data-testid="walk-mode-hud"
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--bg-surface)]/95 border border-[var(--border)] shadow-xl backdrop-blur-md text-xs"
      >
        {/* Room Indicator */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[var(--accent-subtle)] text-[var(--accent)] font-semibold border border-[var(--accent-border)]">
          <Compass className="w-3.5 h-3.5" />
          <span className="truncate max-w-[130px]">{activeRoomName || "Interior Space"}</span>
        </div>

        <div className="w-[1px] h-4 bg-[var(--border-subtle)]" />

        {/* Tour Transport Controls */}
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onPrevStop}
            title="Previous stop"
            className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={onToggleTourPlay}
            data-testid="toggle-tour-play"
            title={isTourPlaying ? "Pause tour" : "Play automated tour"}
            className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-[var(--primary)] text-[var(--on-primary)] font-semibold hover:bg-[var(--primary-hover)] transition-colors cursor-pointer"
          >
            {isTourPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3 fill-current" />}
            <span>{isTourPlaying ? "Pause" : "Tour"}</span>
          </button>

          <button
            type="button"
            onClick={onNextStop}
            title="Next stop"
            className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Tour Timer & Speed */}
        <div className="flex items-center gap-1.5 font-mono text-[11px] text-[var(--text-tertiary)] px-1">
          <span>{formatTourTime(tourElapsedSeconds)}</span>
          <span>/</span>
          <span>{formatTourTime(tourTotalSeconds)}</span>
        </div>

        <button
          type="button"
          onClick={() => onSetTourSpeed(tourSpeed === 1 ? 1.5 : tourSpeed === 1.5 ? 2 : 1)}
          title="Toggle tour speed"
          className="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border-subtle)] cursor-pointer"
        >
          {tourSpeed}x
        </button>

        <div className="w-[1px] h-4 bg-[var(--border-subtle)]" />

        {/* Bookmarks Selector */}
        {navData && navData.suggested_bookmarks.length > 0 && (
          <select
            onChange={(e) => {
              const bmk = navData.suggested_bookmarks.find((b) => b.bookmark_id === e.target.value);
              if (bmk) onSelectBookmark(bmk);
            }}
            defaultValue=""
            className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-2 py-0.5 text-[11px] text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)] max-w-[110px] truncate"
          >
            <option value="" disabled>
              Bookmarks ({navData.suggested_bookmarks.length})
            </option>
            {navData.suggested_bookmarks.map((b) => (
              <option key={b.bookmark_id} value={b.bookmark_id}>
                {b.name}
              </option>
            ))}
          </select>
        )}

        {/* Add Bookmark Button */}
        <button
          type="button"
          onClick={() => setShowAddBookmark(!showAddBookmark)}
          title="Save current viewpoint"
          className="p-1 rounded text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
        >
          <BookmarkPlus className="w-3.5 h-3.5" />
        </button>

        <div className="w-[1px] h-4 bg-[var(--border-subtle)]" />

        {/* Exit Walk Mode */}
        <button
          type="button"
          onClick={onExitWalkMode}
          title="Exit walk mode (Esc)"
          data-testid="exit-walk-mode"
          className="p-1 rounded text-[var(--text-tertiary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Tour Timeline Progress Scrubber */}
      {tourTotalSeconds > 0 && (
        <div className="w-full max-w-sm h-1 bg-[var(--bg-surface)]/80 rounded-full overflow-hidden border border-[var(--border-subtle)]">
          <div
            className="h-full bg-[var(--accent)] transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      )}

      {/* Narration Subtitle Banner */}
      {isTourPlaying && tourNarration && (
        <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-[var(--bg-surface)]/95 border border-[var(--border)] shadow-md text-xs text-[var(--text-primary)] animate-in fade-in slide-in-from-top-1 duration-200">
          <Volume2 className="w-3.5 h-3.5 text-[var(--accent)] shrink-0" />
          <span>{tourNarration}</span>
        </div>
      )}

      {/* Add Bookmark Quick Popup */}
      {showAddBookmark && (
        <form
          onSubmit={handleSaveBookmark}
          className="flex items-center gap-1.5 p-2 rounded-lg bg-[var(--bg-surface)] border border-[var(--border)] shadow-xl text-xs"
        >
          <Bookmark className="w-3.5 h-3.5 text-[var(--accent)]" />
          <input
            type="text"
            value={newBookmarkName}
            onChange={(e) => setNewBookmarkName(e.target.value)}
            placeholder="e.g. Master Suite Window"
            autoFocus
            className="bg-[var(--bg-elevated)] border border-[var(--border)] rounded px-2 py-1 text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent)] text-xs w-44"
          />
          <button
            type="submit"
            disabled={savingBookmark || !newBookmarkName.trim()}
            className="px-2.5 py-1 rounded bg-[var(--primary)] text-[var(--on-primary)] font-medium hover:bg-[var(--primary-hover)] disabled:opacity-50 transition-colors cursor-pointer"
          >
            {savingBookmark ? "Saving..." : "Save"}
          </button>
          <button
            type="button"
            onClick={() => setShowAddBookmark(false)}
            className="p-1 rounded text-[var(--text-tertiary)] hover:text-[var(--text-primary)]"
          >
            <X className="w-3 h-3" />
          </button>
        </form>
      )}
    </div>
  );
}
