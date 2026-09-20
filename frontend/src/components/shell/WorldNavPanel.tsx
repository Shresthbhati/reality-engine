'use client';

import React, { useState, useMemo, useEffect } from 'react';
import { useREStore } from '@/store/re-store';
import type { SpatialScale } from '@/types/reality-engine';
import {
  Globe,
  Camera,
  Layers,
  MapPin,
  Bookmark,
  History,
  ChevronDown,
  ChevronRight,
  Search,
  PanelLeftClose,
  Plus,
  CheckCircle2,
  AlertCircle,
  FileText,
  Image as ImageIcon,
  Boxes,
  RotateCw,
} from 'lucide-react';

interface NavSection {
  id: string;
  title: string;
  icon: React.ReactNode;
}

export default function WorldNavPanel() {
  const selectedWorldId = useREStore((s) => s.selectedWorldId);
  const setSelectedWorldId = useREStore((s) => s.setSelectedWorldId);
  const selectedSessionId = useREStore((s) => s.selectedSessionId);
  const selectedPlaceId = useREStore((s) => s.selectedPlaceId);
  const setSelectedPlaceId = useREStore((s) => s.setSelectedPlaceId);
  const selectedBookmarkId = useREStore((s) => s.selectedBookmarkId);
  const setSelectedBookmarkId = useREStore((s) => s.setSelectedBookmarkId);
  const selectedVersionId = useREStore((s) => s.selectedVersionId);
  const setSelectedVersionId = useREStore((s) => s.setSelectedVersionId);
  const toggleLeftNav = useREStore((s) => s.toggleLeftNav);
  const setSpatialScale = useREStore((s) => s.setSpatialScale);
  const addNotification = useREStore((s) => s.addNotification);

  // Backend state integration
  const worldVersions = useREStore((s) => s.worldVersions);
  const activeWorldVersion = useREStore((s) => s.activeWorldVersion);
  const loadedFromBackend = useREStore((s) => s.loadedFromBackend);
  const loadWorldFromBackend = useREStore((s) => s.loadWorldFromBackend);
  const refreshWorldVersions = useREStore((s) => s.refreshWorldVersions);
  const backendSessions = useREStore((s) => s.backendSessions);
  const loadSessionsFromBackend = useREStore((s) => s.loadSessionsFromBackend);

  useEffect(() => {
    refreshWorldVersions();
    loadSessionsFromBackend().catch(() => {
      // Listing sessions is best-effort on mount; the panel renders the
      // empty state when the bridge is unreachable. The error state is
      // carried by the store.
    });
  }, [refreshWorldVersions, loadSessionsFromBackend]);

  const [searchQuery, setSearchQuery] = useState('');
  const [openSections, setOpenSections] = useState<Record<string, boolean>>({
    worlds: true,
    sessions: true,
    evidence: true,
    places: true,
    bookmarks: true,
    versions: false,
  });

  const toggleSection = (id: string) => {
    setOpenSections((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // NO MOCK DATA: every list below renders real store state. Worlds /
  // versions come from the backend's WorldStore; SESSIONS from the
  // canonical SessionWorkspace via /api/sessions. A section with no
  // backend data renders an explicit "none recorded" row instead of
  // demo entries (the previous hardcoded 'Laser Scan', 'Drone
  // Photogrammetry', 'City Hall Plaza', 'Bridge Crack V7' etc. were
  // fabricated and are gone).
  const sessions = backendSessions;  const displayWorlds = useMemo(() => {
    return worldVersions.map((wv) => ({
      id: wv.world_id,
      versionId: wv.version_id,
      name: wv.name || wv.world_id,
      version: wv.version_id,
      status: activeWorldVersion === wv.version_id ? 'Active' : 'Available',
      area: `${wv.entity_count} nodes`,
      isBackend: true,
    }));
  }, [worldVersions, activeWorldVersion]);

  const displayVersions = useMemo(() => {
    return worldVersions.map((wv) => ({
      id: wv.version_id,
      label: `${wv.version_id} (${wv.entity_count} nodes)`,
      isCurrent: activeWorldVersion === wv.version_id,
      isBackend: true,
    }));
  }, [worldVersions, activeWorldVersion]);

  // Aggregate the real source-type counts across backend sessions
  // (source_types comes from sources.json via the bridge).
  const sourceTypeCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const sess of backendSessions) {
      for (const [t, n] of Object.entries(sess.source_types ?? {})) {
        counts.set(t, (counts.get(t) ?? 0) + (n ?? 0));
      }
    }
    return Array.from(counts.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [backendSessions]);

  const q = searchQuery.toLowerCase().trim();

  return (
    <aside
      className="flex flex-col w-64 h-full bg-[#101217] border-r border-[#1f222b] select-none text-[#ededf2] font-sans shrink-0 overflow-hidden"
      aria-label="World Navigation Panel"
    >
      {/* ── Header ── */}
      <div className="h-10 px-3 border-b border-[#1f222b] flex items-center justify-between shrink-0 bg-[#0d0e12]">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-mono font-bold tracking-wider text-[#9296a6] uppercase">
            World Navigation
          </span>
          {loadedFromBackend && (
            <span className="w-1.5 h-1.5 rounded-full bg-[#2ecc71]" title="Backend Connected" />
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => {
              refreshWorldVersions();
              addNotification({
                type: 'info',
                title: 'Refreshed',
                message: 'Refreshed WorldStore versions from backend.',
              });
            }}
            className="p-1 rounded text-[#54596b] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
            title="Refresh Backend Versions"
          >
            <RotateCw className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={toggleLeftNav}
            className="p-1 rounded text-[#54596b] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
            title="Collapse Panel (⌘B)"
          >
            <PanelLeftClose className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Search Filter ── */}
      <div className="p-2 border-b border-[#1f222b] bg-[#0f1116] shrink-0">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-[#151821] border border-[#1f2433] text-xs">
          <Search className="w-3.5 h-3.5 text-[#54596b]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter entities, sessions..."
            className="w-full bg-transparent outline-none text-[11px] text-[#f0f1f6] placeholder-[#54596b]"
          />
        </div>
      </div>

      {/* ── Accordion List ── */}
      <div className="flex-1 overflow-y-auto px-1 py-1 space-y-1 text-xs">
        {/* 1. WORLDS */}
        <div>
          <button
            type="button"
            onClick={() => toggleSection('worlds')}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 text-[10px] font-mono font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Globe className="w-3.5 h-3.5 text-[#00e5ff]" />
              <span>WORLDS</span>
            </span>
            {openSections.worlds ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.worlds && (
            <div className="mt-0.5 space-y-0.5 pl-2">
              {displayWorlds
                .filter((w) => !q || w.name.toLowerCase().includes(q) || w.version.toLowerCase().includes(q))
                .map((w) => {
                  const isSelected = selectedWorldId === w.id || activeWorldVersion === w.versionId;
                  return (
                    <div
                      key={`${w.id}-${w.versionId}`}
                      onClick={() => {
                        if (w.isBackend) {
                          loadWorldFromBackend(w.versionId);
                        } else {
                          setSelectedWorldId(w.id);
                          addNotification({
                            type: 'info',
                            title: 'World Loaded',
                            message: `Loaded spatial world: ${w.name} (${w.version})`,
                          });
                        }
                      }}
                      className={`flex items-center justify-between px-2 py-1 rounded cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-[#00e5ff]/15 text-[#00e5ff] font-semibold border border-[#00e5ff]/30'
                          : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5'
                      }`}
                    >
                      <div className="flex items-center gap-1.5 truncate">
                        <Globe className="w-3 h-3 shrink-0" />
                        <span className="truncate text-[11px]">{w.name}</span>
                        <span className="text-[9px] font-mono text-[#54596b]">{w.version}</span>
                      </div>
                      <span
                        className={`text-[9px] font-mono px-1 rounded ${
                          w.status === 'Active'
                            ? 'bg-[#2ecc71]/20 text-[#2ecc71]'
                            : 'bg-white/5 text-[#54596b]'
                        }`}
                      >
                        {w.status}
                      </span>
                    </div>
                  );
                })}
            </div>
          )}
        </div>

        {/* 2. SESSIONS — real SessionWorkspace sessions. */}
        <div>
          <button
            type="button"
            onClick={() => toggleSection('sessions')}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 text-[10px] font-mono font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Camera className="w-3.5 h-3.5 text-[#38bdf8]" />
              <span>SESSIONS</span>
            </span>
            {openSections.sessions ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.sessions && (
            <div className="mt-0.5 space-y-0.5 pl-2">
              {sessions
                .filter((s) => !q || (s.name || s.id).toLowerCase().includes(q))
                .map((s) => {
                  const isSelected = selectedSessionId === s.id;
                  const frameCount = s.evidence_count ?? 0;
                  const countLabel = frameCount > 0 ? `${frameCount} evidence` : `${s.source_count ?? 0} source${(s.source_count ?? 0) === 1 ? '' : 's'}`;
                  return (
                    <div
                      key={s.id}
                      className={`flex flex-col px-2 py-1 rounded cursor-pointer transition-colors ${
                        isSelected
                          ? 'bg-[#38bdf8]/15 text-[#38bdf8] border border-[#38bdf8]/30'
                          : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5'
                      }`}
                    >
                      <span className="text-[11px] truncate">{s.name || s.id}</span>
                      <span className="text-[9px] font-mono text-[#54596b]">
                        {s.status || 'open'} · {countLabel}
                      </span>
                    </div>
                  );
                })}
              {sessions.length === 0 && (
                <div className="px-2 py-1 text-[9px] font-mono text-[#54596b]">
                  No sessions recorded yet — import one in the Loader.
                </div>
              )}
            </div>
          )}
        </div>

        {/* 3. EVIDENCE */}
        <div>
          <button
            type="button"
            onClick={() => toggleSection('evidence')}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 text-[10px] font-mono font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-[#2ecc71]" />
              <span>EVIDENCE</span>
            </span>
            {openSections.evidence ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.evidence && (
            <div className="mt-0.5 space-y-0.5 pl-2">
              {sourceTypeCounts.map(([type, count]) => (
                <div
                  key={type}
                  className="flex items-center justify-between px-2 py-1 rounded cursor-pointer text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5 transition-colors"
                >
                  <div className="flex items-center gap-1.5 truncate">
                    <Layers className="w-3.5 h-3.5 text-[#2ecc71] shrink-0" />
                    <span className="text-[11px] truncate">{type}</span>
                  </div>
                  <span className="text-[9px] font-mono text-[#54596b]">{count}</span>
                </div>
              ))}
              {sourceTypeCounts.length === 0 && (
                <div className="px-2 py-1 text-[9px] font-mono text-[#54596b]">
                  No sources recorded yet.
                </div>
              )}
            </div>
          )}
        </div>

        {/* 4. PLACES */}
        <div>
          <button
            type="button"
            onClick={() => toggleSection('places')}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 text-[10px] font-mono font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <MapPin className="w-3.5 h-3.5 text-[#f5a623]" />
              <span>PLACES</span>
            </span>
            {openSections.places ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.places && (
            <div className="mt-0.5 space-y-0.5 pl-2">
              <div className="px-2 py-1 text-[9px] font-mono text-[#54596b]">
                Saved places are not stored by the backend yet — nothing is
                invented here. Navigate the world via WORLDS and VERSIONS.
              </div>
            </div>
          )}
        </div>

        {/* 5. BOOKMARKS */}
        <div>
          <button
            type="button"
            onClick={() => toggleSection('bookmarks')}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 text-[10px] font-mono font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <Bookmark className="w-3.5 h-3.5 text-[#a855f7]" />
              <span>BOOKMARKS</span>
            </span>
            {openSections.bookmarks ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.bookmarks && (
            <div className="mt-0.5 space-y-0.5 pl-2">
              <div className="px-2 py-1 text-[9px] font-mono text-[#54596b]">
                Bookmarks are not persisted by the backend yet — none are
                shown rather than fabricated.
              </div>
            </div>
          )}
        </div>

        {/* 6. VERSIONS */}
        <div>
          <button
            type="button"
            onClick={() => toggleSection('versions')}
            className="w-full flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/5 text-[10px] font-mono font-bold text-[#54596b] uppercase"
          >
            <span className="flex items-center gap-1.5">
              <History className="w-3.5 h-3.5 text-[#ec4899]" />
              <span>VERSIONS</span>
            </span>
            {openSections.versions ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
          </button>

          {openSections.versions && (
            <div className="mt-0.5 space-y-0.5 pl-2">
              {displayVersions.map((v) => {
                const isSelected = activeWorldVersion === v.id || selectedVersionId === v.id;
                return (
                  <div
                    key={v.id}
                    onClick={() => {
                      if (v.isBackend) {
                        loadWorldFromBackend(v.id);
                      } else {
                        setSelectedVersionId(v.id);
                      }
                    }}
                    className={`flex items-center justify-between px-2 py-1 rounded cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-[#ec4899]/15 text-[#ec4899] border border-[#ec4899]/30 font-semibold'
                        : 'text-[#9296a6] hover:text-[#f0f1f6] hover:bg-white/5'
                    }`}
                  >
                    <span className="text-[11px] truncate">{v.label}</span>
                    {v.isCurrent && (
                      <span className="w-1.5 h-1.5 rounded-full bg-[#ec4899] shrink-0" />
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
