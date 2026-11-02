"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { getHealth, useWorlds } from "@/lib/api";
import { connectionState, worldIdFromPath } from "@/lib/shell/shell-logic";
import TopBar from "./TopBar";
import Sidebar, { MobileTabBar } from "./Sidebar";
import StatusFooter from "./StatusFooter";
import CommandPalette from "./CommandPalette";

/**
 * One shell for every desktop route (including the 3D workspace at
 * /worlds/[id]): top bar, left icon rail, content area, status footer.
 * The workspace keeps its own toolbar as a sub-toolbar inside the content area.
 */
export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);
  const { data: worlds, error: worldsError } = useWorlds();
  const [health, setHealth] = useState<{ ok: boolean } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () =>
      getHealth().then(
        () => !cancelled && setHealth({ ok: true }),
        () => !cancelled && setHealth({ ok: false }),
      );
    poll();
    const timer = setInterval(poll, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Apply the user's persisted reduce-motion preference (set in Settings)
  // on every load, not just while the Settings page itself is mounted.
  useEffect(() => {
    try {
      if (window.localStorage.getItem("re-reduce-motion") === "true") {
        document.documentElement.setAttribute("data-reduce-motion", "true");
      }
    } catch {
      // Storage unavailable — falls back to the OS-level media query only.
    }
  }, []);

<<<<<<< HEAD
  const currentWorldId = worldIdFromPath(pathname);
  const worldName = currentWorldId ? (worlds?.find((w) => w.id === currentWorldId)?.name ?? null) : null;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: "var(--bg-base)" }}>
      <TopBar
        onOpenSearch={() => setPaletteOpen(true)}
        worlds={worlds}
        worldsError={worldsError ? worldsError.message : null}
        currentWorldId={currentWorldId}
        connection={connectionState(health)}
      />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 min-w-0 h-full overflow-hidden relative pb-14 md:pb-0">{children}</main>
=======
  // World workspaces (`/worlds/[id]`) occupy the full viewport without dashboard letterboxing.
  // Home (`/`) is the place library and keeps the app chrome.
  const isSpatialStudio = pathname.startsWith("/worlds/") && pathname !== "/worlds/new";

  if (isSpatialStudio) {
    return (
      <div className="flex h-screen w-screen overflow-hidden" style={{ background: "var(--bg-base)" }}>
        <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:px-3 focus:py-2 focus:text-sm"
        style={{ background: "var(--bg-elevated)", color: "var(--text-primary)", border: "1px solid var(--border-strong)" }}
      >
        Skip to content
      </a>
        <main id="main" className="flex-1 min-w-0 h-full overflow-hidden relative">
          {children}
        </main>
        <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:px-3 focus:py-2 focus:text-sm"
        style={{ background: "var(--bg-elevated)", color: "var(--text-primary)", border: "1px solid var(--border-strong)" }}
      >
        Skip to content
      </a>
      <TopBar onOpenSearch={() => setPaletteOpen(true)} />
      <div className="flex flex-1 min-h-0">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
        <main id="main" className="flex-1 min-w-0 h-full overflow-hidden">{children}</main>
>>>>>>> 4027b5c
      </div>
      <StatusFooter worldName={worldName} connection={connectionState(health)} />
      <MobileTabBar />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
