"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import TopBar from "./TopBar";
import Sidebar from "./Sidebar";
import CommandPalette from "./CommandPalette";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(true);
  const [paletteOpen, setPaletteOpen] = useState(false);

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

  // Persistent spatial operating environment for spatial studio routes:
  // Roots (`/`) and World workspaces (`/worlds/[id]`) occupy full viewport without dashboard letterboxing.
  const isSpatialStudio =
    pathname === "/" ||
    (pathname.startsWith("/worlds/") && pathname !== "/worlds/new");

  if (isSpatialStudio) {
    return (
      <div className="flex h-screen w-screen overflow-hidden bg-[#08090b]">
        <main className="flex-1 min-w-0 h-full overflow-hidden relative">
          {children}
        </main>
        <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden">
      <TopBar onOpenSearch={() => setPaletteOpen(true)} />
      <div className="flex flex-1 min-h-0">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
        <main className="flex-1 min-w-0 h-full overflow-hidden">{children}</main>
      </div>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
