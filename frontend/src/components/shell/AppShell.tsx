"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { useWorlds } from "@/lib/api";
import { worldIdFromPath } from "@/lib/shell/shell-logic";
import TopBar from "./TopBar";
import Sidebar from "./Sidebar";
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

  const currentWorldId = worldIdFromPath(pathname);
  const worldName = currentWorldId ? (worlds?.find((w) => w.id === currentWorldId)?.name ?? null) : null;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: "var(--bg-base)" }}>
      <TopBar
        onOpenSearch={() => setPaletteOpen(true)}
        worlds={worlds}
        worldsError={worldsError ? worldsError.message : null}
        currentWorldId={currentWorldId}
      />
      <div className="flex flex-1 min-h-0">
        <Sidebar />
        <main className="flex-1 min-w-0 h-full overflow-hidden relative">{children}</main>
      </div>
      <StatusFooter worldName={worldName} />
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
