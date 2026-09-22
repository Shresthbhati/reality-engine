"use client";

import { useEffect, useState } from "react";
import TopBar from "./TopBar";
import Sidebar from "./Sidebar";
import CommandPalette from "./CommandPalette";

export default function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
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

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden">
      <TopBar onOpenSearch={() => setPaletteOpen(true)} />
      <div className="flex flex-1 min-h-0">
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
        <main className="flex-1 min-w-0 overflow-y-auto">{children}</main>
      </div>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </div>
  );
}
