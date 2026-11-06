"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  CloudUpload,
  ShieldCheck,
  Box,
  Map as MapIcon,
  Camera,
  FileText,
  Settings,
  MoreHorizontal,
  X,
  type LucideIcon,
} from "lucide-react";
import { RAIL_ITEMS as ALL_RAIL_ITEMS, MOBILE_PRIMARY_IDS, isEntryActive, visibleNav, type NavEntry } from "@/lib/shell/shell-logic";
import { SHOW_UNFINISHED } from "@/lib/unfinished";

const RAIL_ITEMS = visibleNav(ALL_RAIL_ITEMS, SHOW_UNFINISHED);

const ICONS: Record<string, LucideIcon> = {
  home: Home,
  ingest: CloudUpload,
  evidence: ShieldCheck,
  worlds: Box,
  sites: MapIcon,
  sessions: Camera,
  reports: FileText,
  settings: Settings,
};

/** Slim icon rail (md and up): icon over a small label, Settings pinned to the bottom. */
export default function Sidebar() {
  const pathname = usePathname();
  const main = RAIL_ITEMS.filter((e) => e.id !== "settings");
  const bottom = RAIL_ITEMS.filter((e) => e.id === "settings");

  return (
    <aside
      className="hidden md:flex w-[68px] shrink-0 h-full flex-col border-r"
      style={{ background: "var(--bg-sidebar)", borderColor: "var(--border)" }}
    >
      <nav aria-label="Primary" className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-1 px-1.5 py-3">
        {main.map((e) => (
          <RailItem key={e.id} entry={e} active={isEntryActive(e, pathname)} />
        ))}
      </nav>
      <nav aria-label="Secondary" className="flex flex-col gap-1 px-1.5 py-3">
        {bottom.map((e) => (
          <RailItem key={e.id} entry={e} active={isEntryActive(e, pathname)} />
        ))}
      </nav>
    </aside>
  );
}

function RailItem({ entry, active }: { entry: NavEntry; active: boolean }) {
  const Icon = ICONS[entry.id];
  return (
    <Link
      href={entry.href ?? "/"}
      aria-current={active ? "page" : undefined}
      className="flex flex-col items-center justify-center gap-1 h-14 rounded-lg transition-colors hover:bg-[var(--bg-hover)]"
      style={{
        color: active ? "var(--accent)" : "var(--text-secondary)",
        background: active ? "var(--accent-subtle)" : undefined,
      }}
    >
      <Icon className="w-5 h-5" aria-hidden="true" />
      <span className="text-[10px] leading-none font-medium">{entry.label}</span>
    </Link>
  );
}

/** Bottom tab bar (below md): five primary tabs plus a "More" sheet for the rest. */
export function MobileTabBar() {
  const pathname = usePathname();
  const [moreOpen, setMoreOpen] = useState(false);
  const primary = RAIL_ITEMS.filter((e) => MOBILE_PRIMARY_IDS.includes(e.id));
  const more = RAIL_ITEMS.filter((e) => !MOBILE_PRIMARY_IDS.includes(e.id));
  const moreActive = more.some((e) => isEntryActive(e, pathname));

  const tab = "flex-1 min-w-0 flex flex-col items-center justify-center gap-1 h-full transition-colors";

  return (
    <>
      {moreOpen && (
        <div
          className="md:hidden fixed inset-0 z-40"
          style={{ background: "color-mix(in srgb, var(--bg-base) 70%, transparent)" }}
          onClick={() => setMoreOpen(false)}
        >
          <div
            role="dialog"
            aria-label="More"
            className="absolute inset-x-0 bottom-14 p-3 border-t rounded-t-xl"
            style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-1 pb-2">
              <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
                More
              </span>
              <button
                type="button"
                title="Close"
                aria-label="Close"
                onClick={() => setMoreOpen(false)}
                className="w-8 h-8 flex items-center justify-center rounded-md hover:bg-[var(--bg-hover)]"
                style={{ color: "var(--text-secondary)" }}
              >
                <X className="w-4 h-4" aria-hidden="true" />
              </button>
            </div>
            <div className="grid grid-cols-3 gap-1">
              {more.map((e) => {
                const Icon = ICONS[e.id];
                const active = isEntryActive(e, pathname);
                return (
                  <Link
                    key={e.id}
                    href={e.href ?? "/"}
                    onClick={() => setMoreOpen(false)}
                    aria-current={active ? "page" : undefined}
                    className="flex flex-col items-center gap-1.5 py-3 rounded-lg hover:bg-[var(--bg-hover)]"
                    style={{ color: active ? "var(--accent)" : "var(--text-secondary)", background: active ? "var(--accent-subtle)" : undefined }}
                  >
                    <Icon className="w-5 h-5" aria-hidden="true" />
                    <span className="text-[11px] font-medium">{e.label}</span>
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
      )}
      <nav
        aria-label="Primary"
        className="md:hidden fixed inset-x-0 bottom-0 z-50 h-14 flex items-stretch border-t"
        style={{ background: "var(--bg-sidebar)", borderColor: "var(--border)" }}
      >
        {primary.map((e) => {
          const Icon = ICONS[e.id];
          const active = isEntryActive(e, pathname);
          return (
            <Link
              key={e.id}
              href={e.href ?? "/"}
              aria-current={active ? "page" : undefined}
              className={tab}
              style={{ color: active ? "var(--accent)" : "var(--text-secondary)" }}
            >
              <Icon className="w-5 h-5" aria-hidden="true" />
              <span className="text-[10px] leading-none font-medium truncate max-w-full">{e.label}</span>
            </Link>
          );
        })}
        <button
          type="button"
          aria-haspopup="dialog"
          aria-expanded={moreOpen}
          onClick={() => setMoreOpen((v) => !v)}
          className={tab}
          style={{ color: moreActive || moreOpen ? "var(--accent)" : "var(--text-secondary)" }}
        >
          <MoreHorizontal className="w-5 h-5" aria-hidden="true" />
          <span className="text-[10px] leading-none font-medium">More</span>
        </button>
      </nav>
    </>
  );
}
