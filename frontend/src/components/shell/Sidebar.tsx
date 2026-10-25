"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  CloudUpload,
  ShieldCheck,
  Globe,
  Camera,
  FileText,
  Settings,
  HelpCircle,
  type LucideIcon,
} from "lucide-react";
import { RAIL_ITEMS, NOT_AVAILABLE_TITLE, isEntryActive } from "@/lib/shell/shell-logic";

const ICONS: Record<string, LucideIcon> = {
  home: Home,
  ingest: CloudUpload,
  evidence: ShieldCheck,
  worlds: Globe,
  sessions: Camera,
  reports: FileText,
  settings: Settings,
  help: HelpCircle,
};

/** Left icon rail (icon + small label). Settings and Help sit at the bottom, as in the references. */
export default function Sidebar() {
  const pathname = usePathname();
  const main = RAIL_ITEMS.filter((e) => e.id !== "settings" && e.id !== "help");
  const bottom = RAIL_ITEMS.filter((e) => e.id === "settings" || e.id === "help");

  return (
    <aside
      className="w-[72px] shrink-0 h-full flex flex-col border-r"
      style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
    >
      <nav aria-label="Primary" className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-1 px-1.5 py-3">
        {main.map((e) => (
          <RailItem key={e.id} entry={e} active={isEntryActive(e, pathname)} />
        ))}
      </nav>
      <nav aria-label="Secondary" className="flex flex-col gap-1 px-1.5 py-3 border-t" style={{ borderColor: "var(--border-subtle)" }}>
        {bottom.map((e) => (
          <RailItem key={e.id} entry={e} active={isEntryActive(e, pathname)} />
        ))}
      </nav>
    </aside>
  );
}

function RailItem({ entry, active }: { entry: (typeof RAIL_ITEMS)[number]; active: boolean }) {
  const Icon = ICONS[entry.id];
  const body = (
    <>
      <Icon className="w-5 h-5" aria-hidden="true" />
      <span className="text-[11px] leading-none font-medium">{entry.label}</span>
    </>
  );
  const base = "flex flex-col items-center justify-center gap-1.5 h-14 rounded-lg border transition-colors";

  if (entry.href === null) {
    return (
      <button
        type="button"
        disabled
        title={NOT_AVAILABLE_TITLE}
        aria-label={`${entry.label} (${NOT_AVAILABLE_TITLE.toLowerCase()})`}
        className={`${base} cursor-not-allowed opacity-50`}
        style={{ color: "var(--text-secondary)", borderColor: "transparent" }}
      >
        {body}
      </button>
    );
  }

  return (
    <Link
      href={entry.href}
      aria-current={active ? "page" : undefined}
      className={`${base} hover:bg-[var(--bg-hover)]`}
      style={{
        color: active ? "var(--accent)" : "var(--text-secondary)",
        background: active ? "var(--accent-subtle)" : undefined,
        borderColor: active ? "var(--accent-border)" : "transparent",
      }}
    >
      {body}
    </Link>
  );
}
