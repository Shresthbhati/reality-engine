"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Camera, List, MoreHorizontal } from "lucide-react";

const TABS = [
  { href: "/m/camera", label: "Capture", icon: Camera },
  { href: "/m/sessions", label: "Sessions", icon: List },
  { href: "/m/more", label: "More", icon: MoreHorizontal },
] as const;

export default function MobileShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden" style={{ background: "var(--bg-base)" }}>
      <header
        className="h-12 shrink-0 flex items-center justify-between px-4 border-b"
        style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
      >
        <Link href="/m" className="flex items-center gap-2">
          <div
            className="w-6 h-6 rounded-md flex items-center justify-center"
            style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
          >
            <Camera className="w-3.5 h-3.5" />
          </div>
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Reality Engine
          </span>
        </Link>
        <span className="w-2 h-2 rounded-full" style={{ background: "var(--success)" }} title="Online" />
      </header>

      <main className="flex-1 min-h-0 overflow-y-auto">{children}</main>

      <nav
        aria-label="Primary"
        className="h-16 shrink-0 flex items-stretch border-t"
        style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
      >
        {TABS.map((tab) => {
          const active = pathname.startsWith(tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={active ? "page" : undefined}
              className="flex-1 flex flex-col items-center justify-center gap-1 text-xs font-medium"
              style={{ color: active ? "var(--accent)" : "var(--text-tertiary)" }}
            >
              <tab.icon className="w-5 h-5" />
              {tab.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
