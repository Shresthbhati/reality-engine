"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSyncExternalStore } from "react";
import { Camera, Home, List, MoreHorizontal, ShieldCheck, WifiOff, type LucideIcon } from "lucide-react";

const TABS: { href: string; label: string; icon: LucideIcon; exact?: boolean }[] = [
  { href: "/m", label: "Home", icon: Home, exact: true },
  { href: "/m/camera", label: "Capture", icon: Camera },
  { href: "/m/evidence", label: "Evidence", icon: ShieldCheck },
  { href: "/m/sessions", label: "Sessions", icon: List },
  { href: "/m/more", label: "More", icon: MoreHorizontal },
];

function subscribeOnline(cb: () => void) {
  window.addEventListener("online", cb);
  window.addEventListener("offline", cb);
  return () => {
    window.removeEventListener("online", cb);
    window.removeEventListener("offline", cb);
  };
}

/** The DEVICE's network state (navigator.onLine); it says nothing about whether the Reality Engine API is up. */
function useDeviceOnline(): boolean {
  return useSyncExternalStore(
    subscribeOnline,
    () => navigator.onLine,
    () => true,
  );
}

export default function MobileShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const online = useDeviceOnline();

  return (
    <div className="flex h-dvh w-screen flex-col overflow-hidden" style={{ background: "var(--bg-base)" }}>
      <header
        className="flex shrink-0 items-center justify-between border-b px-4"
        style={{
          background: "var(--bg-surface)",
          borderColor: "var(--border)",
          paddingTop: "env(safe-area-inset-top)",
          paddingLeft: "max(1rem, env(safe-area-inset-left))",
          paddingRight: "max(1rem, env(safe-area-inset-right))",
          minHeight: "calc(3rem + env(safe-area-inset-top))",
        }}
      >
        <Link href="/m" aria-label="Reality Engine home" className="flex min-h-[44px] items-center gap-2">
          <div
            className="flex h-7 w-7 items-center justify-center rounded-md text-[11px] font-bold"
            style={{ background: "var(--accent-subtle)", border: "1px solid var(--accent-border)", color: "var(--accent)" }}
            aria-hidden
          >
            RE
          </div>
          <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            Reality Engine
          </span>
        </Link>
        <span
          role="status"
          className="flex items-center gap-1.5 text-xs"
          style={{ color: online ? "var(--text-tertiary)" : "var(--warning)" }}
        >
          {online ? (
            <>
              <span className="h-2 w-2 rounded-full" style={{ background: "var(--success)" }} aria-hidden />
              <span className="sr-only">Device network online</span>
              <span aria-hidden>Network</span>
            </>
          ) : (
            <>
              <WifiOff className="h-3.5 w-3.5" aria-hidden />
              Offline
            </>
          )}
        </span>
      </header>

      {!online && (
        <p
          role="alert"
          className="shrink-0 px-4 py-2 text-xs"
          style={{ background: "var(--warning-subtle)", color: "var(--warning)", borderBottom: "1px solid var(--border)" }}
        >
          This device is offline. Photos stay in the upload queue and can be retried once you are back online.
        </p>
      )}

      <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>

      <nav
        aria-label="Primary"
        className="flex shrink-0 items-stretch border-t"
        style={{
          background: "var(--bg-surface)",
          borderColor: "var(--border)",
          paddingBottom: "env(safe-area-inset-bottom)",
          paddingLeft: "env(safe-area-inset-left)",
          paddingRight: "env(safe-area-inset-right)",
        }}
      >
        {TABS.map((tab) => {
          const active = tab.exact ? pathname === tab.href : pathname.startsWith(tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              aria-current={active ? "page" : undefined}
              className="flex min-h-[56px] min-w-[44px] flex-1 flex-col items-center justify-center gap-1 text-[11px] font-medium"
              style={{ color: active ? "var(--accent)" : "var(--text-tertiary)" }}
            >
              <tab.icon className="h-5 w-5" aria-hidden />
              {tab.label}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
