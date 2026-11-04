/**
 * Pure shell logic (no React, no `@/` imports, so the contract tests can load
 * it directly): which nav entry is active, which routes the mode tabs reach,
 * and how the footer's connection / asset text is derived.
 *
 * Every `href` here is an existing route under app/(desktop). A destination
 * that does not exist yet is `href: null` and the UI renders it disabled
 * ("Not available yet") - never a dead link.
 */

export interface NavEntry {
  id: string;
  label: string;
  /** null = no real destination yet; rendered disabled. */
  href: string | null;
  /** Path prefixes (besides href itself) that make this entry active. */
  matches?: string[];
}

export const NOT_AVAILABLE_TITLE = "Not available yet";

/** Top-bar mode tabs, honestly mapped to what exists today. */
export const MODE_TABS: NavEntry[] = [
  { id: "ingest", label: "Capture", href: "/capture", matches: ["/capture", "/reconstruct"] },
  { id: "process", label: "Process", href: "/sessions", matches: ["/sessions"] },
  { id: "explore", label: "Explore", href: "/worlds", matches: ["/worlds", "/evidence"] },
  { id: "compare", label: "Compare", href: null },
  { id: "analyze", label: "Analyze", href: "/analysis", matches: ["/analysis", "/results"] },
  { id: "export", label: "Export", href: "/reports", matches: ["/reports"] },
];

/** Left icon rail. `icon` is resolved in the component to keep this file React-free. */
export const RAIL_ITEMS: NavEntry[] = [
  { id: "home", label: "Home", href: "/" },
  { id: "worlds", label: "3D Studio", href: "/worlds", matches: ["/worlds"] },
  { id: "sites", label: "Sites", href: "/sites", matches: ["/sites"] },
  { id: "ingest", label: "Capture", href: "/capture", matches: ["/capture", "/reconstruct"] },
  { id: "evidence", label: "Evidence", href: "/evidence", matches: ["/evidence"] },
  { id: "sessions", label: "Sessions", href: "/sessions", matches: ["/sessions"] },
  { id: "reports", label: "Reports", href: "/reports", matches: ["/reports"] },
  { id: "settings", label: "Settings", href: "/settings" },
];

/** Mobile bottom tab bar: these five are tabs, the rest go behind "More". */
export const MOBILE_PRIMARY_IDS = ["home", "worlds", "sites", "ingest", "sessions"];

/** Where the top bar's Export button goes (the existing reports/export flow). */
export const EXPORT_HREF = "/reports";

function pathMatches(pathname: string, prefix: string): boolean {
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}

/** Exact-or-prefix match; "/" only matches itself. */
export function isEntryActive(entry: NavEntry, pathname: string): boolean {
  if (entry.href === null) return false;
  if (entry.href === "/") return pathname === "/";
  return [entry.href, ...(entry.matches ?? [])].some((p) => pathMatches(pathname, p));
}

/** The one mode tab to highlight (first match wins), or null on Home/Settings/etc. */
export function activeModeId(pathname: string): string | null {
  return MODE_TABS.find((t) => isEntryActive(t, pathname))?.id ?? null;
}

/** `/worlds/<id>` (not `/worlds/new`) -> the world id, else null. */
export function worldIdFromPath(pathname: string): string | null {
  const m = /^\/worlds\/([^/]+)/.exec(pathname);
  if (!m || m[1] === "new") return null;
  return decodeURIComponent(m[1]);
}

export type ConnectionState = "checking" | "online" | "offline";

/** Footer dot: derived only from a real GET /api/health outcome. */
export function connectionState(health: { ok: boolean } | null): ConnectionState {
  if (health === null) return "checking";
  return health.ok ? "online" : "offline";
}

/** Top-bar status text. Only what the API health check proves; never a fake "Saved". */
export function statusText(state: ConnectionState): string {
  return state === "online" ? "Online" : state === "offline" ? "Offline" : "Checking…";
}

/** Right-hand footer text; null (omitted) unless the API actually reported a count. */
export function evidenceLabel(count: number | null | undefined): string | null {
  if (typeof count !== "number" || !Number.isFinite(count) || count < 0) return null;
  return `${count.toLocaleString("en-US")} evidence ${count === 1 ? "item" : "items"}`;
}

/** "2 hours ago"-style label from an ISO timestamp; null when unparseable. */
export function relativeTime(iso: string | null | undefined, now: number = Date.now()): string | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return null;
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return "just now";
  const units: Array<[number, string]> = [
    [60, "minute"],
    [3600, "hour"],
    [86400, "day"],
    [604800, "week"],
  ];
  let best: [number, string] = units[0];
  for (const u of units) if (s >= u[0]) best = u;
  const n = Math.floor(s / best[0]);
  if (best[1] === "week" && n >= 9) return new Date(t).toISOString().slice(0, 10);
  return `${n} ${best[1]}${n === 1 ? "" : "s"} ago`;
}
