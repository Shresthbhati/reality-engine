"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Globe,
  Camera,
  ShieldCheck,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";

const PRIMARY_NAV = [
  { href: "/", label: "Home", icon: Home },
  { href: "/worlds", label: "Worlds", icon: Globe },
  { href: "/sessions", label: "Sessions", icon: Camera },
  { href: "/evidence", label: "Evidence", icon: ShieldCheck },
] as const;

const SECONDARY_NAV = [
  { href: "/settings", label: "Settings", icon: Settings },
] as const;

export default function Sidebar({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "flex flex-col shrink-0 h-full border-r transition-[width] duration-150",
        collapsed ? "w-14" : "w-56",
      )}
      style={{ background: "var(--bg-surface)", borderColor: "var(--border)" }}
    >
      <nav aria-label="Primary" className="flex-1 flex flex-col gap-0.5 px-2 py-3">
        {PRIMARY_NAV.map((item) => (
          <SidebarLink key={item.href} item={item} collapsed={collapsed} active={pathname === item.href} />
        ))}
      </nav>

      <div className="px-2 pb-3 flex flex-col gap-0.5 border-t pt-3" style={{ borderColor: "var(--border-subtle)" }}>
        {SECONDARY_NAV.map((item) => (
          <SidebarLink key={item.href} item={item} collapsed={collapsed} active={pathname === item.href} />
        ))}

        <button
          type="button"
          onClick={onToggle}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="flex items-center gap-2.5 px-2.5 h-9 rounded-md text-sm mt-1 transition-colors"
          style={{ color: "var(--text-tertiary)" }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          {collapsed ? <PanelLeftOpen className="w-4 h-4 shrink-0" /> : <PanelLeftClose className="w-4 h-4 shrink-0" />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}

function SidebarLink({
  item,
  collapsed,
  active,
}: {
  item: { href: string; label: string; icon: typeof Home };
  collapsed: boolean;
  active: boolean;
}) {
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      title={collapsed ? item.label : undefined}
      aria-current={active ? "page" : undefined}
      className="flex items-center gap-2.5 px-2.5 h-9 rounded-md text-sm font-medium transition-colors"
      style={{
        color: active ? "var(--accent)" : "var(--text-secondary)",
        background: active ? "var(--accent-subtle)" : "transparent",
      }}
      onMouseEnter={(e) => {
        if (!active) e.currentTarget.style.background = "var(--bg-hover)";
      }}
      onMouseLeave={(e) => {
        if (!active) e.currentTarget.style.background = "transparent";
      }}
    >
      <Icon className="w-4 h-4 shrink-0" />
      {!collapsed && <span className="truncate">{item.label}</span>}
    </Link>
  );
}
