import Link from "next/link";
import { ShieldCheck, Globe, Users, Bell, UserCircle, HelpCircle, ChevronRight, type LucideIcon } from "lucide-react";

export default function MobileMorePage() {
  return (
    <div className="flex flex-col gap-4 p-4">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        More
      </h1>

      {/* Spec §69: exactly these rows — no giant desktop navigation. */}
      <div className="rounded-md overflow-hidden" style={{ border: "1px solid var(--border)" }}>
        <MenuRow href="/m/evidence" icon={ShieldCheck} label="Evidence" />
        <MenuRow href="/m/worlds" icon={Globe} label="Worlds" />
        <MenuRow href="/m/team" icon={Users} label="Team" />
        <MenuRow href="/m" icon={Bell} label="Notifications" />
        <MenuRow icon={UserCircle} label="Account" disabled />
        <MenuRow icon={HelpCircle} label="Help" disabled last />
      </div>
    </div>
  );
}

function MenuRow({
  href,
  icon: Icon,
  label,
  disabled,
  last,
}: {
  href?: string;
  icon: LucideIcon;
  label: string;
  disabled?: boolean;
  last?: boolean;
}) {
  const content = (
    <>
      <Icon className="w-5 h-5 shrink-0" style={{ color: disabled ? "var(--text-tertiary)" : "var(--text-secondary)" }} />
      <span className="flex-1 text-sm font-medium" style={{ color: disabled ? "var(--text-tertiary)" : "var(--text-primary)" }}>
        {label}
      </span>
      {disabled ? (
        <span className="text-xs" style={{ color: "var(--text-tertiary)" }}>
          Coming soon
        </span>
      ) : (
        <ChevronRight className="w-4 h-4 shrink-0" style={{ color: "var(--text-tertiary)" }} />
      )}
    </>
  );

  const rowClass = "flex items-center gap-3 min-h-[52px] px-4";
  const rowStyle = {
    background: "var(--bg-surface)",
    borderBottom: last ? "none" : "1px solid var(--border)",
  };

  if (disabled || !href) {
    return (
      <div className={rowClass} style={{ ...rowStyle, opacity: 0.55 }}>
        {content}
      </div>
    );
  }

  return (
    <Link href={href} className={rowClass} style={rowStyle}>
      {content}
    </Link>
  );
}
