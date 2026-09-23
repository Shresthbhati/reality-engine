import Link from "next/link";
import { ExternalLink, Globe, Layers, ShieldCheck } from "lucide-react";

interface DesktopHandoffProps {
  href: string;
  label?: string;
  /** Entity identity this handoff is continuing — shown so the desktop workstation
   * opens on the same World/Session/Version the mobile screen was looking at. */
  worldName?: string | null;
  sessionName?: string | null;
  versionLabel?: string | null;
  /** e.g. "3 evidence items awaiting review" — what's left to do on desktop. */
  pendingWork?: string | null;
}

/**
 * Canonical "continue on desktop" prompt — spec §71. Use on any mobile
 * screen whose deep investigation belongs on the desktop workstation
 * (World workspace, Analysis config, Report editing, advanced Evidence).
 * `href` is the equivalent desktop route for the same entity.
 *
 * Carries identity context (World / Session / Version / pending work) so the
 * handoff answers "what am I continuing?" instead of being a bare QR/link.
 */
export default function DesktopHandoff({
  href,
  label = "Continue on Desktop",
  worldName,
  sessionName,
  versionLabel,
  pendingWork,
}: DesktopHandoffProps) {
  const context = [
    worldName ? { icon: Globe, text: worldName } : null,
    sessionName ? { icon: Layers, text: sessionName } : null,
    versionLabel ? { icon: Layers, text: `Version ${versionLabel}` } : null,
  ].filter((c): c is { icon: typeof Globe; text: string } => c !== null);

  return (
    <div className="flex flex-col gap-2">
      {(context.length > 0 || pendingWork) && (
        <div
          className="flex flex-col gap-1.5 px-3 py-2.5 rounded-md text-xs"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
        >
          {context.map((c, i) => (
            <div key={i} className="flex items-center gap-1.5" style={{ color: "var(--text-secondary)" }}>
              <c.icon className="w-3.5 h-3.5 shrink-0" />
              <span className="truncate">{c.text}</span>
            </div>
          ))}
          {pendingWork && (
            <div className="flex items-center gap-1.5" style={{ color: "var(--warning, var(--text-tertiary))" }}>
              <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
              <span className="truncate">{pendingWork}</span>
            </div>
          )}
        </div>
      )}
      <Link
        href={href}
        className="flex items-center justify-center gap-2 h-11 rounded-md text-sm font-medium"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
      >
        <ExternalLink className="w-4 h-4" />
        {label}
      </Link>
    </div>
  );
}
