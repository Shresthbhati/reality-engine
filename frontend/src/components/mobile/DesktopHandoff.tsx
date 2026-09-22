import Link from "next/link";
import { ExternalLink } from "lucide-react";

/**
 * Canonical "continue on desktop" prompt — spec §71. Use on any mobile
 * screen whose deep investigation belongs on the desktop workstation
 * (World workspace, Analysis config, Report editing, advanced Evidence).
 * `href` is the equivalent desktop route for the same entity.
 */
export default function DesktopHandoff({ href, label = "Continue on Desktop" }: { href: string; label?: string }) {
  return (
    <Link
      href={href}
      className="flex items-center justify-center gap-2 h-11 rounded-md text-sm font-medium"
      style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}
    >
      <ExternalLink className="w-4 h-4" />
      {label}
    </Link>
  );
}
