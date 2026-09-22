import { Users } from "lucide-react";
import DesktopHandoff from "@/components/mobile/DesktopHandoff";

// No TeamRow type or TEAM data source exists in lib/types.ts or lib/data.ts —
// this is necessarily an honest empty/placeholder screen (spec §70).
export default function MobileTeamPage() {
  return (
    <div className="flex flex-col gap-5 p-4">
      <h1 className="text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
        Team
      </h1>

      <div className="flex flex-col items-center gap-2 py-16 px-4 text-center">
        <Users className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
        <p className="text-sm" style={{ color: "var(--text-tertiary)" }}>
          No team members yet.
        </p>
      </div>

      <p className="text-xs text-center" style={{ color: "var(--text-tertiary)" }}>
        Team administration is available on desktop.
      </p>

      <DesktopHandoff href="/team" />
    </div>
  );
}
