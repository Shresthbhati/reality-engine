"use client";

import {
  Loader2,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Clock,
} from "lucide-react";
import type { ReconstructionState, ReconstructionStatus } from "@/lib/interior/status";

const PRESENTATION: Record<
  ReconstructionState,
  { icon: typeof Clock; border: string; bg: string; text: string }
> = {
  processing: {
    icon: Loader2,
    border: "border-[#00e5ff]/40",
    bg: "bg-[#06232b]",
    text: "text-[#00e5ff]",
  },
  partial: {
    icon: AlertTriangle,
    border: "border-[#e67e22]/50",
    bg: "bg-[#2a1f10]",
    text: "text-[#e67e22]",
  },
  complete: {
    icon: CheckCircle2,
    border: "border-[#2ecc71]/50",
    bg: "bg-[#12281a]",
    text: "text-[#2ecc71]",
  },
  failed: {
    icon: XCircle,
    border: "border-[#e74c3c]/50",
    bg: "bg-[#2a1315]",
    text: "text-[#ff6b6b]",
  },
  insufficient_evidence: {
    icon: HelpCircle,
    border: "border-neutral-600",
    bg: "bg-neutral-900",
    text: "text-neutral-300",
  },
  not_reconstructed: {
    icon: Clock,
    border: "border-neutral-700",
    bg: "bg-neutral-900",
    text: "text-neutral-400",
  },
};

/**
 * The reconstruction state, stated plainly.
 *
 * This is the component that enforces "never display a completed world when
 * the backend only has partial output": a version missing rooms, walls,
 * floors, or ceilings renders as PARTIAL or INSUFFICIENT_EVIDENCE with the
 * missing classes named -- never as a green "complete".
 */
export default function ReconstructionStatusBanner({
  status,
}: {
  status: ReconstructionStatus;
}) {
  const p = PRESENTATION[status.state];
  const Icon = p.icon;
  const spinning = status.state === "processing";

  return (
    <div
      role="status"
      data-state={status.state}
      data-testid="reconstruction-status"
      className={`rounded-lg border ${p.border} ${p.bg} px-3 py-2.5`}
    >
      <div className="flex items-start gap-2.5">
        <Icon
          className={`w-4 h-4 shrink-0 mt-0.5 ${p.text} ${spinning ? "animate-spin" : ""}`}
        />
        <div className="min-w-0 flex-1">
          <div className={`text-xs font-semibold ${p.text}`}>{status.title}</div>
          <div className="text-[11px] text-neutral-400 mt-0.5 leading-relaxed">
            {status.detail}
          </div>

          {status.missing.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {status.missing.map((k) => (
                <span
                  key={k}
                  className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/40 text-neutral-400"
                >
                  {k}
                </span>
              ))}
            </div>
          )}

          {status.stages.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {status.stages.map((s) => (
                <span
                  key={s.name}
                  title={s.note}
                  className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    s.status === "ran"
                      ? "bg-[#2ecc71]/15 text-[#2ecc71]"
                      : s.status === "failed"
                        ? "bg-[#e74c3c]/20 text-[#ff6b6b]"
                        : s.status === "skipped"
                          ? "bg-neutral-800 text-neutral-500"
                          : "bg-neutral-800 text-neutral-400"
                  }`}
                >
                  {s.name}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}