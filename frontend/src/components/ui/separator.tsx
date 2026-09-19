import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SeparatorProps extends React.HTMLAttributes<HTMLDivElement> {
  orientation?: "horizontal" | "vertical";
  /** Optional centered label (horizontal only) */
  label?: string;
  /** Muted variant uses a dimmer color */
  muted?: boolean;
}

// ─── Component ────────────────────────────────────────────────────────────────

export const Separator: React.FC<SeparatorProps> = ({
  orientation = "horizontal",
  label,
  muted = false,
  className,
  ...props
}) => {
  const lineColor = muted ? "#1e1e24" : "#2a2a32";

  if (orientation === "vertical") {
    return (
      <div
        role="separator"
        aria-orientation="vertical"
        className={cn("inline-flex self-stretch shrink-0 w-px", className)}
        style={{ backgroundColor: lineColor }}
        {...props}
      />
    );
  }

  if (label) {
    return (
      <div
        role="separator"
        aria-orientation="horizontal"
        className={cn("flex items-center gap-2 w-full", className)}
        {...props}
      >
        <div className="flex-1 h-px" style={{ backgroundColor: lineColor }} />
        <span className="text-[10px] font-medium text-[#4a4a5a] uppercase tracking-widest leading-none shrink-0">
          {label}
        </span>
        <div className="flex-1 h-px" style={{ backgroundColor: lineColor }} />
      </div>
    );
  }

  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      className={cn("w-full h-px shrink-0", className)}
      style={{ backgroundColor: lineColor }}
      {...props}
    />
  );
};

Separator.displayName = "Separator";
