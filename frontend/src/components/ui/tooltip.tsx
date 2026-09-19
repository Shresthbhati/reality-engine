import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface TooltipProps {
  children: React.ReactNode;
  content: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
  /** Delay before the tooltip appears in ms */
  delayMs?: number;
}

// ─── Position styles ──────────────────────────────────────────────────────────

const positionStyles: Record<
  NonNullable<TooltipProps["side"]>,
  { wrapper: string; arrow: string }
> = {
  top: {
    wrapper: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    arrow:   "top-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-b-transparent border-t-[#2a2a32]",
  },
  bottom: {
    wrapper: "top-full left-1/2 -translate-x-1/2 mt-2",
    arrow:   "bottom-full left-1/2 -translate-x-1/2 border-l-transparent border-r-transparent border-t-transparent border-b-[#2a2a32]",
  },
  left: {
    wrapper: "right-full top-1/2 -translate-y-1/2 mr-2",
    arrow:   "left-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-r-transparent border-l-[#2a2a32]",
  },
  right: {
    wrapper: "left-full top-1/2 -translate-y-1/2 ml-2",
    arrow:   "right-full top-1/2 -translate-y-1/2 border-t-transparent border-b-transparent border-l-transparent border-r-[#2a2a32]",
  },
};

// ─── Inline style injection ───────────────────────────────────────────────────

const TOOLTIP_STYLE = `
.reds-tooltip-trigger { position: relative; display: inline-flex; }
.reds-tooltip-trigger .reds-tooltip-box {
  visibility: hidden;
  opacity: 0;
  pointer-events: none;
  transition: opacity 120ms ease, visibility 120ms ease;
}
.reds-tooltip-trigger:hover .reds-tooltip-box,
.reds-tooltip-trigger:focus-within .reds-tooltip-box {
  visibility: visible;
  opacity: 1;
  pointer-events: auto;
}
`;

let styleInjected = false;
function injectStyle() {
  if (styleInjected || typeof document === "undefined") return;
  const el = document.createElement("style");
  el.id = "reds-tooltip-style";
  el.textContent = TOOLTIP_STYLE;
  document.head.appendChild(el);
  styleInjected = true;
}

// ─── Component ────────────────────────────────────────────────────────────────

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  side = "top",
  className,
  delayMs = 300,
}) => {
  React.useEffect(() => { injectStyle(); }, []);

  const pos = positionStyles[side];

  return (
    <span className="reds-tooltip-trigger">
      {children}
      <span
        className={cn("reds-tooltip-box absolute z-50 pointer-events-none", pos.wrapper)}
        style={{ transitionDelay: `${delayMs}ms, ${delayMs}ms` }}
      >
        <span
          className={cn(
            "relative flex items-center whitespace-nowrap",
            "rounded-[5px] px-2 py-1",
            "text-[11px] font-medium leading-none text-[#c8ccd4]",
            "bg-[#1a1a21] border border-[#2a2a32]",
            "shadow-[0_4px_16px_rgba(0,0,0,0.5)]",
            className,
          )}
        >
          {content}
        </span>
        <span
          className={cn("absolute w-0 h-0 border-[4px] border-solid", pos.arrow)}
        />
      </span>
    </span>
  );
};

Tooltip.displayName = "Tooltip";
