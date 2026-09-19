import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ScrollAreaProps extends React.HTMLAttributes<HTMLDivElement> {
  maxHeight?: string | number;
  /** Which axes to allow scrolling on (default: vertical) */
  axis?: "x" | "y" | "both";
}

// ─── Scrollbar style injection ────────────────────────────────────────────────

const SCROLLBAR_STYLE = `
.reds-scroll-area {
  scrollbar-width: thin;
  scrollbar-color: #2a2a3a transparent;
}
.reds-scroll-area::-webkit-scrollbar {
  width: 5px;
  height: 5px;
}
.reds-scroll-area::-webkit-scrollbar-track {
  background: transparent;
}
.reds-scroll-area::-webkit-scrollbar-thumb {
  background-color: #2a2a3a;
  border-radius: 99px;
}
.reds-scroll-area::-webkit-scrollbar-thumb:hover {
  background-color: #3a3a4a;
}
.reds-scroll-area::-webkit-scrollbar-corner {
  background: transparent;
}
`;

let scrollStyleInjected = false;
function injectScrollStyle() {
  if (scrollStyleInjected || typeof document === "undefined") return;
  const el = document.createElement("style");
  el.id = "reds-scroll-style";
  el.textContent = SCROLLBAR_STYLE;
  document.head.appendChild(el);
  scrollStyleInjected = true;
}

// ─── Component ────────────────────────────────────────────────────────────────

export const ScrollArea = React.forwardRef<HTMLDivElement, ScrollAreaProps>(
  ({ maxHeight, axis = "y", className, children, style, ...props }, ref) => {
    React.useEffect(() => { injectScrollStyle(); }, []);

    const overflowStyle: React.CSSProperties =
      axis === "both"
        ? { overflow: "auto" }
        : axis === "x"
          ? { overflowX: "auto", overflowY: "hidden" }
          : { overflowY: "auto", overflowX: "hidden" };

    const maxHeightStyle: React.CSSProperties =
      maxHeight !== undefined
        ? { maxHeight: typeof maxHeight === "number" ? `${maxHeight}px` : maxHeight }
        : {};

    return (
      <div
        ref={ref}
        className={cn("reds-scroll-area w-full", className)}
        style={{ ...overflowStyle, ...maxHeightStyle, ...style }}
        {...props}
      >
        {children}
      </div>
    );
  },
);

ScrollArea.displayName = "ScrollArea";
