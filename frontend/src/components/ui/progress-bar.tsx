import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ProgressBarProps {
  /** 0-100 */
  value: number;
  label?: string;
  showValue?: boolean;
  /** Override fill color */
  color?: string;
  /** Shows animated stripes for in-progress states */
  striped?: boolean;
  /** Height in px (default 4) */
  height?: number;
  className?: string;
  status?: "default" | "success" | "warning" | "error";
}

// ─── Status -> color ──────────────────────────────────────────────────────────

const statusColors: Record<NonNullable<ProgressBarProps["status"]>, string> = {
  default: "#3d8ef7",
  success: "#4ade80",
  warning: "#fbbf24",
  error:   "#f87171",
};

// ─── Stripe animation ─────────────────────────────────────────────────────────

const STRIPE_STYLE = `
@keyframes reds-stripe {
  from { background-position: 0 0; }
  to   { background-position: 20px 0; }
}
.reds-pb-striped {
  background-image: repeating-linear-gradient(
    45deg,
    transparent,
    transparent 4px,
    rgba(255,255,255,0.08) 4px,
    rgba(255,255,255,0.08) 8px
  );
  background-size: 20px 20px;
  animation: reds-stripe 700ms linear infinite;
}
`;

let stripeInjected = false;
function injectStripeStyle() {
  if (stripeInjected || typeof document === "undefined") return;
  const el = document.createElement("style");
  el.id = "reds-pb-stripe-style";
  el.textContent = STRIPE_STYLE;
  document.head.appendChild(el);
  stripeInjected = true;
}

// ─── Component ────────────────────────────────────────────────────────────────

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  label,
  showValue = false,
  color,
  striped = false,
  height = 4,
  className,
  status = "default",
}) => {
  React.useEffect(() => {
    if (striped) injectStripeStyle();
  }, [striped]);

  const clampedValue = Math.min(100, Math.max(0, value));
  const fillColor = color ?? statusColors[status];

  return (
    <div className={cn("flex flex-col gap-1 w-full", className)}>
      {(label || showValue) && (
        <div className="flex items-center justify-between">
          {label && (
            <span className="text-[11px] font-medium text-[#8b8fa8] leading-none">
              {label}
            </span>
          )}
          {showValue && (
            <span className="text-[11px] font-mono text-[#c8ccd4] leading-none tabular-nums">
              {Math.round(clampedValue)}%
            </span>
          )}
        </div>
      )}

      <div
        className="relative w-full overflow-hidden rounded-full bg-[#1e1e24] border border-[#2a2a32]"
        style={{ height }}
        role="progressbar"
        aria-valuenow={clampedValue}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
      >
        <div
          className={cn(
            "absolute inset-y-0 left-0 rounded-full transition-[width] duration-300 ease-out",
            striped && "reds-pb-striped",
          )}
          style={{
            width: `${clampedValue}%`,
            backgroundColor: fillColor,
            boxShadow: clampedValue > 0 ? `0 0 6px ${fillColor}60` : undefined,
          }}
        />
      </div>
    </div>
  );
};

ProgressBar.displayName = "ProgressBar";
