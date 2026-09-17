import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

type BadgeVariant =
  | "default"
  | "success"
  | "warning"
  | "error"
  | "info"
  | "accent";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  dot?: boolean;
  size?: "sm" | "md";
}

// ─── Variant map ──────────────────────────────────────────────────────────────

const variantClasses: Record<BadgeVariant, string> = {
  default: "bg-[#1e1e24] text-[#8b8fa8] border border-[#2a2a32]",
  success: "bg-[#0d2218] text-[#4ade80] border border-[#133824]",
  warning: "bg-[#251c0d] text-[#fbbf24] border border-[#3d2e10]",
  error:   "bg-[#2a1418] text-[#f87171] border border-[#3d1a1f]",
  info:    "bg-[#0e1a2e] text-[#60a5fa] border border-[#152440]",
  accent:  "bg-[#0f1e3d] text-[#3d8ef7] border border-[#1a3060]",
};

const dotColorClasses: Record<BadgeVariant, string> = {
  default: "bg-[#8b8fa8]",
  success: "bg-[#4ade80]",
  warning: "bg-[#fbbf24]",
  error:   "bg-[#f87171]",
  info:    "bg-[#60a5fa]",
  accent:  "bg-[#3d8ef7]",
};

const sizeClasses = {
  sm: "h-4 px-1.5 text-[9px] gap-1 rounded-[3px]",
  md: "h-5 px-2 text-[10px] gap-1 rounded-[4px]",
};

// ─── Component ────────────────────────────────────────────────────────────────

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ variant = "default", dot = false, size = "md", className, children, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center font-medium leading-none tracking-wide uppercase",
          variantClasses[variant],
          sizeClasses[size],
          className,
        )}
        {...props}
      >
        {dot && (
          <span
            className={cn("inline-block rounded-full shrink-0", dotColorClasses[variant])}
            style={{ width: 5, height: 5 }}
          />
        )}
        {children}
      </span>
    );
  },
);

Badge.displayName = "Badge";
