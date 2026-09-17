import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

type ButtonVariant = "default" | "primary" | "ghost" | "danger" | "outline";
type ButtonSize = "xs" | "sm" | "md" | "lg";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  iconLeft?: React.ReactNode;
  iconRight?: React.ReactNode;
}

// ─── Variant / Size maps ──────────────────────────────────────────────────────

const variantClasses: Record<ButtonVariant, string> = {
  default:
    "bg-[#1e1e24] text-[#c8ccd4] border border-[#2a2a32] hover:bg-[#26262e] hover:border-[#3a3a46] active:bg-[#1a1a20]",
  primary:
    "bg-[#3d8ef7] text-white border border-[#3d8ef7] hover:bg-[#5aa0f8] hover:border-[#5aa0f8] active:bg-[#2f7de6] shadow-[0_0_12px_rgba(61,142,247,0.25)]",
  ghost:
    "bg-transparent text-[#8b8fa8] border border-transparent hover:bg-[#1e1e24] hover:text-[#c8ccd4] active:bg-[#17171c]",
  danger:
    "bg-[#2a1418] text-[#f87171] border border-[#3d1a1f] hover:bg-[#371820] hover:border-[#5c2530] active:bg-[#201012]",
  outline:
    "bg-transparent text-[#c8ccd4] border border-[#2a2a32] hover:bg-[#1e1e24] hover:border-[#3a3a46] active:bg-[#17171c]",
};

const sizeClasses: Record<ButtonSize, string> = {
  xs: "h-5 px-2 text-[10px] gap-1 rounded-[3px]",
  sm: "h-6 px-2.5 text-[11px] gap-1.5 rounded-[4px]",
  md: "h-7 px-3 text-[12px] gap-1.5 rounded-[5px]",
  lg: "h-8 px-4 text-[13px] gap-2 rounded-[6px]",
};

// ─── Spinner ──────────────────────────────────────────────────────────────────

const Spinner: React.FC = () => (
  <svg
    className="animate-spin shrink-0"
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
    aria-hidden="true"
  >
    <circle
      cx="6"
      cy="6"
      r="4.5"
      stroke="currentColor"
      strokeOpacity="0.3"
      strokeWidth="1.5"
    />
    <path
      d="M10.5 6a4.5 4.5 0 0 0-4.5-4.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

// ─── Component ────────────────────────────────────────────────────────────────

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = "default",
      size = "md",
      loading = false,
      iconLeft,
      iconRight,
      disabled,
      className,
      children,
      ...props
    },
    ref,
  ) => {
    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={cn(
          "inline-flex items-center justify-center font-medium leading-none",
          "transition-all duration-100 ease-out",
          "select-none whitespace-nowrap",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3d8ef7] focus-visible:ring-offset-1 focus-visible:ring-offset-[#0d0d0f]",
          "disabled:opacity-40 disabled:pointer-events-none",
          variantClasses[variant],
          sizeClasses[size],
          className,
        )}
        {...props}
      >
        {loading ? (
          <Spinner />
        ) : iconLeft ? (
          <span className="shrink-0 flex items-center">{iconLeft}</span>
        ) : null}
        {children && <span>{children}</span>}
        {!loading && iconRight && (
          <span className="shrink-0 flex items-center">{iconRight}</span>
        )}
      </button>
    );
  },
);

Button.displayName = "Button";
