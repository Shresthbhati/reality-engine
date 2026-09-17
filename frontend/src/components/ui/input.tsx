import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size" | "prefix"> {
  /** Icon or text shown before the input value */
  prefix?: React.ReactNode;
  /** Icon or text shown after the input value */
  suffix?: React.ReactNode;
  error?: boolean;
  errorMessage?: string;
  label?: string;
  size?: "sm" | "md" | "lg";
}

// ─── Size map ─────────────────────────────────────────────────────────────────

const sizeMap = {
  sm: { wrapper: "h-6", text: "text-[11px]", px: "px-2" },
  md: { wrapper: "h-7", text: "text-[12px]", px: "px-2.5" },
  lg: { wrapper: "h-8", text: "text-[13px]", px: "px-3" },
};

// ─── Component ────────────────────────────────────────────────────────────────

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      prefix,
      suffix,
      error = false,
      errorMessage,
      label,
      size = "md",
      disabled,
      className,
      ...props
    },
    ref,
  ) => {
    const s = sizeMap[size];

    return (
      <div className="flex flex-col gap-1 w-full">
        {label && (
          <label className="text-[11px] font-medium text-[#8b8fa8] leading-none">
            {label}
          </label>
        )}

        <div
          className={cn(
            "relative flex items-center w-full rounded-[5px]",
            "bg-[#121215] border transition-colors duration-100",
            error
              ? "border-[#5c2530] focus-within:border-[#f87171]"
              : "border-[#2a2a32] focus-within:border-[#3d8ef7]",
            disabled && "opacity-40 pointer-events-none",
            s.wrapper,
          )}
        >
          {prefix && (
            <span className="flex items-center shrink-0 pl-2 text-[#4a4a5a]">
              {prefix}
            </span>
          )}

          <input
            ref={ref}
            disabled={disabled}
            className={cn(
              "flex-1 min-w-0 bg-transparent outline-none border-none",
              "text-[#c8ccd4] placeholder:text-[#4a4a5a]",
              "font-mono leading-none",
              s.text,
              s.px,
              "[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none",
              className,
            )}
            {...props}
          />

          {suffix && (
            <span className="flex items-center shrink-0 pr-2 text-[#4a4a5a]">
              {suffix}
            </span>
          )}
        </div>

        {error && errorMessage && (
          <p className="text-[10px] text-[#f87171] leading-none">{errorMessage}</p>
        )}
      </div>
    );
  },
);

Input.displayName = "Input";
