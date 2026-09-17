import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "onChange" | "size"> {
  value?: string;
  onChange?: (value: string) => void;
  options: SelectOption[];
  placeholder?: string;
  label?: string;
  error?: boolean;
  errorMessage?: string;
  size?: "sm" | "md" | "lg";
}

// ─── Size map ─────────────────────────────────────────────────────────────────

const sizeMap = {
  sm: { wrapper: "h-6", text: "text-[11px]", pl: "pl-2",   pr: "pr-7" },
  md: { wrapper: "h-7", text: "text-[12px]", pl: "pl-2.5", pr: "pr-8" },
  lg: { wrapper: "h-8", text: "text-[13px]", pl: "pl-3",   pr: "pr-9" },
};

// ─── Chevron icon ─────────────────────────────────────────────────────────────

const ChevronDown: React.FC<{ size?: number }> = ({ size = 11 }) => (
  <svg width={size} height={size} viewBox="0 0 12 12" fill="none" aria-hidden="true">
    <path
      d="M2.5 4.5L6 8l3.5-3.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

// ─── Component ────────────────────────────────────────────────────────────────

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      value,
      onChange,
      options,
      placeholder,
      label,
      error = false,
      errorMessage,
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
          <select
            ref={ref}
            value={value}
            onChange={(e) => onChange?.(e.target.value)}
            disabled={disabled}
            className={cn(
              "w-full h-full bg-transparent border-none outline-none appearance-none cursor-pointer",
              "text-[#c8ccd4] font-mono leading-none",
              !value && "text-[#4a4a5a]",
              s.text, s.pl, s.pr,
              className,
            )}
            style={{ colorScheme: "dark" }}
            {...props}
          >
            {placeholder && (
              <option value="" disabled hidden>{placeholder}</option>
            )}
            {options.map((opt) => (
              <option
                key={opt.value}
                value={opt.value}
                disabled={opt.disabled}
                style={{ background: "#1a1a21", color: "#c8ccd4" }}
              >
                {opt.label}
              </option>
            ))}
          </select>

          <span className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-[#4a4a5a]">
            <ChevronDown />
          </span>
        </div>

        {error && errorMessage && (
          <p className="text-[10px] text-[#f87171] leading-none">{errorMessage}</p>
        )}
      </div>
    );
  },
);

Select.displayName = "Select";
