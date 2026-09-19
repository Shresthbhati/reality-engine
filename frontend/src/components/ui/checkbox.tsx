import * as React from "react";
import { cn } from "@/lib/utils";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface CheckboxProps {
  checked?: boolean;
  onChange?: (checked: boolean) => void;
  label?: React.ReactNode;
  /** Shows a dash instead of a checkmark */
  indeterminate?: boolean;
  disabled?: boolean;
  id?: string;
  className?: string;
  size?: "sm" | "md";
}

// ─── Icons ────────────────────────────────────────────────────────────────────

const CheckIcon: React.FC<{ size: number }> = ({ size }) => (
  <svg width={size} height={size} viewBox="0 0 12 12" fill="none" aria-hidden="true">
    <path
      d="M2.5 6L5 8.5l4.5-5"
      stroke="white"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const DashIcon: React.FC<{ size: number }> = ({ size }) => (
  <svg width={size} height={size} viewBox="0 0 12 12" fill="none" aria-hidden="true">
    <path d="M2.5 6h7" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

// ─── Component ────────────────────────────────────────────────────────────────

export const Checkbox: React.FC<CheckboxProps> = ({
  checked = false,
  onChange,
  label,
  indeterminate = false,
  disabled = false,
  id,
  className,
  size = "md",
}) => {
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    if (inputRef.current) {
      inputRef.current.indeterminate = indeterminate && !checked;
    }
  }, [indeterminate, checked]);

  const boxSize  = size === "sm" ? 13 : 15;
  const iconSize = size === "sm" ? 10 : 12;
  const textSize = size === "sm" ? "text-[11px]" : "text-[12px]";

  const isActive = checked || indeterminate;

  const uid = React.useId();
  const inputId = id ?? uid;

  return (
    <label
      htmlFor={inputId}
      className={cn(
        "inline-flex items-center gap-2 cursor-pointer select-none",
        disabled && "opacity-40 pointer-events-none",
        className,
      )}
    >
      <input
        ref={inputRef}
        type="checkbox"
        id={inputId}
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.checked)}
        className="sr-only"
        aria-checked={indeterminate ? "mixed" : checked}
      />

      <span
        className={cn(
          "inline-flex items-center justify-center shrink-0 rounded-[3px]",
          "border transition-all duration-100",
          isActive
            ? "bg-[#3d8ef7] border-[#3d8ef7] shadow-[0_0_8px_rgba(61,142,247,0.35)]"
            : "bg-[#121215] border-[#2a2a32]",
        )}
        style={{ width: boxSize, height: boxSize }}
        aria-hidden="true"
      >
        {checked && !indeterminate && <CheckIcon size={iconSize} />}
        {indeterminate && <DashIcon size={iconSize} />}
      </span>

      {label && (
        <span className={cn("font-medium text-[#c8ccd4] leading-none", textSize)}>
          {label}
        </span>
      )}
    </label>
  );
};

Checkbox.displayName = "Checkbox";
