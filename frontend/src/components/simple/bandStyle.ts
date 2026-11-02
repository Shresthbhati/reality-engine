import type { Band } from "@/lib/api";

/** Colour and shape for a quality band. A shape accompanies the colour so colour is never the only signal. */
export const BAND_COLOR: Record<Band, string> = {
  good: "var(--success)",
  fair: "var(--warning)",
  weak: "var(--error)",
  unknown: "var(--text-tertiary)",
};

export const BAND_MARK: Record<Band, string> = { good: "●", fair: "◐", weak: "○", unknown: "·" };
