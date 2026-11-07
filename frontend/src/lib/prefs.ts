"use client";

import { useSyncExternalStore } from "react";

/**
 * Per-viewer preferences (theme, Pro mode, density, processing).
 * Browser storage only: tolerates blocked storage gracefully.
 */
export type ThemePref = "system" | "light" | "dark";
export type DensityPref = "comfortable" | "compact";

export interface ProcessingPref {
  gpuAcceleration: boolean;
  maxImageDimension: number;
  denseFusion: boolean;
}

const THEME_KEY = "re-theme";
const DENSITY_KEY = "re-density";
const PROCESSING_KEY = "re-processing-pref";
const EVENT = "re-prefs-change";

function read(key: string): string | null {
  try {
    return typeof window !== "undefined" ? window.localStorage.getItem(key) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: string) {
  try {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(key, value);
    }
  } catch {
    // Blocked storage: preference applies to this page load only.
  }
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(EVENT));
  }
}

function subscribe(cb: () => void) {
  if (typeof window === "undefined") return () => {};
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}

export function getTheme(): ThemePref {
  const v = read(THEME_KEY);
  return v === "light" || v === "dark" ? v : "system";
}

/** Applies the theme to <html>. "system" removes the attribute so the OS setting decides. */
export function applyTheme(theme: ThemePref) {
  if (typeof document === "undefined") return;
  if (theme === "system") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
}

export function setTheme(theme: ThemePref) {
  write(THEME_KEY, theme);
  applyTheme(theme);
}

export function useTheme(): ThemePref {
  return useSyncExternalStore(subscribe, getTheme, () => "system" as ThemePref);
}

export function getDensity(): DensityPref {
  const v = read(DENSITY_KEY);
  return v === "compact" ? "compact" : "comfortable";
}

export function setDensity(density: DensityPref) {
  write(DENSITY_KEY, density);
  if (typeof document !== "undefined") {
    document.documentElement.setAttribute("data-density", density);
  }
}

export function useDensity(): DensityPref {
  return useSyncExternalStore(subscribe, getDensity, () => "comfortable" as DensityPref);
}

const DEFAULT_PROCESSING: ProcessingPref = {
  gpuAcceleration: true,
  maxImageDimension: 2048,
  denseFusion: true,
};

export function getProcessingPref(): ProcessingPref {
  const raw = read(PROCESSING_KEY);
  if (!raw) return DEFAULT_PROCESSING;
  try {
    const parsed = JSON.parse(raw);
    return {
      gpuAcceleration: Boolean(parsed.gpuAcceleration ?? true),
      maxImageDimension: Number(parsed.maxImageDimension ?? 2048),
      denseFusion: Boolean(parsed.denseFusion ?? true),
    };
  } catch {
    return DEFAULT_PROCESSING;
  }
}

export function setProcessingPref(pref: Partial<ProcessingPref>) {
  const current = getProcessingPref();
  const updated = { ...current, ...pref };
  write(PROCESSING_KEY, JSON.stringify(updated));
}

export function useProcessingPref(): ProcessingPref {
  return useSyncExternalStore(subscribe, getProcessingPref, () => DEFAULT_PROCESSING);
}
