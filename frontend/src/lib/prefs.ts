"use client";

import { useSyncExternalStore } from "react";

/**
 * Per-viewer preferences (theme, Pro mode). Browser storage only: nothing here is shared or must persist reliably,
 * and every read/write tolerates blocked storage.
 */
export type ThemePref = "system" | "light" | "dark";

const THEME_KEY = "re-theme";
const PRO_KEY = "re-pro-mode";
const EVENT = "re-prefs-change";

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Blocked storage: the preference applies to this page load only.
  }
  window.dispatchEvent(new Event(EVENT));
}

function subscribe(cb: () => void) {
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

export function setProMode(on: boolean) {
  write(PRO_KEY, String(on));
}

/** Pro mode shows the engineer workstation and technical terms. Server render and first paint are Simple mode. */
export function useProMode(): boolean {
  return useSyncExternalStore(subscribe, () => read(PRO_KEY) === "true", () => false);
}
