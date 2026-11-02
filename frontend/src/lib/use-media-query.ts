/**
 * Shared responsive breakpoint hook. Single source of truth for the
 * mobile/desktop shell gate — do not duplicate matchMedia listeners
 * elsewhere.
 */
'use client';

import { useSyncExternalStore } from 'react';

// Portrait phones are narrow; landscape phones are short. Match either so a
// phone rotated to landscape (e.g. 812x375) still gets the mobile shell
// instead of a squeezed desktop layout with no vertical room for chrome.
const MOBILE_QUERY = '(max-width: 767px), (max-height: 500px) and (max-width: 926px)';

function subscribe(query: string, onChange: () => void) {
  const mql = window.matchMedia(query);
  mql.addEventListener('change', onChange);
  return () => mql.removeEventListener('change', onChange);
}

export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => subscribe(query, onChange),
    () => window.matchMedia(query).matches,
    () => false, // SSR snapshot: assume desktop until hydrated on the client
  );
}

/** True below 768px (phone widths, portrait or landscape). */
export function useIsMobile(): boolean {
  return useMediaQuery(MOBILE_QUERY);
}
