/**
 * Analysis, Results, Reports and Team have no backend (see lib/api/unsupported.ts). Their routes stay in the tree but
 * return 404 unless NEXT_PUBLIC_SHOW_UNFINISHED=1, so a consumer never lands on a screen that cannot work.
 */
export const SHOW_UNFINISHED = process.env.NEXT_PUBLIC_SHOW_UNFINISHED === "1";
