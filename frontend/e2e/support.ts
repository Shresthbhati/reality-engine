import fs from "node:fs";
import { expect, type Page } from "@playwright/test";
import { API_URL, SEED_FILE } from "../playwright.config";

export { API_URL };

export function seed(): { worldId: string; worldName: string } {
  return JSON.parse(fs.readFileSync(SEED_FILE, "utf8"));
}

/** Console errors that are environmental noise, not product faults. Keep this list short. */
const CONSOLE_ALLOWLIST: RegExp[] = [
  /\[HMR\]|webpack-hmr|_next\/webpack|WebSocket connection to .*(_next|hmr)/i, // Next dev-server hot reload socket
  /favicon/i,
  // A procedural world has no point cloud: the API answers 404 and the viewport handles it (not a fault).
  /\/api\/worlds\/[^/]+\/points(\?.*)?$/,
  // Likewise a world that has never been reconstructed has no report yet (pipeline page renders that state).
  /\/api\/worlds\/[^/]+\/report(\?.*)?$/,
];

export function isAllowed(text: string): boolean {
  return CONSOLE_ALLOWLIST.some((re) => re.test(text));
}

/** Collects uncaught page errors and non-allowlisted console.error output. */
export function watchErrors(page: Page): { errors: string[] } {
  const bag = { errors: [] as string[] };
  page.on("pageerror", (e) => bag.errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    // The browser's generic "Failed to load resource" line carries no URL; the response listener below does.
    if (m.type() === "error" && !isAllowed(m.text()) && !/^Failed to load resource/.test(m.text())) {
      bag.errors.push(`console.error: ${m.text()}`);
    }
  });
  page.on("response", (r) => {
    if (r.status() >= 400 && !isAllowed(r.url())) bag.errors.push(`HTTP ${r.status()} ${r.request().method()} ${r.url()}`);
  });
  return bag;
}

/** Loads a route, asserts HTTP 200 and returns the collected errors bag. */
export async function visit(page: Page, url: string) {
  const bag = watchErrors(page);
  const res = await page.goto(url);
  expect(res?.status(), `HTTP status of ${url}`).toBe(200);
  return bag;
}

export async function noHorizontalScroll(page: Page) {
  const m = await page.evaluate(() => ({
    sw: document.documentElement.scrollWidth,
    cw: document.documentElement.clientWidth,
  }));
  expect(m.sw, `scrollWidth ${m.sw} > clientWidth ${m.cw}`).toBeLessThanOrEqual(m.cw);
}
