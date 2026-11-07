import { test, expect } from "@playwright/test";
import { seed, visit } from "./support";

const { worldId } = seed();

// Every page.tsx under src/app/(desktop) that needs no dynamic id, plus the seeded-world ones.
const ROUTES = [
  "/",
  "/worlds",
  "/worlds/new",
  `/worlds/${worldId}`,
  `/worlds/${worldId}/adoptions`,
  `/worlds/${worldId}/pipeline`,
  "/sessions",
  "/sessions/new",
  "/evidence",
  "/reconstruct",
  "/activity",
  "/settings",
];

// Backend-less screens: their layout 404s unless the build sets NEXT_PUBLIC_SHOW_UNFINISHED=1 (lib/unfinished.ts).
const GATED = ["/analysis", "/analysis/new", "/results", "/reports", "/reports/new", "/team"];

for (const route of GATED) {
  test(`unfinished route ${route} is not served`, async ({ page }) => {
    test.skip(process.env.NEXT_PUBLIC_SHOW_UNFINISHED === "1", "build exposes unfinished routes");
    const res = await page.goto(route);
    expect(res?.status()).toBe(404);
  });
}

for (const route of ROUTES) {
  test(`route ${route.replace(worldId, "<seeded>")} loads healthy`, async ({ page }) => {
    const bag = await visit(page, route);
    // Settle: the shell footer reports the API state once the first poll answers.
    await expect(page.locator("footer").getByRole("status")).toContainText(/Online|Offline/);
    expect(bag.errors, `uncaught errors / console.error on ${route}`).toEqual([]);
  });
}
