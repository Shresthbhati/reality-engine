import { test, expect } from "@playwright/test";
import { seed, visit } from "./support";

const { worldId } = seed();

// Every page.tsx under src/app/(desktop) that needs no dynamic id, plus the seeded-world ones.
const ROUTES = [
  "/",
  "/worlds",
  "/worlds/new",
  `/worlds/${worldId}`,
  `/worlds/${worldId}/pipeline`,
  "/sessions",
  "/sessions/new",
  "/evidence",
  "/analysis",
  "/analysis/new",
  "/results",
  "/reports",
  "/reports/new",
  "/reconstruct",
  "/activity",
  "/settings",
  "/team",
];

for (const route of ROUTES) {
  test(`route ${route.replace(worldId, "<seeded>")} loads healthy`, async ({ page }) => {
    const bag = await visit(page, route);
    // Settle: the shell footer reports the API state once the first poll answers.
    await expect(page.locator("footer").getByRole("status")).toContainText(/Online|Offline/);
    expect(bag.errors, `uncaught errors / console.error on ${route}`).toEqual([]);
  });
}
