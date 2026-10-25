import { test, expect } from "@playwright/test";
import { seed } from "./support";

const { worldId } = seed();

for (const route of ["/", "/worlds", "/evidence", `/worlds/${worldId}`]) {
  test(`shell present on ${route.replace(worldId, "<seeded>")}`, async ({ page }) => {
    await page.goto(route);
    await expect(page.getByRole("link", { name: "Reality Engine home" })).toContainText("Reality Engine");

    const rail = page.getByRole("navigation", { name: "Primary" });
    for (const label of ["Home", "Ingest", "Evidence", "Worlds", "Sessions", "Reports"]) {
      await expect(rail.getByText(label, { exact: true })).toBeVisible();
    }
    const secondary = page.getByRole("navigation", { name: "Secondary" });
    await expect(secondary.getByText("Settings", { exact: true })).toBeVisible();
    await expect(secondary.getByText("Help", { exact: true })).toBeVisible();

    // The API is up, so the footer must say Online - not merely show an indicator.
    await expect(page.locator("footer").getByRole("status")).toHaveText("Online");
  });
}
