import { test, expect } from "@playwright/test";
import { API_URL, seed } from "./support";

const { worldId } = seed();
const WORLD = `/worlds/${worldId}`;

test("seeded world renders the 3D canvas", async ({ page }) => {
  await page.goto(WORLD);
  await expect(page.locator("canvas").first()).toBeVisible();
});

test("Build dialog creates a room that lands in the WorldIR", async ({ page, request }) => {
  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("tab", { name: "room" }).click();
  await dialog.locator("input").first().fill("Annex");
  await dialog.getByRole("button", { name: "Build room" }).click();
  await expect(dialog).toBeHidden();

  await expect
    .poll(async () => {
      const wir = (await (await request.get(`${API_URL}/api/worlds/${worldId}/worldir`)).json()) as {
        entities: Record<string, unknown>;
      };
      return Object.keys(wir.entities);
    })
    .toContain("Annex-room");
});

test("Measure panel opens: Distance enabled, Volume disabled", async ({ page }) => {
  await page.goto(WORLD);
  await page.getByRole("button", { name: "Toggle measurement mode" }).click();
  const panel = page.getByLabel("Measurement mode", { exact: true });
  await expect(panel).toBeVisible();
  await expect(panel.getByRole("button", { name: "Distance" })).toBeEnabled();
  await expect(panel.getByRole("button", { name: "Volume" })).toBeDisabled();
});

test("Geospatial Map tab says the local frame is not georeferenced", async ({ page }) => {
  await page.goto(WORLD);
  await page.getByRole("button", { name: "Geospatial Map" }).click();
  await expect(page.getByText(/not georeferenced/i).first()).toBeVisible();
});
