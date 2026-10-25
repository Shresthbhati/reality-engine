import { test, expect } from "@playwright/test";
import { visit, noHorizontalScroll } from "./support";

const TABS = ["Home", "Capture", "Evidence", "Sessions", "More"];

for (const route of ["/m", "/m/camera", "/m/evidence", "/m/sessions", "/m/more"]) {
  test(`mobile ${route}: tab bar visible, no overflow, no errors`, async ({ page }) => {
    const bag = await visit(page, route);
    const tabs = page.getByRole("navigation", { name: "Primary" });
    for (const t of TABS) await expect(tabs.getByRole("link", { name: t })).toBeVisible();
    await noHorizontalScroll(page);
    expect(bag.errors).toEqual([]);
  });
}

test("camera page explains itself when getUserMedia is unavailable (not blank)", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "mediaDevices", { value: undefined, configurable: true });
  });
  await page.goto("/m/camera");
  const alert = page.getByRole("alert").filter({ hasText: /camera/i });
  await expect(alert).toBeVisible();
  await expect(alert.getByRole("button", { name: /Try camera again/i })).toBeVisible();
});
