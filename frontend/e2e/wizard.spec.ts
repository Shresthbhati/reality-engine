import { test, expect } from "@playwright/test";

test("City scale warns it is not a bigger photograph; time estimate is Unknown without runs/files", async ({ page }) => {
  await page.goto("/worlds/new");
  const city = page.getByRole("radio", { name: /^City:/ });
  // The card is a <label> around a visually hidden radio; retry the click until React has hydrated.
  await expect(async () => {
    await page.getByText("An entire city or metropolitan area").click();
    await expect(city).toBeChecked({ timeout: 1000 });
  }).toPass();
  await expect(page.getByText("A city is not a bigger photograph.")).toBeVisible();

  // No files are selected, so the API cannot derive a time (and a fresh DB has no past runs).
  const profile = page.getByRole("region", { name: "Expected Resource Profile" });
  const time = profile.locator("div", { has: page.getByText("Estimated Time", { exact: true }) }).last();
  await expect(time).toContainText("Unknown");
});
