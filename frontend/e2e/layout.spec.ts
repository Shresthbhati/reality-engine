import { test, expect } from "@playwright/test";
import { noHorizontalScroll } from "./support";

for (const size of [
  { width: 1280, height: 800 },
  { width: 1024, height: 768 },
]) {
  for (const route of ["/", "/worlds", "/evidence"]) {
    test(`no horizontal scrollbar on ${route} at ${size.width}x${size.height}`, async ({ page }) => {
      await page.setViewportSize(size);
      await page.goto(route);
      await expect(page.locator("footer").getByRole("status")).toContainText(/Online|Offline/);
      await noHorizontalScroll(page);
    });
  }
}
