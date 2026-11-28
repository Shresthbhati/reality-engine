import path from "node:path";
import { test } from "@playwright/test";
test("dbg", async ({ page }) => {
  test.setTimeout(120000);
  page.on("console", (m) => console.log("CONSOLE", m.type(), m.text().slice(0, 300)));
  page.on("framenavigated", (f) => console.log("NAV", f.url()));
  await page.goto("/reconstruct");
  await page.getByTestId("photo-input").setInputFiles(path.resolve(__dirname, "../../.scratch/media/walk.mp4"));
  await page.waitForTimeout(25000);
  console.log("URL", page.url());
  console.log((await page.locator("body").innerText()).slice(0, 800));
});
