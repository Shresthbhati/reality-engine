import { test, expect } from "@playwright/test";
import { API_URL, seed } from "./support";

test("Home lists the seeded world under Recent Worlds", async ({ page }) => {
  await page.goto("/");
  const recent = page.getByRole("region", { name: "Recent Worlds" });
  await expect(recent.getByRole("link", { name: new RegExp(seed().worldName) })).toBeVisible();
});

test("AI Assistant is visibly Coming soon and disabled", async ({ page }) => {
  await page.goto("/");
  const card = page.getByRole("region", { name: "AI Assistant" });
  await expect(card.getByText("Coming soon")).toBeVisible();
  await expect(card.getByRole("textbox")).toBeDisabled();
  for (const b of await card.getByRole("button").all()) await expect(b).toBeDisabled();
});

test("Overview numbers equal what the API reports (no invented figures)", async ({ page, request }) => {
  const status = (await (await request.get(`${API_URL}/api/status`)).json()) as {
    counts: { sessions: number; evidence: number };
  };
  await page.goto("/");
  const overview = page.getByRole("region", { name: "Overview" });
  const val = (label: string) => overview.getByText(label, { exact: true }).locator("xpath=preceding-sibling::p[1]");
  await expect(val("Sessions")).toHaveText(status.counts.sessions.toLocaleString("en-US"));
  await expect(val("Evidence items")).toHaveText(status.counts.evidence.toLocaleString("en-US"));
});
