import { test, expect } from "@playwright/test";
import { API_URL } from "./support";

test("/results says it is not served by the API and shows no rows", async ({ page }) => {
  await page.goto("/results");
  await expect(page.getByText(/not served by the Reality Engine API/i)).toBeVisible();
  await expect(page.getByText(/no mock data/i)).toBeVisible();
});

test("/evidence counters match GET /api/evidence/quality", async ({ page, request }) => {
  const body = (await (await request.get(`${API_URL}/api/evidence/quality`)).json()) as {
    items: { result: string | null }[];
  };
  const n = (r: string) => body.items.filter((i) => i.result === r).length;
  await page.goto("/evidence");
  const counter = (label: string) =>
    page.getByText(label, { exact: true }).first().locator("xpath=preceding-sibling::div[1]");
  await expect(counter("Total files")).toHaveText(String(body.items.length));
  await expect(counter("Accepted")).toHaveText(String(n("accepted")));
  await expect(counter("Rejected")).toHaveText(String(n("rejected")));
  await expect(counter("Needs review")).toHaveText(String(n("needs_review")));
  await expect(counter("Accepted with warning")).toHaveText(String(n("accepted_with_warning")));
});

test("unreachable API shows an error state, not an empty state (/worlds and Home)", async ({ page }) => {
  await page.route("**/api/**", (r) => r.abort("connectionrefused"));

  await page.goto("/worlds");
  await expect(page.getByRole("alert").first()).toBeVisible();
  await expect(page.getByText(/No worlds yet/i)).toHaveCount(0);

  await page.goto("/");
  await expect(page.getByRole("region", { name: "Recent Worlds" }).getByRole("alert")).toBeVisible();
  await expect(page.getByText(/No worlds yet/i)).toHaveCount(0);
  await expect(page.locator("footer").getByRole("status")).toHaveText("Offline");
});
