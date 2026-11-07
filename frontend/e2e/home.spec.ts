import { test, expect } from "@playwright/test";
import { API_URL, seed } from "./support";

test("Home lists the seeded world under Recent Worlds", async ({ page }) => {
  await page.goto("/");
  const recent = page.getByRole("region", { name: "Recent Worlds" });
  await expect(recent.getByRole("link", { name: new RegExp(seed().worldName) })).toBeVisible();
});

test("Home offers exactly two primary actions: New session and Open world", async ({ page }) => {
  await page.goto("/");
  const header = page.getByRole("banner").last().or(page.locator("header").last());
  await expect(header.getByRole("link", { name: "New session" })).toHaveAttribute("href", "/reconstruct");
  await expect(header.getByRole("link", { name: "Open world" })).toHaveAttribute("href", "/worlds");
  // the removed dashboard cards stay removed (no placeholder content on Home)
  await expect(page.getByRole("region", { name: "AI Assistant" })).toHaveCount(0);
  await expect(page.getByRole("region", { name: "Storage Usage" })).toHaveCount(0);
});

test("Needs attention shows a real failed job with the backend's own error and a link to it", async ({ page, request }) => {
  // A session with no photographs cannot be reconstructed: the backend fails the job and says why.
  const sid = (await (await request.post(`${API_URL}/api/sessions`, { data: { name: "Attention e2e" } })).json()).id as string;
  const wid = (await (await request.post(`${API_URL}/api/worlds`, { data: { name: "Attention e2e world" } })).json()).id as string;
  expect((await request.post(`${API_URL}/api/worlds/${wid}/attach/${sid}`)).ok()).toBeTruthy();
  const job = (await (await request.post(`${API_URL}/api/sessions/${sid}/reconstruct`)).json()) as { job_id: string };

  let final: { status: string; error: string | null } = { status: "queued", error: null };
  await expect
    .poll(async () => {
      final = await (await request.get(`${API_URL}/api/jobs/${job.job_id}`)).json();
      return final.status;
    }, { timeout: 30_000 })
    .toMatch(/failed|partial|succeeded/);
  expect(final.status).toBe("failed");
  expect(final.error).toBeTruthy();

  await page.goto("/");
  const panel = page.getByRole("region", { name: "Needs attention" });
  const row = panel.getByRole("link", { name: /Reconstruction failed/ }).first();
  await expect(row).toBeVisible();
  // the backend's own reason (final line of its error), never the raw traceback or a local file path
  const reason = final.error!.trim().split(/\r?\n/).pop()!.replace(/^[A-Za-z_][\w.]*:\s+/, "");
  await expect(row).toContainText(reason.slice(0, 25));
  await expect(row).not.toContainText("Traceback");
  await expect(row).not.toContainText(/\.py\b/);
  await expect(row).toHaveAttribute("href", new RegExp(`^/(sessions|worlds)/`));
});

test("Home has no horizontal scroll at phone width", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/");
  await expect(page.getByRole("region", { name: "Recent Worlds" })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
