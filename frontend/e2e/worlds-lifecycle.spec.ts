import { test, expect } from "@playwright/test";
import { API_URL } from "./support";

test.describe("World Card Lifecycle and Management (Task 1.4)", () => {
  test.describe.configure({ mode: "serial" });

  let createdWorldId: string;
  const initialWorldName = `Lifecycle Site ${Date.now()}`;

  test.beforeAll(async ({ request }) => {
    // Create an isolated world for lifecycle testing
    const res = await request.post(`${API_URL}/api/worlds`, {
      data: { name: initialWorldName },
    });
    expect(res.ok()).toBeTruthy();
    const body = (await res.json()) as { id: string };
    createdWorldId = body.id;
  });

  test("World card menu provides Open, Export, Rename, Duplicate, Archive, and Delete", async ({ page }) => {
    await page.goto("/worlds");

    // Locate the specific world card
    const card = page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`);
    await expect(card).toBeVisible();
    await expect(card).toContainText(initialWorldName);

    // Open card menu
    const menuBtn = card.getByTestId("world-menu-btn");
    await menuBtn.click();

    // Verify Open and Export links
    const openBtn = card.getByTestId("world-open-btn");
    await expect(openBtn).toBeVisible();
    await expect(openBtn).toHaveAttribute("href", `/worlds/${createdWorldId}`);

    const exportBtn = card.getByTestId("world-export-btn");
    await expect(exportBtn).toBeVisible();
    await expect(exportBtn).toHaveAttribute("href", `/worlds/${createdWorldId}?tab=export`);

    // Verify Rename, Duplicate, Archive, and Delete buttons are present
    await expect(card.getByTestId("world-rename-btn")).toBeVisible();
    await expect(card.getByTestId("world-duplicate-btn")).toBeVisible();
    await expect(card.getByTestId("world-archive-btn")).toBeVisible();
    await expect(card.getByTestId("world-delete-btn")).toBeVisible();
  });

  test("Duplicate creates a copy that appears in active worlds", async ({ page }) => {
    await page.goto("/worlds");
    const card = page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`);
    await expect(card).toBeVisible();

    await card.getByTestId("world-menu-btn").click();
    await card.getByTestId("world-duplicate-btn").click();

    // Wait for the copy to appear in active worlds
    const copyCard = page.locator('[data-testid="world-card"]').filter({ hasText: `${initialWorldName} (Copy)` });
    await expect(copyCard.first()).toBeVisible({ timeout: 10_000 });
  });

  test("Delete confirmation lists what will be deleted and moves world to Trash", async ({ page }) => {
    await page.goto("/worlds");
    const card = page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`);
    await expect(card).toBeVisible();

    await card.getByTestId("world-menu-btn").click();
    await card.getByTestId("world-delete-btn").click();

    // Verify Delete Confirmation Modal lists affected contents
    const affectedItems = page.getByTestId("delete-affected-items");
    await expect(affectedItems).toBeVisible();
    await expect(affectedItems).toContainText(initialWorldName);
    await expect(affectedItems).toContainText("World:");
    await expect(affectedItems).toContainText("Sessions:");
    await expect(affectedItems).toContainText("Evidence:");

    // Confirm Move to Trash
    const confirmBtn = page.getByTestId("confirm-delete-btn");
    await expect(confirmBtn).toHaveText("Move to Trash");
    const [delResponse] = await Promise.all([
      page.waitForResponse((r) => r.url().includes("/api/worlds") && r.request().method() === "DELETE"),
      confirmBtn.click(),
    ]);
    expect(delResponse.status()).toBe(200);

    // Verify it is no longer listed in active worlds
    await expect(page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`)).toHaveCount(0);

    // Verify it appears in Trash tab
    await page.goto("/worlds?status=trash");
    const trashCard = page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`);
    await expect(trashCard).toBeVisible();
  });

  test("Trash view allows restoring or permanently purging a world", async ({ page }) => {
    await page.goto("/worlds?status=trash");
    const trashCard = page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`);
    await expect(trashCard).toBeVisible();

    // Menu in Trash view should offer Restore and Permanently Delete
    await trashCard.getByTestId("world-menu-btn").click();
    await expect(trashCard.getByTestId("world-restore-btn")).toBeVisible();
    await expect(trashCard.getByTestId("world-delete-btn")).toHaveText("Permanently Delete");

    // Click Permanently Delete
    await trashCard.getByTestId("world-delete-btn").click();
    const affectedItems = page.getByTestId("delete-affected-items");
    await expect(affectedItems).toBeVisible();
    await expect(affectedItems).toContainText("permanently purged");

    const purgeBtn = page.getByTestId("confirm-delete-btn");
    await expect(purgeBtn).toHaveText("Purge Permanently");
    const [purgeResponse] = await Promise.all([
      page.waitForResponse((r) => r.url().includes("/api/worlds") && r.request().method() === "DELETE"),
      purgeBtn.click(),
    ]);
    expect(purgeResponse.status()).toBe(200);

    // Verify world is gone from Trash
    await expect(page.locator(`[data-testid="world-card"][data-world-id="${createdWorldId}"]`)).toHaveCount(0);
  });

  test("Visual inspection screenshots at 1280 and 1920 viewports", async ({ page }) => {
    // 1280x800 desktop
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.goto("/worlds");
    await expect(page.getByRole("heading", { name: "Worlds" })).toBeVisible();
    await page.screenshot({ path: "test-results/worlds-1280.png" });

    // 1920x1080 wide desktop
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/worlds");
    await expect(page.getByRole("heading", { name: "Worlds" })).toBeVisible();
    await page.screenshot({ path: "test-results/worlds-1920.png" });
  });
});
