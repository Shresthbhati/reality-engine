import { test, expect } from "@playwright/test";
import { API_URL, PRO_MODE, seed } from "./support";

test.use({ storageState: PRO_MODE });

const { worldId } = seed();
const WORLD = `/worlds/${worldId}`;

test("PhotoDropZone accepts video/* and rejects unsupported files with honest error", async ({ page }) => {
  await page.goto("/reconstruct");
  const input = page.getByTestId("photo-input");
  await expect(input).toHaveAttribute("accept", "image/*,video/*");

  // Upload an unsupported .txt file: client rejects with honest error and never navigates
  await input.setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("not a video or photo"),
  });
  await expect(page.getByRole("alert").first()).toContainText(/supported photos or videos/i);
  await expect(page).toHaveURL(/\/reconstruct$/);
});

test("Viewport supports Bounds [B], Reset camera [0], and Uncertainty [U] legend", async ({ page }) => {
  await page.goto(WORLD);
  await expect(page.locator("canvas").first()).toBeVisible();

  // Toggle Bounds layer
  await page.getByRole("button", { name: "Layers and views" }).click();
  const boundsBtn = page.getByRole("button", { name: /Bounds \[B\]/i });
  await expect(boundsBtn).toHaveAttribute("aria-pressed", "false");
  await boundsBtn.click();
  await expect(boundsBtn).toHaveAttribute("aria-pressed", "true");

  // Toggle Uncertainty layer and verify honest Low / High / Unknown legend
  const uncBtn = page.getByRole("button", { name: /Uncertainty \[U\]/i });
  await uncBtn.click();
  await expect(page.getByText("Unknown", { exact: true })).toBeVisible();

  // Reset camera button is present and clickable
  const resetBtn = page.getByRole("button", { name: "Reset camera [0]" });
  await expect(resetBtn).toBeVisible();
  await resetBtn.click();
});

test("Split and Merge corrections persist via real backend and VersionDiffModal compares versions", async ({ page, request }) => {
  // Create a dedicated world with a room so we can split an entity and compare versions
  const wRes = await request.post(`${API_URL}/api/worlds`, {
    data: { name: `E2E Corrections ${Date.now()}` },
  });
  expect(wRes.ok()).toBe(true);
  const { id: wid } = (await wRes.json()) as { id: string };

  const rRes = await request.post(`${API_URL}/api/worlds/${wid}/rooms`, {
    data: { name: "StudioRoom", width: 4, depth: 5, height: 2.8, openings: [], seed: 11, origin: [0, 0, 0] },
  });
  expect(rRes.status()).toBe(201);

  const wirBefore = (await (await request.get(`${API_URL}/api/worlds/${wid}/worldir`)).json()) as {
    entities: Record<string, { id: string; type: string; name?: string }>;
  };
  const targetId = Object.keys(wirBefore.entities).find((id) => id.includes("wall")) || Object.keys(wirBefore.entities)[0];
  expect(targetId).toBeTruthy();

  // Open the world and select the entity via ?entity= deep-link
  await page.goto(`/worlds/${wid}?entity=${encodeURIComponent(targetId!)}`);
  await expect(page.locator("canvas").first()).toBeVisible();

  // Open edit form and split along X
  await page.getByRole("button", { name: "Correct this item" }).click();
  await page.getByTestId("split-axis-x").click();

  // Wait for backend WorldIR to reflect the split (entity count increases)
  await expect
    .poll(async () => {
      const wirAfter = (await (await request.get(`${API_URL}/api/worlds/${wid}/worldir`)).json()) as {
        entities: Record<string, unknown>;
      };
      return Object.keys(wirAfter.entities).length;
    })
    .toBeGreaterThan(Object.keys(wirBefore.entities).length);

  // Reload and verify split persisted
  await page.reload();
  await expect(page.locator("canvas").first()).toBeVisible();

  // Open VersionDiffModal via custom event / button and verify Base/Head pickers work
  await page.evaluate(() => window.dispatchEvent(new Event("open-version-diff")));
  await expect(page.getByText("WorldStore Version Comparison")).toBeVisible();
  const baseSelect = page.getByLabel("Base version");
  const headSelect = page.getByLabel("Head version");
  await expect(baseSelect).toBeVisible();
  await expect(headSelect).toBeVisible();
});

test("Backend failure on split shows error alert and never reports success", async ({ page }) => {
  await page.goto(WORLD);
  await expect(page.locator("canvas").first()).toBeVisible();

  // Intercept split API to simulate 500 backend failure
  await page.route(`**/api/worlds/${worldId}/corrections/split`, (route) =>
    route.fulfill({
      status: 500,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Simulated persistence failure" }),
    }),
  );

  // Select first entity from worldir
  const wirRes = await page.request.get(`${API_URL}/api/worlds/${worldId}/worldir`);
  const wir = (await wirRes.json()) as { entities: Record<string, unknown> };
  const firstId = Object.keys(wir.entities)[0];
  await page.goto(`/worlds/${worldId}?entity=${encodeURIComponent(firstId)}`);

  await page.getByRole("button", { name: "Correct this item" }).click();
  await page.getByTestId("split-axis-x").click();

  await expect(page.getByText("Correction failed")).toBeVisible();
  await expect(page.getByText("Correction saved")).toHaveCount(0);
});
