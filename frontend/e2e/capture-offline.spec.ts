import { test, expect, type Page } from "@playwright/test";

/**
 * M5 on a phone-sized screen: capture survives a lost connection and a closed tab (T-5.3), and the session holds a GPS track (T-5.1).
 * The photos are generated in the page (noise JPEGs); the only thing simulated is the connection (the upload request is aborted).
 */
test.use({
  viewport: { width: 390, height: 844 },
  permissions: ["geolocation"],
  geolocation: { latitude: 51.5, longitude: -0.12, accuracy: 8 },
});

async function jpeg(page: Page, seed: number): Promise<{ name: string; mimeType: string; buffer: Buffer }> {
  const b64 = await page.evaluate((s) => {
    const c = document.createElement("canvas");
    c.width = 800;
    c.height = 600;
    const g = c.getContext("2d")!;
    const img = g.createImageData(800, 600);
    let x = s * 7919 + 1;
    for (let i = 0; i < img.data.length; i += 4) {
      x = (x * 1103515245 + 12345) & 0x7fffffff;
      img.data[i] = (x >> 8) & 255;
      img.data[i + 1] = (x >> 16) & 255;
      img.data[i + 2] = x & 255;
      img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return c.toDataURL("image/jpeg", 0.9).split(",")[1];
  }, seed);
  return { name: `offline-${seed}.jpg`, mimeType: "image/jpeg", buffer: Buffer.from(b64, "base64") };
}

const storedCount = (page: Page) =>
  page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        const open = indexedDB.open("reality-capture", 1);
        open.onsuccess = () => {
          const req = open.result.transaction("photos").objectStore("photos").count();
          req.onsuccess = () => resolve(req.result);
        };
      }),
  );

test("photos added offline survive a closed tab and upload when the connection returns", async ({ page, context }) => {
  await page.route("**/api/reconstructions", (r) => r.abort("failed")); // offline: every upload fails at the network level
  await page.goto("/m/camera");
  const files = [await jpeg(page, 1), await jpeg(page, 2)];
  await page.locator('input[type="file"][multiple]').first().setInputFiles(files);
  await expect.poll(() => storedCount(page)).toBe(2); // on the device, not just in memory

  await page.close(); // kill the tab, come back online
  const page2 = await context.newPage();
  const uploads: string[] = [];
  page2.on("request", (r) => r.method() === "POST" && /\/api\/reconstructions$/.test(r.url()) && uploads.push(r.url()));
  const fixPosts: number[] = [];
  page2.on("request", (r) => {
    if (r.method() === "POST" && /\/api\/sessions\/[^/]+\/locations$/.test(r.url())) fixPosts.push((r.postDataJSON()?.points ?? []).length);
  });
  await page2.goto("/m/camera");
  await expect(page2.getByText(/2 photos from before the page closed were kept and will upload now/)).toBeVisible();
  await expect.poll(() => uploads.length, { timeout: 30_000 }).toBeGreaterThanOrEqual(1);
  await expect.poll(() => storedCount(page2), { timeout: 60_000 }).toBeLessThan(2); // an uploaded photo leaves the device store
  await expect.poll(() => fixPosts.reduce((a, n) => a + n, 0), { timeout: 30_000 }).toBeGreaterThanOrEqual(1); // the track reaches the session
});

test("the session holds a track of GPS fixes, not one reading", async ({ page, context }) => {
  await page.goto("/m/camera");
  for (let i = 1; i <= 4; i++) {
    await context.setGeolocation({ latitude: 51.5 + i * 0.0001, longitude: -0.12, accuracy: 8 });
    await page.waitForTimeout(2300);
  }
  await expect(page.getByText(/GPS ±\d+ m · [2-9]\d* fixes/)).toBeVisible();
});

test("after a batch the coverage API's advice is shown once, calmly, and stays quiet when dismissed", async ({ page }) => {
  const status = { world_id: "w-hint", name: "Hint world", state: "PARTIALLY_COMPLETE", state_label: "Partially complete", in_progress: false, job: null, has_model: true,
    model: { version_id: "v-1", level: 2, level_name: "sparse", model_state: "ROUGH", outcome: "PARTIAL_SUCCESS", attempts: [], dense: { state: "sparse", detail: "" },
      scale: { state: "relative", meters_per_unit: null, references: [], note: null, scale_basis: null, dimension_state: null, provisional_unit_k: null, provisional_unit_base_k: null, dropped: [] },
      plane_sweep: null, device_depth_scale: null, identity: null, structure_points: [], images_used: 0, images_registered: 0, changes: [], verdict: "ACCEPT",
      uncertainties: [], conflicts: [], physical: { available: false, reason: "no previous structure to compare" }, strategy: null },
    last_run: null, evidence: [], evidence_summary: { count: 0, headline: null, quality: [], excluded: [] }, guidance: [], versions: [], failure: null };
  const coverage = { available: true, reason: null, provisional: false, walls: [], hints: ["Take 2 photos of wall 3 (between corner C and D)"], summary: { seen: 2, weak: 0, not_seen: 1 } };
  let coverageCalls = 0;
  await page.route("**/api/worlds/w-hint/status", (r) => r.fulfill({ json: status }));
  await page.route("**/api/worlds/w-hint/capture-coverage", (r) => { coverageCalls++; return r.fulfill({ json: coverage }); });
  await page.goto("/m/camera?world=w-hint");
  const hint = page.getByRole("status", { name: "Capture hint" });
  await expect(hint).toContainText("Take 2 photos of wall 3");
  await hint.getByRole("button", { name: "Got it" }).click();
  await expect(hint).toHaveCount(0);
  await page.waitForTimeout(1500);
  await expect(hint).toHaveCount(0); // not shown again for the same advice
  expect(coverageCalls).toBeGreaterThanOrEqual(1);
});
