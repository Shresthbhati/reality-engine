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

test("camera page shows live coverage hint when backend coverage is available", async ({ page }) => {
  const worldId = "wld_test_coverage_hint";
  await page.route(`**/api/worlds/${worldId}/capture-coverage`, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        available: true,
        reason: null,
        walls: [
          {
            index: 1,
            label: "wall 1",
            between: ["A", "B"],
            p0: [-3, -3],
            p1: [3, -3],
            status: "weak",
            covered_fraction: 0.4,
            multi_view_fraction: 0.1,
            hint: "Take 2 more photos of wall 1, standing on opposite side.",
          },
        ],
        hints: ["Take 2 more photos of wall 1, standing on opposite side."],
        summary: { seen: 0, weak: 1, not_seen: 0 },
      }),
    });
  });

  await page.goto(`/m/camera?world=${worldId}`);
  const hint = page.getByTestId("live-coverage-hint");
  await expect(hint).toBeVisible();
  await expect(hint).toContainText("Take 2 more photos of wall 1, standing on opposite side.");
});

test("camera page recovers offline queued photos from IndexedDB across page reload", async ({ page }) => {
  await page.goto("/m/camera");
  // Seed pending item in IndexedDB before reload
  await page.evaluate(async () => {
    const req = indexedDB.open("reality-engine-offline-capture", 1);
    req.onupgradeneeded = (e: IDBVersionChangeEvent) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains("capture-photos")) {
        db.createObjectStore("capture-photos", { keyPath: "id" });
      }
    };
    await new Promise<void>((resolve, reject) => {
      req.onsuccess = () => {
        const db = req.result;
        const tx = db.transaction("capture-photos", "readwrite");
        const store = tx.objectStore("capture-photos");
        const blob = new Blob(["fake photo data"], { type: "image/jpeg" });
        store.put({
          id: "photo-offline-test-1",
          name: "offline_test.jpg",
          size: 1024,
          type: "image/jpeg",
          blob,
          status: "queued",
          worldId: null,
          attempts: 0,
          createdAt: Date.now(),
        });
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      };
      req.onerror = () => reject(req.error);
    });
  });

  // Reload page - should hydrate offline item
  await page.reload();
  const queueList = page.getByRole("list");
  await expect(queueList).toBeVisible();
  await expect(queueList.getByText("offline_test.jpg")).toBeVisible();
});
