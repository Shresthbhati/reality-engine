import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expect, test } from "@playwright/test";

/**
 * Offline path: photos / video that live on THIS PC go in either by the browser (file input) or by typing a
 * local path (the API reads the disk itself). Whatever the engine then does, the UI must say it -- a failed or
 * unreconstructable import is never an empty viewport.
 */
let media: string;

test.beforeAll(() => {
  media = fs.mkdtempSync(path.join(os.tmpdir(), "re-e2e-media-"));
  execFileSync("python", [path.join(__dirname, "make_local_media.py"), media], { stdio: "pipe" });
});

test.afterAll(() => {
  fs.rmSync(media, { recursive: true, force: true });
});

test("a bad local path is refused with the exact reason", async ({ page }) => {
  await page.goto("/reconstruct");
  await page.getByText("Import a folder or video from this computer").click();
  await page.getByTestId("local-path-input").fill(path.join(media, "does-not-exist"));
  await page.getByTestId("local-path-submit").click();
  const alert = page.getByRole("alert").filter({ hasText: "does not exist on this machine" });
  await expect(alert).toBeVisible();
  await expect(page).toHaveURL(/\/reconstruct/);
});

test("a folder on this PC becomes a world and the UI never goes silent", async ({ page }) => {
  test.setTimeout(300_000); // a cold dev server compiles the world page on first visit; the engine then really runs
  await page.goto("/reconstruct");
  await page.getByText("Import a folder or video from this computer").click();
  await page.getByTestId("local-path-input").fill(path.join(media, "photos"));
  await page.getByTestId("local-path-submit").click();
  await page.waitForURL(/\/worlds\/wld_/, { timeout: 120_000 });
  // while it runs, a state label; when it ends, either a model or the engine's own reason -- never a blank viewport
  const explained = page
    .getByTestId("no-geometry-explanation")
    .or(page.getByRole("region", { name: "Reconstruction status" }));
  await expect(explained.first()).toBeVisible();
  await expect(page.getByTestId("no-geometry-explanation").getByRole("heading")).not.toHaveText("No model yet", {
    timeout: 120_000,
  });
});

test("a video chosen in the browser is accepted (not filtered out as a non-image)", async ({ page }) => {
  await page.goto("/reconstruct");
  test.setTimeout(180_000);
  const posted = page.waitForResponse((r) => r.url().endsWith("/api/reconstructions") && r.request().method() === "POST");
  await page.getByTestId("photo-input").setInputFiles(path.join(media, "walk.mp4"));
  const body = (await (await posted).json()) as { evidence: { name: string; type: string }[] };
  expect(body.evidence.find((e) => e.name === "walk.mp4")?.type).toBe("video");
  expect(body.evidence.filter((e) => e.type === "photo").length).toBeGreaterThanOrEqual(2); // sampled frames
  await page.waitForURL(/\/worlds\/wld_/, { timeout: 120_000 });
});

test("settings state where COLMAP is, or exactly where it looked", async ({ page }) => {
  await page.goto("/settings");
  const status = page.getByTestId("colmap-status");
  await expect(status).toHaveText(/found|not found/);
  if ((await status.textContent()) === "not found") {
    await expect(page.getByText("Looked in:")).toBeVisible();
  } else {
    await expect(page.getByTestId("reconstruction-tools")).toContainText("via");
  }
  await expect(page.getByTestId("video-status")).toContainText("Video:");
});
