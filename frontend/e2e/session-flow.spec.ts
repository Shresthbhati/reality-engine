import fs from "node:fs";
import path from "node:path";
import { test, expect } from "@playwright/test";
import { seed } from "./support";

/**
 * U.5: one guided flow for starting a session, with processing states that match the backend's real stages.
 * Only the status RESPONSE is stubbed (to hold a job in each stage); stage names are read from apps/api/jobs.py, so a new
 * backend stage without a label fails tests/contract/stage-labels.test.mts and a UI that invents progress fails here.
 */
const jobsPy = fs.readFileSync(path.resolve(__dirname, "../../apps/api/jobs.py"), "utf8");
const BACKEND_STAGES = [...new Set([...jobsPy.matchAll(/job\.stage\s*=\s*"([a-z_]+)"/g)].map((m) => m[1]))];

test("the backend stage list was found", () => {
  expect(BACKEND_STAGES.length).toBeGreaterThanOrEqual(5);
});

test("/sessions/new is not a form any more: it lands on the guided flow", async ({ page }) => {
  await page.goto("/sessions/new");
  await expect(page).toHaveURL(/\/reconstruct$/);
  await expect(page.getByRole("heading", { name: "New session" })).toBeVisible();
  await expect(page.getByLabel("Name")).toHaveCount(0); // no up-front configuration form
});

test("/sessions/new keeps ?world= so adding to an existing world still works", async ({ page }) => {
  const { worldId } = seed();
  await page.goto(`/sessions/new?world=${worldId}`);
  await expect(page).toHaveURL(new RegExp(`/reconstruct\\?world=${worldId}$`));
});

test("Home's New session opens the guided flow with step 1 as the only action", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "New session" }).first().click();
  await expect(page).toHaveURL(/\/reconstruct$/);
  const flow = page.getByTestId("guided-flow");
  await expect(flow.getByTestId("flow-step-photos")).toHaveAttribute("data-state", "action");
  for (const id of ["reconstruct", "scale", "review", "measure"]) {
    await expect(flow.getByTestId(`flow-step-${id}`)).toHaveAttribute("data-state", "unavailable"); // nothing pretends to have started
  }
  await expect(page.getByRole("progressbar")).toHaveCount(0);
});

const LABEL: Record<string, string> = {
  verifying_artifact: "Checking the uploaded files",
  registering: "Registering the world",
  checking_reconstruction_backend: "Checking the reconstruction engine",
  resolving_evidence: "Collecting your photos",
  analyzing_evidence: "Analysing the photos",
  committing_version: "Saving the new version",
};

for (const stage of BACKEND_STAGES) {
  test(`stage "${stage}" shows its plain-language name, no percentage, no progress bar`, async ({ page }) => {
    const { worldId } = seed();
    await page.route(`**/api/worlds/${worldId}/status**`, (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          world_id: worldId, state: "PROCESSING", state_label: "Processing", in_progress: true, has_model: false,
          evidence_summary: { count: 6 }, evidence: [], guidance: [], model: null, failure: null,
          job: { id: "job_stub", status: "running", stage, session_id: "sess_stub" },
        }),
      }),
    );
    await page.goto(`/reconstruct?world=${worldId}`);
    const step = page.getByTestId("flow-step-reconstruct");
    await expect(step).toHaveAttribute("data-state", "active");
    const expected = LABEL[stage] ?? stage.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
    await expect(step).toContainText(expected);
    // the processing step itself carries no percentage (the static capture tip "overlap 60%" is advice, not progress)
    await expect(step).not.toContainText(/\d+\s*%/);
    await expect(page.getByRole("progressbar")).toHaveCount(0);
  });
}
