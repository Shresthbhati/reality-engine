import path from "node:path";
import { execFileSync } from "node:child_process";
import { test, expect, type APIRequestContext } from "@playwright/test";
import { API_URL, seed } from "./support";

const root = path.resolve(__dirname, "..", "..");
const sandbox = path.join(__dirname, "..", "test-results", "e2e");

/** A fresh world whose head is a RECONSTRUCTED floor + wall shell (see seed_shell.py for why this is not via a public API). */
async function shellWorld(request: APIRequestContext): Promise<string> {
  const w = await request.post(`${API_URL}/api/worlds`, { data: { name: `Shell ${Date.now()}` } });
  expect(w.ok()).toBeTruthy();
  const { id } = (await w.json()) as { id: string };
  execFileSync("python", [path.join(__dirname, "seed_shell.py"), id, "6", "5"], {
    cwd: root,
    env: {
      ...process.env,
      PYTHONPATH: root,
      DATABASE_URL: `sqlite+aiosqlite:///${path.join(sandbox, "app.db").replace(/\\/g, "/")}`,
      STORAGE_ROOT: path.join(sandbox, "storage"),
      WORLDSTORE_ROOT: path.join(sandbox, "worldstore"),
    },
    stdio: "pipe",
  });
  return id;
}

async function head(request: APIRequestContext, id: string): Promise<string> {
  return ((await (await request.get(`${API_URL}/api/worlds/${id}`)).json()) as { current_version_id: string }).current_version_id;
}

test("no reconstructed shell: the honest 422 text is shown and nothing is proposed", async ({ page }) => {
  // The seeded world only holds GENERATED rooms/cities, which never count as a shell.
  await page.goto(`/worlds/${seed().worldId}`);
  await page.getByTestId("open-complete").click();
  const dialog = page.getByRole("dialog", { name: "Complete interior" });
  await dialog.getByRole("button", { name: "Propose", exact: true }).click();
  await expect(dialog.getByTestId("complete-error")).toContainText("no usable shell");
  await expect(dialog.getByTestId("complete-proposal")).toHaveCount(0);
});

test("propose -> labelled diff -> accept as GENERATED; name clash 409 verbatim; reject leaves the world alone", async ({ page, request }) => {
  const id = await shellWorld(request);
  const name = `flat${Date.now()}`;
  await page.goto(`/worlds/${id}`);
  await page.getByTestId("open-complete").click();
  const dialog = page.getByRole("dialog", { name: "Complete interior" });
  await expect(dialog.getByTestId("complete-empty")).toBeVisible();

  // Batch A: propose, review, accept.
  await dialog.getByLabel("Interior name").fill(name);
  await dialog.getByRole("button", { name: "Propose", exact: true }).click();
  const cards = dialog.getByTestId("complete-proposal");
  await expect(cards.first()).toBeVisible({ timeout: 60_000 });
  expect(await cards.count()).toBeGreaterThanOrEqual(1);
  await expect(cards.first().getByTestId("complete-badge")).toHaveText("PROPOSED");
  await expect(cards.first()).toContainText("axis-aligned bounding box");
  await expect(cards.first()).toContainText("model units");
  // Real floor plan: studio program = living, kitchen, bath (plus any hallway the solver adds).
  const plan = cards.first().getByTestId("complete-plan");
  await expect(plan).toBeVisible();
  expect(await plan.getByTestId("complete-plan-room").count()).toBeGreaterThanOrEqual(3);
  await expect(cards.first()).toContainText("GENERATED rooms");

  const diff = dialog.getByTestId("complete-diff");
  await expect(diff.getByTestId("complete-diff-counts")).toContainText(/^\d+ added, 0 changed, 0 removed$/);
  await expect(diff.getByTestId("complete-diff-existing")).toHaveText("2 RECONSTRUCTED");
  await expect(diff.getByTestId("complete-diff-added")).toHaveText(/^\d+ GENERATED \(PROPOSED\)$/);
  await expect(diff).toContainText("not observed");
  await expect(diff).toContainText("not metres");

  const before = await head(request, id);
  await diff.getByRole("button", { name: "Accept as GENERATED" }).click();
  await expect(dialog.getByTestId("complete-notice")).toContainText("Accepted:");
  const picked = cards.and(dialog.locator('[aria-pressed="true"]'));
  await expect(picked.getByTestId("complete-badge")).toHaveText("ACCEPTED - GENERATED");
  await expect(diff.getByRole("button", { name: "Accept as GENERATED" })).toBeDisabled();
  expect(await head(request, id)).not.toBe(before);

  const wir = (await (await request.get(`${API_URL}/api/worlds/${id}/worldir`)).json()) as {
    entities: Record<string, { provenance: string; custom_properties?: Record<string, unknown> }>;
  };
  const generated = Object.values(wir.entities).filter((e) => e.custom_properties?.completion_proposal);
  expect(generated.length).toBeGreaterThan(0);
  expect(generated.every((e) => e.provenance === "GENERATED")).toBe(true);
  expect(wir.entities["floor"].provenance).toBe("RECONSTRUCTED");

  // Batch B reuses the name: Accept must surface the server's 409 text verbatim.
  await dialog.getByRole("button", { name: "Propose", exact: true }).click();
  await expect(diff.getByRole("button", { name: "Accept as GENERATED" })).toBeEnabled({ timeout: 60_000 });
  const afterA = await head(request, id);
  await diff.getByRole("button", { name: "Accept as GENERATED" }).click();
  await expect(dialog.getByTestId("complete-error")).toHaveText(`a generated object named '${name}' already exists in this world`);
  expect(await head(request, id)).toBe(afterA);

  // Reject: marked REJECTED, no new version.
  await diff.getByRole("button", { name: "Reject" }).click();
  await expect(dialog.getByTestId("complete-notice")).toContainText("Rejected");
  await expect(diff.getByRole("button", { name: "Reject" })).toBeDisabled();
  expect(await head(request, id)).toBe(afterA);
});
