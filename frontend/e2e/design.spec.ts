import path from "node:path";
import { execFileSync } from "node:child_process";
import { test, expect, type APIRequestContext } from "@playwright/test";
import { API_URL, PRO_MODE } from "./support";

test.use({ storageState: PRO_MODE });

const root = path.resolve(__dirname, "..", "..");
const sandbox = path.join(__dirname, "..", "test-results", "e2e");

/** A fresh world whose head is a metric room with one sofa (see seed_furniture.py for why this is not via a public API). */
async function roomWithSofa(request: APIRequestContext): Promise<string> {
  const w = await request.post(`${API_URL}/api/worlds`, { data: { name: `Design ${Date.now()}` } });
  expect(w.ok()).toBeTruthy();
  const { id } = (await w.json()) as { id: string };
  execFileSync("python", [path.join(__dirname, "seed_furniture.py"), id], {
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

type Wir = { entities: Record<string, { name: string; transform?: { position: { x: number; z: number } }; custom_properties?: Record<string, unknown> }> };

/** T-4b.1 in the real Studio: the designer selects a piece, turns it, swaps it, deletes it; every step is saved and Undo reverses it. */
test("Turn, swap and delete a sofa; Undo brings it back", async ({ page, request }) => {
  const wid = await roomWithSofa(request);
  const wir = async () => (await (await request.get(`${API_URL}/api/worlds/${wid}/worldir`)).json()) as Wir;

  await page.goto(`/worlds/${wid}`);
  await page.getByRole("button", { name: "Expand Furniture (1)" }).click();
  await page.getByRole("tree", { name: "Scene" }).getByRole("button", { name: /^Sofa/ }).first().click();
  const panel = page.getByTestId("furniture-design");
  await expect(panel).toBeVisible();

  await panel.getByRole("button", { name: "Turn right 90 degrees" }).click();
  await expect.poll(async () => (await wir()).entities.sofa.custom_properties?.yaw_rad).toBeCloseTo(Math.PI / 2, 3);

  await expect(panel.getByLabel("Turn angle")).toHaveValue("90");        // the panel reloaded with the saved numbers
  await panel.getByLabel("Swap for a catalog piece").selectOption("armchair");
  await panel.getByRole("button", { name: "Swap" }).click();
  await expect.poll(async () => (await wir()).entities.sofa.name).toBe("Armchair");

  await panel.getByRole("button", { name: "Delete piece" }).click();
  await panel.getByRole("button", { name: "Delete", exact: true }).click();
  await expect.poll(async () => Object.keys((await wir()).entities)).not.toContain("sofa");

  await page.getByRole("button", { name: "Undo" }).first().click();      // the header's Undo; the delete is undone, the armchair is back
  await expect.poll(async () => (await wir()).entities.sofa?.name).toBe("Armchair");
});
