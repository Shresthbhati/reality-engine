import path from "node:path";
import { execFileSync } from "node:child_process";
import { test, expect, type APIRequestContext } from "@playwright/test";
import { API_URL, PRO_MODE } from "./support";

test.use({ storageState: PRO_MODE });

const root = path.resolve(__dirname, "..", "..");
const sandbox = path.join(__dirname, "..", "test-results", "e2e");

async function roomWithSofa(request: APIRequestContext): Promise<string> {
  const w = await request.post(`${API_URL}/api/worlds`, { data: { name: `Finish ${Date.now()}` } });
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

type Wir = { entities: Record<string, { custom_properties?: Record<string, unknown> }> };

/** T-4b.2 in the real Studio: paint the wall and choose a floor material; each is saved, the survey colour stays, Undo reverses it. */
test("Paint a wall, pick a floor material, Undo", async ({ page, request }, testInfo) => {
  const wid = await roomWithSofa(request);
  const wir = async () => (await (await request.get(`${API_URL}/api/worlds/${wid}/worldir`)).json()) as Wir;
  const finishOf = async (id: string) => (await wir()).entities[id].custom_properties?.design_finish as { colour_rgb: number[]; paint?: string; material?: string } | undefined;

  await page.goto(`/worlds/${wid}`);
  await page.getByRole("button", { name: "Expand Walls (1)" }).click();
  await page.getByRole("tree", { name: "Scene" }).getByRole("button", { name: /wall-n$/ }).first().click();
  const panel = page.getByTestId("surface-finish");
  await expect(panel).toBeVisible();
  await expect(panel.getByTestId("finish-current")).toHaveText("Survey colour (no finish applied)");

  await panel.getByRole("button", { name: "Navy" }).click();
  await expect.poll(async () => (await finishOf("wall-n"))?.paint).toBe("navy");
  await expect(panel.getByTestId("finish-current")).toHaveText("Navy (paint)");
  expect((await wir()).entities["wall-n"].custom_properties?.mean_color_rgb).toEqual([200, 190, 180]);   // the survey colour is untouched
  await page.screenshot({ path: testInfo.outputPath("painted-wall.png") });

  await page.getByRole("tree", { name: "Scene" }).getByRole("button", { name: /^Floor/ }).first().click();
  const floor = page.getByTestId("surface-finish");
  await expect(floor.getByText("Floor material")).toBeVisible();
  await floor.getByRole("button", { name: "Walnut" }).click();
  await expect.poll(async () => (await finishOf("floor"))?.material).toBe("walnut");
  await expect(page.getByText("Initializing 3D spatial viewport")).toBeHidden();      // the scene has been rebuilt with the finishes
  await page.waitForTimeout(800);
  await page.screenshot({ path: testInfo.outputPath("finished-room.png") });

  await page.getByRole("button", { name: "Undo" }).first().click();             // undo the floor material
  await expect.poll(async () => await finishOf("floor")).toBeUndefined();
  expect((await finishOf("wall-n"))?.paint).toBe("navy");                          // the wall paint is a separate, earlier edit
});
