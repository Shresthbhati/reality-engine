import { test, expect } from "@playwright/test";
import { API_URL, seed } from "./support";

const { worldId } = seed();
const WORLD = `/worlds/${worldId}`;

test("seeded world renders the 3D canvas", async ({ page }) => {
  await page.goto(WORLD);
  await expect(page.locator("canvas").first()).toBeVisible();
});

test("Build dialog creates a room that lands in the WorldIR", async ({ page, request }) => {
  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("tab", { name: "room" }).click();
  await dialog.locator("input").first().fill("Annex");
  await dialog.getByRole("button", { name: "Build room" }).click();
  await expect(dialog).toBeHidden();

  await expect
    .poll(async () => {
      const wir = (await (await request.get(`${API_URL}/api/worlds/${worldId}/worldir`)).json()) as {
        entities: Record<string, unknown>;
      };
      return Object.keys(wir.entities);
    })
    .toContain("Annex-room");
});

test("Generated panel: widening a built room changes only that room's floor extent", async ({ page, request }) => {
  type Geom = { bounds_min: { x: number }; bounds_max: { x: number } };
  type Wir = {
    entities: Record<string, { geometry_ids: string[]; custom_properties?: Record<string, unknown> }>;
    geometries: Record<string, Geom>;
  };
  const getWir = async () => (await (await request.get(`${API_URL}/api/worlds/${worldId}/worldir`)).json()) as Wir;
  const name = `Resize${Date.now()}`;
  const id = `${name}-room`;

  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const build = page.getByRole("dialog", { name: "Build" });
  await build.getByRole("tab", { name: "room" }).click();
  await build.locator("input").first().fill(name);
  await build.getByLabel("Width (m)").fill("4");
  await build.getByRole("button", { name: "Build room" }).click();
  await expect(build).toBeHidden();

  await expect.poll(async () => id in (await getWir()).entities).toBe(true);
  const before = await getWir();
  const floorX = (w: Wir) => {
    const g = w.geometries[w.entities[`${name}-floor`].geometry_ids[0]];
    return g.bounds_max.x - g.bounds_min.x;
  };
  const owned = (w: Wir) => new Set(Object.keys(w.entities).filter((e) => w.entities[e].custom_properties?.generated_by === id));
  const rest = (w: Wir) => {
    const own = owned(w);
    const ownGeoms = new Set([...own].flatMap((e) => w.entities[e].geometry_ids));
    return JSON.stringify({
      entities: Object.fromEntries(Object.entries(w.entities).filter(([e]) => !own.has(e))),
      geometries: Object.fromEntries(Object.entries(w.geometries).filter(([g]) => !ownGeoms.has(g))),
    });
  };
  const xBefore = floorX(before);

  await page.getByTestId("open-generated").click();
  const panel = page.getByRole("dialog", { name: "Generated" });
  await panel.getByTestId("generated-list").getByRole("button", { name: new RegExp(name) }).click();
  const edit = panel.getByTestId("generated-edit");
  await edit.getByLabel("Width (m)").fill("6");
  await edit.getByRole("button", { name: "Save changes" }).click();
  await expect(panel.getByTestId("generated-result")).toHaveText(/\d+ changed - \d+ unchanged - \d+ removed/);

  await expect.poll(async () => floorX(await getWir())).not.toBe(xBefore);
  const after = await getWir();
  expect(floorX(after)).toBeCloseTo(xBefore + 2, 3);
  expect(rest(after)).toBe(rest(before));
});

test("Measure panel opens: Distance enabled, Volume disabled", async ({ page }) => {
  await page.goto(WORLD);
  await page.getByRole("button", { name: "Toggle measurement mode" }).click();
  const panel = page.getByLabel("Measurement mode", { exact: true });
  await expect(panel).toBeVisible();
  await expect(panel.getByRole("button", { name: "Distance" })).toBeEnabled();
  await expect(panel.getByRole("button", { name: "Volume" })).toBeDisabled();
});

test("Geospatial Map tab says the local frame is not georeferenced", async ({ page }) => {
  await page.goto(WORLD);
  await page.getByRole("button", { name: "Geospatial Map" }).click();
  await expect(page.getByText(/not georeferenced/i).first()).toBeVisible();
});

test("Apartment tab: 2-bedroom template -> Find layouts -> Build creates a building and rooms", async ({ page, request }) => {
  const name = `Flat${Date.now()}`;
  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await dialog.getByRole("tab", { name: "apartment" }).click();
  await dialog.getByLabel("Apartment name").fill(name);
  await dialog.getByLabel("Apartment template").selectOption({ label: "2-bedroom" });
  await dialog.getByRole("button", { name: "Find layouts" }).click();
  await expect(dialog.getByTestId("apartment-plan").first()).toBeVisible({ timeout: 60_000 });
  await expect(dialog.getByRole("alert")).toHaveCount(0);
  await dialog.getByRole("button", { name: "Build apartment" }).click();
  await expect(dialog).toBeHidden();

  await expect
    .poll(async () => {
      const wir = (await (await request.get(`${API_URL}/api/worlds/${worldId}/worldir`)).json()) as {
        entities: Record<string, { type: string }>;
      };
      const mine = Object.entries(wir.entities).filter(([id]) => id.startsWith(`${name}-`));
      const count = (t: string) => mine.filter(([, e]) => e.type === t).length;
      return { building: count("building") >= 1, rooms: count("room") >= 5 };
    })
    .toEqual({ building: true, rooms: true });
});
