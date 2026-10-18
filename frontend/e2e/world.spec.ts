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
  const chosen = dialog.getByTestId("apartment-plan").and(dialog.locator('[aria-pressed="true"]'));
  const summary = (await chosen.innerText()).match(/(\d+) of (\d+) required adjacencies met/);
  expect(summary, "chosen plan card shows the adjacency summary").not.toBeNull();
  expect(Number(summary![2])).toBeGreaterThanOrEqual(4);
  expect(summary![1]).toBe(summary![2]);
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

test("Building tab: create against the live API -> appears in Generated -> edit storeys -> delete", async ({ page, request }) => {
  const name = `Tower${Date.now()}`;
  type Item = { kind: string; name: string; params: { storeys: number } };
  const generated = async () =>
    (await (await request.get(`${API_URL}/api/worlds/${worldId}/generated`)).json()) as Item[];

  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await dialog.getByRole("tab", { name: "building" }).click();
  await expect(dialog.getByTestId("building-provenance")).toHaveText("GENERATED");
  await dialog.getByLabel("Building name").fill(name);
  await dialog.getByLabel("Footprint width (m)").fill("12");
  await dialog.getByLabel("Footprint depth (m)").fill("10");
  await dialog.getByLabel("Storeys", { exact: true }).fill("2");
  await dialog.getByRole("button", { name: "Build building" }).click();
  await expect(dialog).toBeHidden();

  await expect.poll(async () => (await generated()).some((g) => g.kind === "building" && g.name === name)).toBe(true);

  await page.getByTestId("open-generated").click();
  const panel = page.getByRole("dialog", { name: "Generated" });
  const row = panel.getByTestId("generated-list").getByRole("button", { name: new RegExp(name) });
  await expect(row).toContainText("building");
  await expect(row).toContainText("2 storeys, 12 x 10 m");
  await row.click();
  const edit = panel.getByTestId("generated-edit");
  await edit.getByLabel("Storeys (1-12)").fill("3");
  await expect(edit.getByTestId("building-heights").getByRole("spinbutton")).toHaveCount(3);
  await edit.getByRole("button", { name: "Save changes" }).click();
  await expect(panel.getByTestId("generated-result")).toHaveText(/\d+ changed - \d+ unchanged - \d+ removed/);
  await expect.poll(async () => (await generated()).find((g) => g.name === name)?.params.storeys).toBe(3);

  await edit.getByRole("button", { name: "Delete", exact: true }).click();
  await edit.getByRole("button", { name: "Confirm delete" }).click();
  await expect.poll(async () => (await generated()).some((g) => g.name === name)).toBe(false);
});

test("Building tab: a server 422 is shown verbatim, never as an empty state", async ({ page }) => {
  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await dialog.getByRole("tab", { name: "building" }).click();
  await dialog.getByLabel("Building name").fill(`Squat${Date.now()}`);
  // Passes the client's early checks (0 < side <= 100) but is below the grammar's 4 m minimum.
  await dialog.getByLabel("Footprint width (m)").fill("3");
  await dialog.getByLabel("Footprint depth (m)").fill("3");
  await dialog.getByRole("button", { name: "Build building" }).click();
  await expect(dialog.getByRole("alert")).toContainText("footprint sides must be greater than 4 m");
  await expect(dialog).toBeVisible();
});

test("District tab: create against the live API -> appears in Generated -> edit detail policy -> delete", async ({ page, request }) => {
  const name = `Town${Date.now()}`;
  type Item = { kind: string; name: string; params: { detail: { max_full: number | null } } };
  const generated = async () =>
    (await (await request.get(`${API_URL}/api/worlds/${worldId}/generated`)).json()) as Item[];

  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await dialog.getByRole("tab", { name: "district" }).click();
  await expect(dialog.getByTestId("district-provenance")).toHaveText("GENERATED");
  await dialog.getByLabel("District name").fill(name);
  await dialog.getByLabel("Parcels east-west").fill("2");
  await dialog.getByLabel("Parcels north-south").fill("2");
  await dialog.getByLabel("Fully detailed buildings").fill("1");
  await expect(dialog.getByTestId("district-note")).toContainText("4 parcels");
  await expect(dialog.getByTestId("district-note")).toContainText("6 named streets");
  await dialog.getByRole("button", { name: "Build district" }).click();
  await expect(dialog).toBeHidden();

  await expect.poll(async () => (await generated()).some((g) => g.kind === "district" && g.name === name)).toBe(true);

  await page.getByTestId("open-generated").click();
  const panel = page.getByRole("dialog", { name: "Generated" });
  const row = panel.getByTestId("generated-list").getByRole("button", { name: new RegExp(name) });
  await expect(row).toContainText("district");
  await expect(row).toContainText("grid, 2 x 2 parcels, lot 24 x 24 m");
  await row.click();
  const edit = panel.getByTestId("generated-edit");
  await edit.getByLabel("Fully detailed buildings (empty = all)").fill("2");
  await edit.getByRole("button", { name: "Save changes" }).click();
  await expect(panel.getByTestId("generated-result")).toHaveText(/\d+ changed - \d+ unchanged - \d+ removed/);
  await expect.poll(async () => (await generated()).find((g) => g.name === name)?.params.detail.max_full).toBe(2);

  await edit.getByRole("button", { name: "Delete", exact: true }).click();
  await edit.getByRole("button", { name: "Confirm delete" }).click();
  await expect.poll(async () => (await generated()).some((g) => g.name === name)).toBe(false);
});

test("District tab: a server 422 is shown verbatim, never as an empty state", async ({ page }) => {
  await page.goto(WORLD);
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await dialog.getByRole("tab", { name: "district" }).click();
  await dialog.getByLabel("District name").fill(`Bad${Date.now()}`);
  // Passes the client's early checks (lot 8-200 m) but the 3 m setbacks leave a 2 m building footprint.
  await dialog.getByLabel("Lot size (m)").fill("8");
  await dialog.getByRole("button", { name: "Build district" }).click();
  await expect(dialog.getByRole("alert")).toContainText("footprint sides must be greater than 4 m");
  await expect(dialog).toBeVisible();
});
