import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { test, expect, type APIRequestContext } from "@playwright/test";
import { API_URL, watchErrors } from "./support";

/**
 * Phase G acceptance journey (UI + real API): photos -> reconstruction -> build -> scoped edit -> completion ->
 * assistant -> measure -> export -> reload -> diff -> delete with dependents.
 *
 * Nothing is mocked. Photos are the real South Building set and go through the real COLMAP/MiDaS engines; if the
 * engines or the dataset are missing, the two reconstruction steps are SKIPPED with the reason (never faked).
 * The backend twin of this test is tests/acceptance/test_journey.py, which also proves a true process restart
 * (here "restart" is a fresh browser context re-reading what the server persisted).
 *
 * Steps are recorded, a table is printed, and any FAIL fails the test.
 */

const root = path.resolve(__dirname, "..", "..");
const sandbox = path.join(__dirname, "..", "test-results", "e2e");
const IMAGES = path.join(root, "datasets", "south_building", "images");
// Six strongly overlapping views (indices into the name-sorted list; see tests/integration/test_progressive_product_journey.py).
const SIX = fs.existsSync(IMAGES)
  ? fs.readdirSync(IMAGES).filter((f) => /\.jpg$/i.test(f)).sort().slice(15, 21).map((f) => path.join(IMAGES, f))
  : [];

type Wir = {
  entities: Record<string, { type: string; provenance: string; geometry_ids: string[]; custom_properties?: Record<string, unknown> }>;
  geometries: Record<string, { bounds_min: { x: number }; bounds_max: { x: number } }>;
};
type Row = { n: number; title: string; status: "PASS" | "SKIPPED" | "FAIL"; note: string };

class Skip extends Error {}

// Without this an action waits for the whole test timeout; a failing step must fail fast and be recorded.
test.use({ actionTimeout: 30_000 });

const get = async <T>(request: APIRequestContext, url: string): Promise<T> => {
  const r = await request.get(`${API_URL}${url}`);
  expect(r.ok(), `GET ${url} -> ${r.status()}`).toBeTruthy();
  return (await r.json()) as T;
};
const wirOf = (request: APIRequestContext, id: string) => get<Wir>(request, `/api/worlds/${id}/worldir`);
const headOf = async (request: APIRequestContext, id: string) =>
  (await get<{ current_version_id: string }>(request, `/api/worlds/${id}`)).current_version_id;
const fp = (w: Wir, e: string) => JSON.stringify([w.entities[e], w.entities[e].geometry_ids.map((g) => w.geometries[g])]);
const owner = (w: Wir, e: string) => w.entities[e].custom_properties?.generated_by as string | undefined;
const provCount = (w: Wir) => {
  const c: Record<string, number> = {};
  for (const e of Object.values(w.entities)) c[e.provenance] = (c[e.provenance] ?? 0) + 1;
  return c;
};

test("acceptance journey: photos -> model -> build -> edit -> complete -> assist -> measure -> export -> reload -> diff -> delete", async ({
  page,
  browser,
  request,
}, testInfo) => {
  test.setTimeout(20 * 60_000);
  const bag = watchErrors(page);
  const rows: Row[] = [];
  const S: Record<string, any> = {};

  async function step(n: number, title: string, fn: () => Promise<string | void>) {
    const row: Row = { n, title, status: "PASS", note: "" };
    rows.push(row);
    const t = Date.now();
    try {
      row.note = (await fn()) ?? "";
    } catch (e) {
      if (e instanceof Skip) {
        row.status = "SKIPPED";
        row.note = e.message;
      } else {
        row.status = "FAIL";
        row.note = String((e as Error).message).split("\n").slice(0, 4).join(" ").slice(0, 400);
      }
    }
    row.note += ` [${((Date.now() - t) / 1000).toFixed(1)}s]`;
    console.log(`step ${n} ${row.status}: ${row.note}`);
  }

  // Can this machine really reconstruct? (the same probe the API exposes to the Studio)
  const runtime = (await get<{ runtime: { name: string; status: string }[] }>(request, "/api/system/verification")).runtime;
  const up = (prefix: string) => runtime.some((a) => a.name.startsWith(prefix) && a.status === "AVAILABLE");
  const optedOut = !!process.env.JOURNEY_SKIP_RECON; // fast local iteration only; the real run never sets it
  const canReconstruct = !optedOut && SIX.length === 6 && up("colmap") && up("torch");
  const whyNot = optedOut ? "JOURNEY_SKIP_RECON is set" : SIX.length < 6 ? "South Building photo set not present" : `real engines unavailable (colmap=${up("colmap")}, torch=${up("torch")})`;

  // 1 ------------------------------------------------------------------------------------------------------------
  await step(1, "create world", async () => {
    const w = await request.post(`${API_URL}/api/worlds`, { data: { name: `Journey ${Date.now()}` } });
    expect(w.status()).toBe(201);
    S.id = ((await w.json()) as { id: string }).id;
    expect(await headOf(request, S.id)).toBeNull();
    await page.goto(`/worlds/${S.id}`);
    await expect(page.locator("canvas").first()).toBeVisible();
    return "empty world opens in the studio";
  });
  const id: string = S.id;
  const WORLD = `/worlds/${id}`;

  // 2 + 3 --------------------------------------------------------------------------------------------------------
  await step(2, "ingest evidence through the UI (6 real photos)", async () => {
    if (!canReconstruct) throw new Skip(whyNot);
    await page.getByTestId("add-photos").click();
    const dialog = page.getByRole("dialog", { name: "Add photographs" });
    await dialog.getByTestId("photo-input").setInputFiles(SIX);
    await expect(dialog).toBeHidden({ timeout: 60_000 }).catch(() => undefined);
    await expect
      .poll(async () => (await get<{ evidence: unknown[] }>(request, `/api/worlds/${id}/status`)).evidence.length, { timeout: 60_000 })
      .toBe(6);
    return "6 photos accepted by POST /api/reconstructions";
  });

  await step(3, "reconstruct: stage status -> WorldIR", async () => {
    if (!canReconstruct) throw new Skip(whyNot);
    await page.goto(WORLD);
    await expect(page.getByTestId("model-state")).toContainText("6/6 photos placed", { timeout: 8 * 60_000 });
    const st = await get<{ state: string; model: { level: number; model_state: string; images_registered: number; images_used: number; scale: { state: string }; dense?: { state: string } } }>(
      request, `/api/worlds/${id}/status`);
    expect(["READY_TO_INSPECT", "PARTIALLY_COMPLETE"]).toContain(st.state);
    expect(st.model.images_registered).toBe(6);
    await expect(page.getByTestId("version-V1")).toBeVisible();
    const wir = await wirOf(request, id);
    const walls = Object.keys(wir.entities).filter((e) => wir.entities[e].type === "wall");
    expect(walls.length).toBeGreaterThanOrEqual(3);
    S.reconIds = Object.keys(wir.entities);
    S.reconFp = Object.fromEntries(S.reconIds.map((e: string) => [e, fp(wir, e)]));
    S.reconHead = await headOf(request, id);
    const ply = await request.get(`${API_URL}/api/worlds/${id}/points`);
    expect(ply.ok()).toBeTruthy();
    return `state=${st.state} level=${st.model.level} model=${st.model.model_state} scale=${st.model.scale.state} dense=${st.model.dense?.state} walls=${walls.length}`;
  });

  // 4 ------------------------------------------------------------------------------------------------------------
  await step(4, "build room + building (UI), corridor + apartment + district (API)", async () => {
    await page.goto(WORLD);
    await page.getByTestId("open-build").click();
    let build = page.getByRole("dialog", { name: "Build" });
    await build.getByRole("tab", { name: "room" }).click();
    await build.locator("input").first().fill("Study");
    await build.getByLabel("Width (m)").fill("4");
    await build.getByRole("button", { name: "Build room" }).click();
    await expect(build).toBeHidden();
    await expect.poll(async () => "Study-room" in (await wirOf(request, id)).entities).toBe(true);

    await page.getByTestId("open-build").click();
    build = page.getByRole("dialog", { name: "Build" });
    await build.getByRole("tab", { name: "building" }).click();
    await build.getByLabel("Building name").fill("Tower");
    await build.getByLabel("Footprint width (m)").fill("12");
    await build.getByLabel("Footprint depth (m)").fill("10");
    await build.getByLabel("Storeys", { exact: true }).fill("2");
    await build.getByRole("button", { name: "Build building" }).click();
    await expect(build).toBeHidden();
    await expect.poll(async () => "Tower-building" in (await wirOf(request, id)).entities).toBe(true);

    const corridor = await request.post(`${API_URL}/api/worlds/${id}/corridors`, {
      data: { name: "Hall", length: 4, attach: { to: "Study-room", side: "east" } },
    });
    expect(corridor.status(), await corridor.text()).toBe(201);
    const apartment = await request.post(`${API_URL}/api/worlds/${id}/apartments`, {
      data: {
        name: "flat", footprint: [10, 8], entrance: "hall", seed: 1, plan_index: 0, origin: [30, 0, 0],
        rooms: [
          { id: "living", kind: "living", min_area: 18, max_area: 30 },
          { id: "kitchen", kind: "kitchen", min_area: 8, max_area: 14 },
          { id: "bedroom_1", kind: "bedroom", min_area: 12, max_area: 18 },
          { id: "bath_1", kind: "bathroom", min_area: 4, max_area: 7 },
          { id: "hall", kind: "hallway", min_area: 6, max_area: 12, aspect_max: 3.5 },
        ],
        adjacency: [["living", "kitchen", "required"], ["living", "hall", "required"], ["hall", "bedroom_1", "required"], ["bedroom_1", "bath_1", "preferred"]],
      },
    });
    expect(apartment.status(), await apartment.text()).toBe(201);
    const district = await request.post(`${API_URL}/api/worlds/${id}/districts`, {
      data: { name: "dist", blocks_x: 3, blocks_y: 2, seed: 5, detail: { max_full: 1 }, origin: [120, 0, 0] },
    });
    expect(district.status(), await district.text()).toBe(201);

    const wir = await wirOf(request, id);
    const gen = (await get<{ entity_id: string }[]>(request, `/api/worlds/${id}/generated`)).map((g) => g.entity_id).sort();
    expect(gen).toEqual(["Hall-corridor", "Study-room", "Tower-building", "dist-district", "flat-apartment"].sort());
    if (canReconstruct) for (const e of S.reconIds) expect(fp(wir, e), `reconstructed ${e} disturbed by building`).toBe(S.reconFp[e]);
    return `${Object.keys(wir.entities).length} entities, provenance ${JSON.stringify(provCount(wir))}`;
  });

  // 5 ------------------------------------------------------------------------------------------------------------
  await step(5, "scoped edit in the Generated panel: unrelated entities byte-identical", async () => {
    const before = await wirOf(request, id);
    await page.goto(WORLD);
    await page.getByTestId("open-generated").click();
    const panel = page.getByRole("dialog", { name: "Generated" });
    await panel.getByTestId("generated-list").getByRole("button", { name: /Study/ }).filter({ hasNotText: /corridor/ }).first().click();
    const edit = panel.getByTestId("generated-edit");
    await edit.getByLabel("Width (m)").fill("6");
    await edit.getByRole("button", { name: "Save changes" }).click();
    await expect(panel.getByTestId("generated-result")).toHaveText(/\d+ changed - \d+ unchanged - \d+ removed/);
    let after = before;
    await expect
      .poll(async () => {
        after = await wirOf(request, id);
        const g = after.geometries[after.entities["Study-floor"].geometry_ids[0]];
        return g.bounds_max.x - g.bounds_min.x;
      })
      .toBeCloseTo(6, 3);
    const changed = Object.keys(after.entities).filter((e) => fp(after, e) !== fp(before, e));
    expect(changed.length).toBeGreaterThan(0);
    for (const e of changed) expect(["Study-room", "Hall-corridor"], `${e} changed but is not Study/Hall`).toContain(owner(after, e));
    const rest = Object.keys(after.entities).filter((e) => !changed.includes(e));
    expect(rest.length).toBeGreaterThan(changed.length);
    for (const o of ["flat-apartment", "Tower-building", "dist-district"]) expect(rest.some((e) => owner(after, e) === o)).toBe(true);
    if (canReconstruct) for (const e of S.reconIds) expect(rest).toContain(e);
    return `changed=${changed.length} (Study+Hall), byte-identical=${rest.length}`;
  });

  // 6 ------------------------------------------------------------------------------------------------------------
  await step(6, "completion: refused on photo shell; propose -> accept on a RECONSTRUCTED shell", async () => {
    const notes: string[] = [];
    if (canReconstruct) {
      await page.goto(WORLD);
      await page.getByTestId("open-complete").click();
      const d = page.getByRole("dialog", { name: "Complete interior" });
      await d.getByRole("button", { name: "Propose", exact: true }).click();
      await expect(d.getByTestId("complete-error")).toContainText("no usable shell");
      notes.push("photo world honestly refused");
      await d.getByRole("button", { name: "Close" }).first().click().catch(() => undefined);
    }
    // A facade photo set yields INFERRED walls only; the accept path runs on a seeded RECONSTRUCTED shell
    // (seed_shell.py commits it through the API's own write path, exactly like completion.spec.ts).
    const w = await request.post(`${API_URL}/api/worlds`, { data: { name: `Journey shell ${Date.now()}` } });
    const shellId = ((await w.json()) as { id: string }).id;
    execFileSync("python", [path.join(__dirname, "seed_shell.py"), shellId, "6", "5"], {
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
    const head0 = await headOf(request, shellId);
    await page.goto(`/worlds/${shellId}`);
    await page.getByTestId("open-complete").click();
    const dialog = page.getByRole("dialog", { name: "Complete interior" });
    await dialog.getByLabel("Interior name").fill("jflat");
    await dialog.getByRole("button", { name: "Propose", exact: true }).click();
    const cards = dialog.getByTestId("complete-proposal");
    await expect(cards.first()).toBeVisible({ timeout: 60_000 });
    await expect(cards.first().getByTestId("complete-badge")).toHaveText("PROPOSED");
    expect(await headOf(request, shellId), "proposing must not move HEAD").toBe(head0);
    const diff = dialog.getByTestId("complete-diff");
    await expect(diff.getByTestId("complete-diff-counts")).toContainText(/^\d+ added, 0 changed, 0 removed$/);
    await diff.getByRole("button", { name: "Accept as GENERATED" }).click();
    await expect(dialog.getByTestId("complete-notice")).toContainText("Accepted:");
    expect(await headOf(request, shellId)).not.toBe(head0);
    const wir = await wirOf(request, shellId);
    const made = Object.values(wir.entities).filter((e) => e.custom_properties?.completion_proposal);
    expect(made.length).toBeGreaterThan(0);
    expect(made.every((e) => e.provenance === "GENERATED")).toBe(true);
    expect(wir.entities["floor"].provenance).toBe("RECONSTRUCTED");
    notes.push(`accepted ${made.length} GENERATED entities on the seeded shell`);
    return notes.join("; ");
  });

  // 7 ------------------------------------------------------------------------------------------------------------
  await step(7, "assistant: propose_edit -> confirm (API; no assistant UI exists yet)", async () => {
    const head = await headOf(request, id);
    const p = await request.post(`${API_URL}/api/worlds/${id}/assistant/call`, {
      data: { tool: "propose_edit", args: { operation: { op: "update_generated", kind: "room", name: "Study", patch: { height: 3.2 } } } },
    });
    expect(p.status(), await p.text()).toBe(200);
    const plan = ((await p.json()) as { result: { plan_id: string; requires_confirmation: boolean; preview: { entities_changed: number } } }).result;
    expect(plan.requires_confirmation).toBe(true);
    expect(plan.preview.entities_changed).toBeGreaterThan(0);
    expect(await headOf(request, id), "a proposal must not move HEAD").toBe(head);
    const c = await request.post(`${API_URL}/api/worlds/${id}/assistant/confirm`, { data: { plan_id: plan.plan_id } });
    expect(c.status(), await c.text()).toBe(200);
    expect(await headOf(request, id)).not.toBe(head);
    const again = await request.post(`${API_URL}/api/worlds/${id}/assistant/confirm`, { data: { plan_id: plan.plan_id } });
    expect(again.status()).toBe(409);
    return `plan changed ${plan.preview.entities_changed} entities; replay -> 409`;
  });

  // 8 ------------------------------------------------------------------------------------------------------------
  await step(8, "measure: authored exact + unknown-scale honesty (API)", async () => {
    const w = await wirOf(request, id);
    const g = w.geometries[w.entities["Study-floor"].geometry_ids[0]] as unknown as { bounds_min: { x: number; y: number; z: number }; bounds_max: { x: number } };
    const m = await request.post(`${API_URL}/api/worlds/${id}/measurements`, {
      data: {
        kind: "distance", scale_state: "authored", method: "Study floor x-extent",
        points: [[g.bounds_min.x, g.bounds_min.y, g.bounds_min.z], [g.bounds_max.x, g.bounds_min.y, g.bounds_min.z]],
      },
    });
    expect(m.status(), await m.text()).toBe(201);
    const a = (await m.json()) as { value: number; unit: string; uncertainty: number };
    expect(a.value).toBeCloseTo(6, 9);
    expect(a.unit).toBe("m");
    expect(a.uncertainty).toBe(0);
    const u = await request.post(`${API_URL}/api/worlds/${id}/measurements`, {
      data: { kind: "distance", scale_state: "unknown", method: "two picked points", points: [[0, 0, 0], [3, 4, 0]] },
    });
    const unknown = (await u.json()) as { value: number; unit: string; uncertainty: number | null };
    expect(unknown.value).toBe(5);
    expect(unknown.unit).not.toBe("m");
    expect(unknown.uncertainty).toBeNull();
    S.nMeasurements = (await get<unknown[]>(request, `/api/worlds/${id}/measurements`)).length;
    expect(S.nMeasurements).toBe(2);
    return `authored ${a.value} ${a.unit}; unknown-scale unit=${unknown.unit}`;
  });

  // 9 ------------------------------------------------------------------------------------------------------------
  await step(9, "export from the UI: Godot bundle with a valid GLB", async () => {
    await page.goto(WORLD);
    // retry the click until React has hydrated (same pattern as wizard.spec.ts)
    await expect(async () => {
      if (!(await page.getByRole("heading", { name: "Export World" }).isVisible())) await page.getByTitle("Export World").click();
      await expect(page.getByRole("heading", { name: "Export World" })).toBeVisible({ timeout: 2000 });
    }).toPass({ timeout: 30_000 });
    const panel = page.getByRole("heading", { name: "Export World" }).locator("xpath=ancestor::div[contains(@class,'absolute')][1]");
    await panel.getByLabel("Format").selectOption("godot");
    await panel.getByRole("button", { name: "Export", exact: true }).click();
    const link = panel.getByRole("link", { name: /Download godot/ });
    await expect(link).toBeVisible({ timeout: 60_000 });
    await expect(panel).toContainText(/\d+ entities exported/);
    const href = (await link.getAttribute("href"))!;
    const dl = await request.get(href);
    expect(dl.status()).toBe(200);
    const bundle = (await dl.json()) as Record<string, { encoding: string; content: string }>;
    expect(Object.keys(bundle).sort()).toEqual(["manifest.json", "scene.tscn", "world.glb"]);
    const glb = Buffer.from(bundle["world.glb"].content, "base64");
    expect(glb.subarray(0, 4).toString("latin1")).toBe("glTF");
    expect(glb.readUInt32LE(4)).toBe(2);
    expect(glb.readUInt32LE(8)).toBe(glb.length);
    const gltf = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString("utf8"));
    expect(gltf.meshes.length).toBeGreaterThan(0);
    return `glb ${glb.length}B, ${gltf.meshes.length} meshes`;
  });

  // 10 -----------------------------------------------------------------------------------------------------------
  const pre = await wirOf(request, id);
  const preHead = await headOf(request, id);
  await step(10, "reload in a fresh browser context: state persisted (backend test does the process restart)", async () => {
    const ctx = await browser.newContext();
    try {
      const p2 = await ctx.newPage();
      const bag2 = watchErrors(p2);
      await p2.goto(WORLD);
      await expect(p2.locator("canvas").first()).toBeVisible();
      if (canReconstruct) await expect(p2.getByTestId("model-state")).toContainText("6/6 photos placed", { timeout: 60_000 });
      await p2.getByTestId("open-generated").click();
      const list = p2.getByRole("dialog", { name: "Generated" }).getByTestId("generated-list");
      for (const n of ["Study", "Hall", "Tower", "flat", "dist"]) await expect(list.getByRole("button", { name: new RegExp(n) }).first()).toBeVisible();
      expect(bag2.errors, bag2.errors.join("\n")).toEqual([]);
    } finally {
      await ctx.close();
    }
    const post = await wirOf(request, id);
    expect(await headOf(request, id)).toBe(preHead);
    expect(Object.keys(post.entities).length).toBe(Object.keys(pre.entities).length);
    expect(provCount(post)).toEqual(provCount(pre));
    return `HEAD=${preHead} entities=${Object.keys(pre.entities).length} provenance=${JSON.stringify(provCount(pre))}`;
  });

  // 11 -----------------------------------------------------------------------------------------------------------
  await step(11, "diff first -> last version (API)", async () => {
    const versions = (await get<{ items: { id: string; parent_version_id: string | null }[] }>(request, `/api/worlds/${id}/versions`)).items;
    const byId = new Map(versions.map((v) => [v.id, v]));
    const chain: string[] = [];
    for (let cur: string | null = preHead; cur; cur = byId.get(cur)!.parent_version_id) chain.unshift(cur);
    expect(chain.length).toBeGreaterThan(2);
    const first = chain[0];
    const last = chain[chain.length - 1];
    const firstWir = await get<Wir>(request, `/api/worlds/${id}/worldir?version=${first}`);
    const d = await get<{ categories: Record<string, number>; entity_diffs: { entity_id: string; kind: string }[] }>(
      request, `/api/worlds/${id}/diff?base=${first}&head=${last}`);
    const added = new Set(d.entity_diffs.filter((x) => x.kind === "added").map((x) => x.entity_id));
    const expected = Object.keys(pre.entities).filter((e) => !(e in firstWir.entities));
    expect([...added].sort()).toEqual(expected.sort());
    expect(expected.length).toBeGreaterThan(0);
    expect(d.entity_diffs.filter((x) => x.kind === "removed")).toEqual([]);
    if (canReconstruct) {
      expect(first).toBe(S.reconHead);
      expect(d.entity_diffs.filter((x) => x.kind === "modified").map((x) => x.entity_id).filter((e) => S.reconIds.includes(e))).toEqual([]);
    }
    const same = await get<{ categories: Record<string, number> }>(request, `/api/worlds/${id}/diff?base=${last}&head=${last}`);
    expect(Object.values(same.categories).reduce((a, b) => a + b, 0)).toBe(0);
    return `${chain.length} versions; +${added.size} added`;
  });

  // 12 -----------------------------------------------------------------------------------------------------------
  await step(12, "delete with dependents (UI): blocked -> cascade", async () => {
    const head = await headOf(request, id);
    await page.goto(WORLD);
    await page.getByTestId("open-generated").click();
    const panel = page.getByRole("dialog", { name: "Generated" });
    await panel.getByTestId("generated-list").getByRole("button", { name: /Study/ }).filter({ hasNotText: /corridor/ }).first().click();
    const edit = panel.getByTestId("generated-edit");
    await edit.getByRole("button", { name: "Delete", exact: true }).click();
    await edit.getByRole("button", { name: "Confirm delete" }).click();
    const deps = panel.getByTestId("delete-dependents");
    await expect(deps).toContainText("Hall");
    expect(await headOf(request, id), "a refused delete must not move HEAD").toBe(head);
    await deps.getByRole("button", { name: "Delete with dependents" }).click();
    await deps.getByRole("button", { name: "Confirm delete all" }).click();
    await expect.poll(async () => "Study-room" in (await wirOf(request, id)).entities).toBe(false);
    const after = await wirOf(request, id);
    expect("Hall-corridor" in after.entities).toBe(false);
    for (const o of ["flat-apartment", "Tower-building", "dist-district"]) expect(Object.keys(after.entities).some((e) => owner(after, e) === o)).toBe(true);
    if (canReconstruct) for (const e of S.reconIds) expect(fp(after, e)).toBe(S.reconFp[e]);
    return "Study blocked by Hall; cascade removed both; other objects and reconstruction untouched";
  });

  const table = [
    "",
    "ACCEPTANCE JOURNEY (UI)",
    ...rows.map((r) => `${String(r.n).padStart(2)}. ${r.title.padEnd(86)} ${r.status === "SKIPPED" ? `SKIPPED(${r.note.replace(/ \[[\d.]+s\]$/, "")})` : `${r.status}  ${r.note}`}`),
    "",
  ].join("\n");
  console.log(table);
  await testInfo.attach("journey-table", { body: table, contentType: "text/plain" });
  expect(rows.filter((r) => r.status === "FAIL").map((r) => `${r.n} ${r.title}: ${r.note}`)).toEqual([]);
  expect(bag.errors, bag.errors.join("\n")).toEqual([]);
});
