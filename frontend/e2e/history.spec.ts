import { test, expect } from "@playwright/test";
import { API_URL, PRO_MODE, seed } from "./support";

test.use({ storageState: PRO_MODE });

const { worldId } = seed();

/** T-6.7 wired: the Undo/Redo buttons change the SAVED world (not only the screen), and the history survives a reload. */
test("Undo and Redo move the saved world, and survive a page reload", async ({ page, request }) => {
  const name = `Undo${Date.now()}`;
  const id = `${name}-room`;
  const entities = async () =>
    Object.keys(((await (await request.get(`${API_URL}/api/worlds/${worldId}/worldir`)).json()) as { entities: Record<string, unknown> }).entities);

  await page.goto(`/worlds/${worldId}`);
  await page.getByTestId("open-actions").click();
  await page.getByTestId("open-build").click();
  const dialog = page.getByRole("dialog", { name: "Build" });
  await dialog.getByRole("tab", { name: "room" }).click();
  await dialog.locator("input").first().fill(name);
  await dialog.getByRole("button", { name: "Build room" }).click();
  await expect(dialog).toBeHidden();
  await expect.poll(entities).toContain(id);

  const undo = page.getByRole("button", { name: "Undo" });
  const redo = page.getByRole("button", { name: "Redo" });
  await expect(undo).toBeEnabled();

  await undo.click();
  await expect.poll(entities).not.toContain(id);              // the saved world changed, not just the view
  await expect(redo).toBeEnabled();

  await page.reload();                                         // nothing in the browser remembers the history
  await expect(redo).toBeEnabled();
  await redo.click();
  await expect.poll(entities).toContain(id);
});
