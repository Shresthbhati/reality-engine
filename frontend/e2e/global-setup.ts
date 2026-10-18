import fs from "node:fs";
import path from "node:path";
import { request } from "@playwright/test";
import { API_URL, SEED_FILE } from "../playwright.config";

/**
 * Creates this run's own world through the API. Tests never assume an empty DB: the API may be an
 * already-running instance with other data in it.
 */
export default async function globalSetup() {
  const api = await request.newContext({ baseURL: API_URL });
  const name = `E2E Smoke World ${Date.now()}`;
  const w = await api.post("/api/worlds", { data: { name } });
  if (!w.ok()) throw new Error(`POST /api/worlds failed: ${w.status()} ${await w.text()}`);
  const { id } = (await w.json()) as { id: string };
  const room = await api.post(`/api/worlds/${id}/rooms`, {
    data: { name: "Study", width: 4, depth: 5, height: 2.8, openings: [], seed: 7, origin: [0, 0, 0] },
  });
  if (room.status() !== 201) throw new Error(`POST rooms failed: ${room.status()} ${await room.text()}`);
  const city = await api.post(`/api/worlds/${id}/cities`, {
    data: { name: "Town", blocks_x: 2, blocks_y: 2, origin: [40, 0, 0] },
  });
  if (city.status() !== 201) throw new Error(`POST cities failed: ${city.status()} ${await city.text()}`);
  fs.mkdirSync(path.dirname(SEED_FILE), { recursive: true });
  fs.writeFileSync(SEED_FILE, JSON.stringify({ worldId: id, worldName: name }));
  await api.dispose();
}
