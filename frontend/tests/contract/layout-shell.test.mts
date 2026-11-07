/** A world built from one photo's learned layout must be drawn by Studio as a SOLID room shell, not legacy boxes. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildRoomShellParts } from "../../src/lib/viewport/room-shell-model.ts";
import { countEntities } from "./layout-shell-helpers.mts";

const world = JSON.parse(readFileSync(new URL("./fixtures/layout-world.json", import.meta.url), "utf8"));

test("the shell has floor, ceiling, walls and window parts (this is what flips hasShell on)", () => {
  const kinds = new Set(buildRoomShellParts(world).map((p) => p.kind));
  for (const k of ["floor", "ceiling", "wall", "window_glass"]) assert.ok(kinds.has(k as never), `missing ${k}: ${[...kinds]}`);
});

test("every part belongs to a real entity of the world", () => {
  for (const p of buildRoomShellParts(world)) assert.ok(world.entities[p.entityId], p.entityId);
});

test("the room summary counts the furniture (it read 0 before the furniture_candidate label)", () => {
  assert.ok(countEntities(world, "furniture_candidate") >= 1);
});
