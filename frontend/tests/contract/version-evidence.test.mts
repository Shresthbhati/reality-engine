/**
 * The per-version evidence panel: what an EARLIER version was built from.
 *
 * Pure logic (src/lib/api/version-evidence.ts) driven with the same shape the status API sends. The rule these pin:
 * a photograph's relation to an inspected version comes from THAT version's own record, and a version that did not
 * record its evidence is "unknown" -- never guessed from the current model.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { evidenceInVersion, versionEvidenceSummary } from "../../src/lib/api/version-evidence.ts";

const v2 = { label: "V2", evidence_ids: ["a", "b", "c"], registered_ids: ["a", "b"] };
const photos = ["a", "b", "c", "d", "e"].map((id) => ({ id }));

test("a photo is placed, waiting, or added after the inspected version", () => {
  assert.equal(evidenceInVersion("a", v2), "placed");
  assert.equal(evidenceInVersion("c", v2), "waiting"); // in the version, not placed by it
  assert.equal(evidenceInVersion("d", v2), "added_later"); // uploaded after V2 existed
});

test("a version that did not record its evidence is unknown, never guessed", () => {
  assert.equal(evidenceInVersion("a", { evidence_ids: null, registered_ids: null }), "unknown");
  assert.equal(evidenceInVersion("a", { evidence_ids: undefined, registered_ids: undefined }), "unknown");
  assert.equal(evidenceInVersion("a", null), "unknown");
});

test("a version with an evidence list but no placement list reports its photos as waiting, not placed", () => {
  assert.equal(evidenceInVersion("a", { evidence_ids: ["a"], registered_ids: null }), "waiting");
});

test("the summary counts used, placed, waiting and later photos", () => {
  assert.equal(versionEvidenceSummary(photos, v2), "V2: 3 photos used, 2 placed, 1 waiting · 2 added since");
  const only = { label: "V1", evidence_ids: ["a"], registered_ids: ["a"] };
  assert.equal(versionEvidenceSummary(photos, only), "V1: 1 photo used, 1 placed · 4 added since");
  const all = { label: "V3", evidence_ids: ["a", "b", "c", "d", "e"], registered_ids: ["a", "b", "c", "d", "e"] };
  assert.equal(versionEvidenceSummary(photos, all), "V3: 5 photos used, 5 placed");
});

test("the summary of an unrecorded version says so", () => {
  assert.equal(
    versionEvidenceSummary(photos, { label: "V1", evidence_ids: null, registered_ids: null }),
    "V1 did not record which photos it used.",
  );
});
