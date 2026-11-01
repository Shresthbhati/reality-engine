import { test } from "node:test";
import assert from "node:assert/strict";
import { buildIdentityEvidence } from "../../src/lib/viewport/identity-model.ts";

test("a merge across several photos is reported with the photo count", () => {
  const r = buildIdentityEvidence({ association: { resolver: "association", observations: 5, images: 5, decision: "SAME_ENTITY_CONFIRMED", ambiguous_with: [] } });
  assert.equal(r.state, "merged");
  assert.equal(r.photoCount, 5);
  assert.equal(r.text, "Same object in 5 photos");
});

test("an ambiguous decision is never presented as settled", () => {
  const r = buildIdentityEvidence({ association: { images: 2, decision: "AMBIGUOUS", ambiguous_with: ["ent-chair-0001", "ent-chair-0002"] } });
  assert.equal(r.state, "ambiguous");
  assert.deepEqual(r.ambiguousWith, ["ent-chair-0001", "ent-chair-0002"]);
  assert.match(r.text, /unresolved/);
});

test("ambiguity wins even when the decision label is missing", () => {
  assert.equal(buildIdentityEvidence({ association: { images: 3, ambiguous_with: ["ent-x"] } }).state, "ambiguous");
});

test("one photo is a single sighting, not a merge", () => {
  const r = buildIdentityEvidence({ association: { images: 1, decision: "DIFFERENT_ENTITY" } });
  assert.equal(r.state, "single");
  assert.equal(r.text, "Seen in 1 photo");
});

test("missing or malformed records are 'not recorded', never a confident merge", () => {
  for (const cp of [undefined, {}, { association: null }, { association: "x" }, { association: { images: "5" } }, { association: { images: -2 } }]) {
    const r = buildIdentityEvidence(cp as Record<string, unknown> | undefined);
    assert.equal(r.state, "not_recorded");
    assert.equal(r.text, "Identity not recorded");
  }
});
