/**
 * A duplicate commit retry persisted nothing; the Studio must not present it as a new version.
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { commitOutcome } from "../../src/lib/api/commit-outcome.ts";

test("a fresh commit is 'created' and keeps its parent", () => {
  assert.deepEqual(commitOutcome({ version_id: "v2", parent_version_id: "v1", duplicate: false }), {
    kind: "created", versionId: "v2", parentVersionId: "v1",
  });
});

test("a duplicate retry is 'duplicate', names the current version, and carries no parent to diff against", () => {
  const out = commitOutcome({ version_id: "v2", parent_version_id: "v2", duplicate: true });
  assert.deepEqual(out, { kind: "duplicate", versionId: "v2" });
  assert.equal("parentVersionId" in out, false);
});

test("an older server without the field is treated as a created version", () => {
  assert.equal(commitOutcome({ version_id: "v3", parent_version_id: "v2" }).kind, "created");
});
