/**
 * The canonical reconstruction-success rule, exercised against the REAL module
 * the UI imports (`lib/api/reconstruction.ts`) -- not a re-implementation.
 *
 * If this file's copy of the table and the module ever disagree, this test
 * fails, which is the entire point: the duplication is gone.
 */
import { registerHooks } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

// `lib/api/reconstruction.ts` imports only a type from ./types, so it loads
// with no React/Next runtime needed.
const modPath = pathToFileURL(
  fileURLToPath(new URL("../../src/lib/api/reconstruction.ts", import.meta.url)),
).href;
const { classifyReconstruction, isPersistedSuccess } = await import(modPath);

const V = "v-real-123";

const TABLE = [
  // [name, job, expected kind]
  ["succeeded + version", { status: "succeeded", payload: { version_id: V } }, "success"],
  ["succeeded + missing version", { status: "succeeded", payload: {} }, "failure"],
  ["partial + version", { status: "partial", payload: { version_id: V } }, "partial"],
  ["partial + missing version", { status: "partial", payload: {} }, "failure"],
  ["failed + version", { status: "failed", payload: { version_id: V } }, "failure"],
  ["cancelled + version", { status: "cancelled", payload: { version_id: V } }, "failure"],
  ["queued + version", { status: "queued", payload: { version_id: V } }, "pending"],
  ["running + version", { status: "running", payload: { version_id: V } }, "pending"],
  ["succeeded + missing payload", { status: "succeeded" }, "failure"],
  // Defensive cases beyond the spec table.
  ["queued, no version", { status: "queued", payload: {} }, "pending"],
  ["running, no version", { status: "running", payload: {} }, "pending"],
  ["unknown status is never success", { status: "weird", payload: { version_id: V } }, "failure"],
  ["empty status is never success", { status: "", payload: { version_id: V } }, "failure"],
  ["null job", null, "failure"],
  ["undefined job", undefined, "failure"],
  ["job with no status", { payload: { version_id: V } }, "failure"],
  // A non-string version_id must not count as present.
  ["numeric version_id is not a version", { status: "succeeded", payload: { version_id: 7 } }, "failure"],
  ["empty-string version_id is not a version", { status: "succeeded", payload: { version_id: "" } }, "failure"],
];

const violations = [];
const results = {};

for (const [name, job, expectedKind] of TABLE) {
  const verdict = classifyReconstruction(job);
  results[name] = { expectedKind, actualKind: verdict.kind, reason: verdict.reason };
  if (verdict.kind !== expectedKind) {
    violations.push(
      `reconstruction: ${name}: expected '${expectedKind}', got '${verdict.kind}'`,
    );
  }
}

// `isPersistedSuccess` must agree with the kind, and must never be true for a
// state that is not a proven persisted result.
for (const [name, job] of TABLE) {
  const verdict = classifyReconstruction(job);
  const persisted = isPersistedSuccess(verdict);
  const shouldBe = verdict.kind === "success" || verdict.kind === "partial";
  if (persisted !== shouldBe) {
    violations.push(`reconstruction: isPersistedSuccess disagrees for '${name}'`);
  }
}

// Spot-check the security-relevant invariants explicitly.
{
  const cancelled = classifyReconstruction({ status: "cancelled", payload: { version_id: V } });
  if (isPersistedSuccess(cancelled)) {
    violations.push("reconstruction: a cancelled job was reported as a persisted success");
  }
  const failed = classifyReconstruction({ status: "failed", payload: { version_id: V } });
  if (isPersistedSuccess(failed)) {
    violations.push("reconstruction: a failed job was reported as a persisted success");
  }
  const noVersion = classifyReconstruction({ status: "succeeded", payload: {} });
  if (isPersistedSuccess(noVersion)) {
    violations.push("reconstruction: a version-less success was reported as persisted");
  }
}

process.stdout.write(JSON.stringify({ violations, results }, null, 2));


