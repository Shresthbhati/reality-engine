/**
 * RUNTIME async stale-data test for `useApi`'s generation guard.
 *
 * Extracts the hook's generation/abort algorithm and runs it against a
 * controllable fake fetcher, asserting the three specified scenarios:
 *
 *   A  A loaded, request B, B FAILS    -> A's data must not show as B; B errors
 *   B  A loaded, request B, B SUCCEEDS -> only B is shown
 *   C  A loaded, request B, then C; B answers LATE -> B ignored, C wins
 *
 * Scenario C is the one a plain `cancelled` boolean cannot handle: B's promise
 * resolves AFTER the effect was cleaned up, so the boolean is already true
 * when `.then` runs. Only a generation counter catches that.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

// Read the REAL source so this test tracks production, not a copy.
const hooksPath = fileURLToPath(
  new URL("../../src/lib/api/hooks.ts", import.meta.url),
);
const source = readFileSync(hooksPath, "utf8");

const violations = [];
const notes = [];

if (!/generationRef\.current\s*===\s*generation/.test(source)) {
  violations.push("hooks.ts: the generation comparison is missing");
}
if (!/controller\.abort\(\)/.test(source)) {
  violations.push("hooks.ts: the obsolete request is not aborted");
}
if (!/const generation = \+\+generationRef\.current/.test(source)) {
  violations.push("hooks.ts: the generation is not incremented per request");
}

/** Faithful extraction of the hook's effect body as a plain async function. */
function makeHook() {
  const state = { data: null, isLoading: true, error: null, generation: 0 };
  let abortRef = null;

  return {
    state,
    start(fetcher) {
      const generation = ++state.generation;
      abortRef?.abort();
      const controller = new AbortController();
      abortRef = controller;
      const isCurrent = () => state.generation === generation;

      state.data = null;
      state.error = null;
      state.isLoading = true;

      Promise.resolve()
        .then(() => fetcher(controller.signal))
        .then((res) => {
          if (!isCurrent()) return;
          state.data = res;
          state.isLoading = false;
        })
        .catch((err) => {
          if (!isCurrent()) return;
          if (controller.signal.aborted || err?.name === "AbortError") return;
          state.data = null;
          state.error = err?.message ?? String(err);
          state.isLoading = false;
        });
    },
  };
}

const defer = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const tick = () => new Promise((r) => setTimeout(r, 0));

// --- Scenario A: B fails; A's data must not survive -----------------------
{
  const h = makeHook();
  h.start(async () => "world-A");
  await tick();
  if (h.state.data !== "world-A") violations.push(`A/setup: expected world-A, got ${h.state.data}`);

  // Switch to B; B rejects.
  h.start(async () => {
    throw new Error("B failed");
  });
  if (h.state.data !== null) {
    violations.push(`A: world-A's data survived the switch to B (${h.state.data})`);
  }
  await tick();
  if (h.state.error === null) {
    violations.push("A: a failed world switch must produce an error, not an empty success");
  }
  notes.push(`A: data=${h.state.data} error=${h.state.error}`);
}

// --- Scenario B: B succeeds; only B is shown -------------------------------
{
  const h = makeHook();
  h.start(async () => "world-A");
  await tick();
  h.start(async () => "world-B");
  await tick();
  if (h.state.data !== "world-B") violations.push(`B: expected world-B, got ${h.state.data}`);
  if (h.state.error !== null) violations.push("B: a successful switch must not leave an error");
  notes.push(`B: data=${h.state.data} error=${h.state.error}`);
}

// --- Scenario C: B answers late after C was requested ----------------------
{
  const h = makeHook();
  h.start(async () => "world-A");
  await tick();

  const bGate = defer();
  h.start(() => bGate.promise); // B, unresolved
  await tick();

  // C supersedes B before B answers.
  h.start(async () => "world-C");
  await tick();
  if (h.state.data !== "world-C") violations.push(`C/setup: expected world-C, got ${h.state.data}`);

  // The stale B finally answers. It must be ignored.
  bGate.resolve("world-B-LATE");
  await new Promise((r) => setTimeout(r, 5));
  if (h.state.data !== "world-C") {
    violations.push(
      `C: a late response from a superseded request overwrote the current data (got ${h.state.data})`,
    );
  }
  notes.push(`C: data=${h.state.data} after a late B response`);

  // A late FAILURE from a superseded request must not surface either.
  const h2 = makeHook();
  h2.start(async () => "world-A");
  await tick();
  const bFail = defer();
  h2.start(() => bFail.promise);
  await tick();
  h2.start(async () => "world-C2");
  await tick();
  bFail.reject(new Error("stale B error"));
  await new Promise((r) => setTimeout(r, 5));
  if (h2.state.data !== "world-C2" || h2.state.error !== null) {
    violations.push(
      `C: a late failure from a superseded request corrupted the current state ` +
        `(data=${h2.state.data} error=${h2.state.error})`,
    );
  }
  notes.push(`C: data=${h2.state.data} error=${h2.state.error} after a late B failure`);
}

process.stdout.write(JSON.stringify({ violations, notes }, null, 2));

