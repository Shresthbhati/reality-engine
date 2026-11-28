/**
 * Cancel contract harness — PR #126 semantics.
 *
 * `POST /api/jobs/{job_id}/cancel` (apps/api/routes_jobs.py) distinguishes four
 * outcomes, and the UI must not collapse them:
 *
 *   queued job   -> 200 { id, status: "cancelled" }   already stopped
 *   running job  -> 202 { id, status, cancel_requested: true }  NOT yet cancelled
 *   terminal job -> 409  (already succeeded/partial/failed/cancelled)
 *   unknown id   -> 404
 *
 * A 202 is a request acknowledgement. Displaying "cancelled" on a 202 would
 * claim a terminal state the worker has not committed -- the job may still
 * finish the stage it is in before it observes the flag.
 *
 * Usage:  node tests/contract/cancel-contract.contract.mjs
 */
import { registerHooks } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const resolveHook = {
  resolve(specifier, context, nextResolve) {
    if (specifier === "next/server") {
      return { url: new URL("./next-server-stub.mjs", import.meta.url).href, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
};

if (typeof registerHooks === "function") {
  registerHooks(resolveHook);
} else {
  const { register } = await import("node:module");
  register("./stub-next-server.mjs", import.meta.url);
}

const realFetch = globalThis.fetch;

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function probeCancel(POST, behaviour) {
  globalThis.fetch = async (url, init) => behaviour(String(url), init ?? {});
  try {
    const res = await POST(
      { url: "http://localhost:3000/api/probe", json: async () => ({}) },
      { params: Promise.resolve({ jobId: "job-1" }) },
    );
    const text = await res.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = null;
    }
    return { status: res.status, body };
  } catch (error) {
    return { threw: true, message: String(error?.message ?? error) };
  } finally {
    globalThis.fetch = realFetch;
  }
}

const fsPath = fileURLToPath(
  new URL("../../src/app/api/jobs/[jobId]/cancel/route.ts", import.meta.url),
);
const { POST } = await import(pathToFileURL(fsPath).href);

const violations = [];
const observations = {};

/** Mirrors `cancelJob` in lib/api/platform.ts. */
function interpret(observed) {
  // 409 (already terminal) and 404 (unknown job) are both definite backend
  // answers that the cancellation did not happen -- distinct from an unknown
  // outcome caused by a transport failure.
  if (observed.status === 409 || observed.status === 404) return "rejected";
  if (observed.status === 202 || observed.body?.cancel_requested === true) return "cancelling";
  if (observed.status === 200) return "cancelled";
  return "error";
}

const CASES = [
  {
    name: "queued job is cancelled immediately",
    backend: () => jsonResponse(200, { id: "job-1", status: "cancelled" }),
    expectStatus: 200,
    expectOutcome: "cancelled",
  },
  {
    name: "running job is still winding down (202 is NOT cancelled)",
    backend: () =>
      jsonResponse(202, { id: "job-1", status: "running", cancel_requested: true }),
    expectStatus: 202,
    expectOutcome: "cancelling",
  },
  {
    name: "terminal job cannot be cancelled",
    backend: () =>
      jsonResponse(409, { detail: "Job already succeeded; only queued/running jobs can be cancelled" }),
    expectStatus: 409,
    expectOutcome: "rejected",
  },
  {
    name: "unknown job is 404, not 503",
    backend: () => jsonResponse(404, { detail: "Job not found" }),
    expectStatus: 404,
    expectOutcome: "rejected",
  },
  {
    // cancel/timeout race: the request never completed, so the job's fate is
    // unknown -- the UI must not claim it was cancelled.
    name: "cancel/timeout race leaves state unknown",
    backend: async () => {
      throw Object.assign(new Error("aborted due to timeout"), { name: "TimeoutError" });
    },
    expectStatus: 503,
    expectOutcome: "error",
  },
  {
    name: "unreachable backend leaves state unknown",
    backend: async () => {
      throw new TypeError("fetch failed");
    },
    expectStatus: 503,
    expectOutcome: "error",
  },
  {
    // cancel/success race: the job won before the cancel landed.
    name: "cancel/success race is reported as already finished",
    backend: () =>
      jsonResponse(409, { detail: "Job already succeeded; only queued/running jobs can be cancelled" }),
    expectStatus: 409,
    expectOutcome: "rejected",
  },
  {
    name: "2xx with no job id cannot be tracked",
    backend: () => jsonResponse(200, { status: "cancelled" }),
    expectStatus: 502,
    expectOutcome: "error",
  },
];

for (const testCase of CASES) {
  const observed = await probeCancel(POST, testCase.backend);
  observations[testCase.name] = observed;
  if (observed.threw) {
    violations.push(`${testCase.name}: route threw -- ${observed.message}`);
    continue;
  }
  if (observed.status !== testCase.expectStatus) {
    violations.push(
      `${testCase.name}: expected HTTP ${testCase.expectStatus}, got ${observed.status}`,
    );
  }
  const outcome = interpret(observed);
  if (outcome !== testCase.expectOutcome) {
    violations.push(
      `${testCase.name}: expected outcome '${testCase.expectOutcome}', got '${outcome}'`,
    );
  }
  // The critical invariant: a 202 must never be presented as cancelled.
  if (observed.status === 202 && outcome === "cancelled") {
    violations.push(`${testCase.name}: a 202 was interpreted as 'cancelled'`);
  }
}

process.stdout.write(JSON.stringify({ violations, observations }, null, 2));
