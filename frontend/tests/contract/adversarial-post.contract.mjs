/**
 * Adversarial contract harness for every Next.js POST proxy route.
 *
 * The GET sweep (`adversarial-routes.contract.mjs`) covers reads. This covers
 * WRITES, where the stakes differ: a POST that fabricates success tells the
 * operator a correction was persisted, a version was committed, or a
 * reconstruction was enqueued when the backend never did it.
 *
 * Invariants asserted for every route and every failure mode:
 *   1. A transport failure never answers 2xx.
 *   2. A malformed or empty 2xx body never becomes a fabricated success.
 *   3. 404 stays distinguishable from a transport failure (503).
 *   4. 409 (stale parent / duplicate / already-in-progress) stays 409.
 *   5. Backend 5xx stay 5xx.
 *   6. A malformed request body is rejected with 4xx, never forwarded.
 *
 * Usage:  node tests/contract/adversarial-post.contract.mjs --violations
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

function stubFetch(behaviour) {
  globalThis.fetch = async (url, init) => behaviour(String(url), init ?? {});
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** A Request-alike good enough for routes that only read .json(). */
function makeRequest(body) {
  return {
    url: "http://localhost:3000/api/probe",
    json: async () => {
      if (body === "__MALFORMED__") throw new SyntaxError("Unexpected token");
      return body;
    },
  };
}

async function probe(POST, behaviour, params, body) {
  stubFetch(behaviour);
  try {
    const response = await POST(makeRequest(body), { params: Promise.resolve(params) });
    const text = await response.text();
    let parsed = null;
    try {
      parsed = text ? JSON.parse(text) : null;
    } catch {
      parsed = null;
    }
    return { status: response.status, body: parsed, rawLength: text.length };
  } catch (error) {
    return { threw: true, message: String(error?.message ?? error) };
  } finally {
    globalThis.fetch = realFetch;
  }
}


/**
 * Every POST proxy route, its path params, a valid request body, and the keys
 * a truthful success response MUST contain.
 *
 * `resultKeys` marks routes where a 2xx lacking those keys would falsely read
 * as "persisted" in the UI.
 */
const POST_ROUTES = [
  {
    file: "sessions/[sessionId]/reconstruct/route.ts",
    params: { sessionId: "s-1" },
    body: undefined,
    resultKeys: ["job_id", "status"],
    successStatus: 200,
  },
  {
    file: "worlds/[id]/commit/route.ts",
    params: { id: "w-1" },
    body: { entity_id: "e-1", changes: { name: "Wall A" } },
    resultKeys: ["version_id", "world_id"],
    successStatus: 200,
  },
  {
    file: "worlds/route.ts",
    params: {},
    body: { name: "New World" },
    resultKeys: ["id", "name"],
    // The backend serves POST /api/worlds with status_code=201.
    successStatus: 201,
  },
  {
    file: "jobs/ingest/route.ts",
    params: {},
    body: { folder: "d", output: "o" },
    resultKeys: [],
    successStatus: 200,
  },
  {
    file: "jobs/compile/route.ts",
    params: {},
    body: { dataset_path: "d" },
    resultKeys: [],
    successStatus: 200,
  },
  {
    file: "sessions/mobile-tasks/route.ts",
    params: {},
    body: { bundle_path: "b", output_path: "o" },
    resultKeys: [],
    successStatus: 200,
  },
  {
    file: "sessions/ingest-mobile/route.ts",
    params: {},
    body: { bundle_path: "b", session_dir: "s" },
    resultKeys: [],
    successStatus: 200,
  },
  {
    file: "jobs/[jobId]/cancel/route.ts",
    params: { jobId: "job-1" },
    body: undefined,
    resultKeys: ["id"],
    // Cancel answers 200 (queued, stopped) or 202 (running, winding down).
    successStatus: 200,
  },
];

/** Failure modes shared by every POST route. */
const FAILURE_MODES = {
  backend400: async () => jsonResponse(400, { detail: "bad request" }),
  backend404: async () => jsonResponse(404, { detail: "Session not found" }),
  backend409: async () => jsonResponse(409, { detail: "Reconstruction already in progress" }),
  backend422: async () => jsonResponse(422, { detail: "validation failed" }),
  backend500: async () => jsonResponse(500, { detail: "database is locked" }),
  backend502: async () => new Response("<html>502 Bad Gateway</html>", { status: 502 }),
  backend503: async () => jsonResponse(503, { detail: "worker offline" }),
  // A 2xx whose body is not JSON: a broken or intercepted proxy.
  malformedSuccess: async () =>
    new Response("{not json at all", {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  // A 2xx with a completely empty body.
  emptySuccess: async () => new Response("", { status: 200 }),
  timeout: async () => {
    throw Object.assign(new Error("The operation was aborted due to timeout"), {
      name: "TimeoutError",
    });
  },
  unreachable: async () => {
    throw new TypeError("fetch failed");
  },
};

/**
 * A success response that carries every required key.
 *
 * Returned as a zero-arg closure: the harness invokes the fetch stub as
 * `behaviour(url, init)`, so a plain `(route) => ...` function would receive
 * the URL string instead of the route descriptor.
 */
function successFor(route) {
  return async () => {
    const base = { detail: "ok" };
    for (const key of route.resultKeys) base[key] = `ok-${key}`;
    return jsonResponse(200, base);
  };
}


const results = {};
const violations = [];

function record(route, scenario, message) {
  violations.push(`${route} :: ${scenario} :: ${message}`);
}

for (const route of POST_ROUTES) {
  // fileURLToPath keeps `[`/`]` literal; a manual pathname decode would
  // percent-encode them and miss the real route file.
  const fsPath = fileURLToPath(new URL(`../../src/app/api/${route.file}`, import.meta.url));
  const url = pathToFileURL(fsPath).href;

  let mod;
  try {
    mod = await import(url);
  } catch (error) {
    record(route.file, "import", `import failed -- ${error?.message ?? error}`);
    continue;
  }
  const POST = mod.POST;
  if (typeof POST !== "function") {
    record(route.file, "export", "no POST export");
    continue;
  }

  const entry = {};

  // --- happy path: a real success must be forwarded intact ------------------
  entry.success = await probe(POST, successFor(route), route.params, route.body);
  if (entry.success.threw) {
    record(route.file, "success", `threw -- ${entry.success.message}`);
  } else if (entry.success.status !== route.successStatus) {
    record(
      route.file,
      "success",
      `expected ${route.successStatus}, got ${entry.success.status}`,
    );
  } else {
    for (const key of route.resultKeys) {
      if (!entry.success.body || entry.success.body[key] === undefined) {
        record(route.file, "success", `success body lost required key '${key}'`);
      }
    }
  }

  // --- malformed request body: must be 4xx, never forwarded -----------------
  const malformedReq = await probe(
    POST,
    async () => jsonResponse(200, { ok: true }),
    route.params,
    "__MALFORMED__",
  );
  entry.malformedRequest = malformedReq;
  if (!malformedReq.threw && malformedReq.status < 400) {
    record(
      route.file,
      "malformedRequest",
      `malformed JSON body was accepted (status ${malformedReq.status})`,
    );
  }

  // --- every failure mode --------------------------------------------------
  for (const [name, behaviour] of Object.entries(FAILURE_MODES)) {
    const outcome = await probe(POST, behaviour, route.params, route.body);
    entry[name] = outcome;

    if (outcome.threw) {
      record(route.file, name, `route threw -- ${outcome.message}`);
      continue;
    }

    const transport = name === "unreachable" || name === "timeout";
    const isTwoXx = outcome.status >= 200 && outcome.status < 300;

    if (isTwoXx) {
      if (name === "malformedSuccess" || name === "emptySuccess") {
        // A 2xx with an unusable body must not masquerade as a real success.
        if (route.resultKeys.length > 0) {
          const missing = route.resultKeys.filter(
            (k) => !outcome.body || outcome.body[k] === undefined,
          );
          if (missing.length > 0) {
            record(
              route.file,
              name,
              `2xx with no usable body returned as success; missing ` +
                `${missing.join(", ")} -- the UI would show a write that never happened`,
            );
          }
        }
      } else {
        record(route.file, name, `failure answered 2xx (status ${outcome.status})`);
      }
      continue;
    }

    if (name === "backend404" && outcome.status !== 404) {
      record(route.file, name, `404 was rewritten to ${outcome.status}`);
    }
    if (transport && outcome.status === 404) {
      record(
        route.file,
        name,
        "transport failure reported as 404 (claims backend state it never observed)",
      );
    }
    if (transport && outcome.status !== 503 && outcome.status !== 502) {
      record(
        route.file,
        name,
        `transport failure answered ${outcome.status}; expected 503/502 so it stays ` +
          "distinguishable from a real 404",
      );
    }
    if (name === "backend409" && outcome.status !== 409) {
      record(
        route.file,
        name,
        `409 became ${outcome.status}; a conflict must never look like success`,
      );
    }
    if (name === "backend500" && (outcome.status < 500 || outcome.status >= 600)) {
      record(route.file, name, `500 became ${outcome.status}`);
    }
  }

  results[route.file] = entry;
}

if (process.argv.includes("--violations")) {
  process.stdout.write(violations.length ? violations.join("\n") + "\n" : "");
} else {
  process.stdout.write(JSON.stringify({ violations, results }, null, 2));
}

