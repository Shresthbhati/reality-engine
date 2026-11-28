/**
 * Adversarial contract harness for EVERY Next.js proxy route under
 * `src/app/api`.
 *
 * `sessions-proxy.contract.mjs` pins one route. This one sweeps the whole GET
 * surface and asserts the invariant that matters most:
 *
 *   A FAILURE MUST NEVER LOOK LIKE A VALID EMPTY SUCCESS.
 *
 * For every route and every failure mode (backend 5xx, malformed body, thrown
 * transport error) it asserts the route never answers 2xx, and specifically
 * that a transport failure is never laundered into 404 -- "not found" is a
 * claim about backend state that an unreachable backend cannot support, and a
 * 500 is not evidence that a resource is missing.
 *
 * The success path is pinned too: a 2xx backend response must be forwarded
 * with its real status and body, not replaced by an invented envelope.
 *
 * Usage:  node tests/contract/adversarial-routes.contract.mjs
 *         node tests/contract/adversarial-routes.contract.mjs --violations
 *           (prints only invariant violations; empty output means clean)
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
  globalThis.fetch = async (url, options) => behaviour(String(url), options ?? {});
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function probe(handler, behaviour, params) {
  stubFetch(behaviour);
  try {
    const response = await handler({ url: "http://localhost:3000/api/probe" }, { params });
    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    return { status: response.status, body, rawLength: text.length };
  } catch (error) {
    return { threw: true, message: String(error?.message ?? error) };
  } finally {
    globalThis.fetch = realFetch;
  }
}


/**
 * Every GET proxy route, with the params shape each one destructures.
 *
 * `binary: true` routes stream raw artifact bytes (PLY point clouds, evidence
 * images) rather than JSON. For those, forwarding an unparseable body verbatim
 * is CORRECT -- the proxy has no JSON contract to enforce -- so the
 * "malformed body" failure mode is not a violation for them. They are still
 * held to the 5xx/transport rules.
 */
const GET_ROUTES = [
  { file: "sessions/route.ts", params: {} },
  { file: "worlds/route.ts", params: {} },
  { file: "jobs/[jobId]/route.ts", params: { jobId: "job-1" } },
  { file: "worlds/[id]/versions/route.ts", params: { id: "w-1" } },
  { file: "worlds/[id]/worldir/route.ts", params: { id: "w-1" } },
  { file: "worlds/[id]/cameras/route.ts", params: { id: "w-1" } },
  { file: "worlds/[id]/report/route.ts", params: { id: "w-1" } },
  { file: "worlds/[id]/points/route.ts", params: { id: "w-1" }, binary: true },
  { file: "worlds/[id]/diff/route.ts", params: { id: "w-1" } },
  { file: "worlds/[id]/space-graph/route.ts", params: { id: "w-1" } },
  {
    file: "worlds/[id]/entities/[entityId]/provenance/route.ts",
    params: { id: "w-1", entityId: "e-1" },
  },
  {
    file: "worlds/[id]/evidence/[evidenceId]/image/route.ts",
    params: { id: "w-1", evidenceId: "ev-1" },
    binary: true,
  },
];

const FAILURE_MODES = {
  backend500: async () => jsonResponse(500, { detail: "database is locked" }),
  backend502: async () => new Response("<html>502 Bad Gateway</html>", { status: 502 }),
  backend503: async () => jsonResponse(503, { detail: "worker offline" }),
  // A 200 whose body is not JSON at all: a broken/intercepted proxy.
  malformed: async () =>
    new Response("{not json at all", {
      status: 200,
      headers: { "content-type": "application/json" },
    }),
  // AbortSignal.timeout rejects from fetch.
  timeout: async () => {
    throw Object.assign(new Error("The operation was aborted due to timeout"), {
      name: "TimeoutError",
    });
  },
  // Connection refused / DNS failure surfaces as a TypeError.
  unreachable: async () => {
    throw new TypeError("fetch failed");
  },
};


const results = {};
const violations = [];

for (const route of GET_ROUTES) {
  // fileURLToPath keeps `[`/`]` literal; a manual pathname decode would
  // percent-encode them and miss the real route file.
  const fsPath = fileURLToPath(new URL(`../../src/app/api/${route.file}`, import.meta.url));
  const url = pathToFileURL(fsPath).href;

  let mod;
  try {
    mod = await import(url);
  } catch (error) {
    violations.push(`${route.file}: import failed -- ${error?.message ?? error}`);
    continue;
  }
  const GET = mod.GET;
  if (typeof GET !== "function") {
    violations.push(`${route.file}: no GET export`);
    continue;
  }

  const entry = {};
  // Success must be forwarded with its real status, not re-enveloped.
  entry.success = await probe(
    GET,
    async () => jsonResponse(200, { items: [{ id: "x" }], cameras: [], detail: "ok" }),
    Promise.resolve(route.params),
  );
  if (entry.success.threw) {
    violations.push(`${route.file}: success path threw -- ${entry.success.message}`);
  } else if (entry.success.status !== 200) {
    violations.push(
      `${route.file}: success path returned ${entry.success.status}, expected 200`,
    );
  }

  for (const [name, behaviour] of Object.entries(FAILURE_MODES)) {
    const outcome = await probe(GET, behaviour, Promise.resolve(route.params));
    entry[name] = outcome;

    if (outcome.threw) {
      violations.push(
        `${route.file}/${name}: route threw instead of returning an error response`,
      );
      continue;
    }
    if (outcome.status >= 200 && outcome.status < 300) {
      // Binary routes stream bytes; an unparseable body is expected there.
      if (name === "malformed" && route.binary) continue;
      violations.push(
        `${route.file}/${name}: failure answered 2xx (status ${outcome.status})`,
      );
      continue;
    }
    if ((name === "unreachable" || name === "timeout") && outcome.status === 404) {
      violations.push(
        `${route.file}/${name}: transport failure reported as 404 ` +
          "(claims backend state it never observed)",
      );
    }
    if (name === "backend500" && outcome.status === 404) {
      violations.push(`${route.file}/backend500: 500 was laundered into 404 "not found"`);
    }
  }
  results[route.file] = entry;
}

if (process.argv.includes("--violations")) {
  process.stdout.write(violations.length ? violations.join("\n") + "\n" : "");
} else {
  process.stdout.write(JSON.stringify({ violations, results }, null, 2));
}
