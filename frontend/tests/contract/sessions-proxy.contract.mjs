/**
 * Contract harness for the Next.js `GET /api/sessions` proxy.
 *
 * Runs the real route module (src/app/api/sessions/route.ts) against a stubbed
 * backend and prints one JSON document describing the observed behaviour of
 * each scenario. The pytest wrapper
 * (tests/test_frontend_sessions_proxy_contract.py) asserts on that document, so
 * the route is exercised rather than merely grepped.
 *
 * The contract under test:
 *   - success            -> the backend body verbatim, i.e. { items: [...] }
 *   - backend 4xx/5xx    -> that status and body; never a 200 session list
 *   - transport failure  -> 503; never an empty success
 *
 * The historical bug this pins: on any failure the route answered 200 with
 * { error, sessions: [], count: 0 }, so an API outage rendered as "this user
 * has no sessions" -- a fabricated, plausible-looking empty state.
 */
import { registerHooks } from "node:module";

/**
 * Resolve `next/server` to the local stub. `registerHooks` is the current API;
 * `register` is kept as a fallback for older Node runtimes.
 */
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

const ROUTE = new URL("../../src/app/api/sessions/route.ts", import.meta.url).href;

const realFetch = globalThis.fetch;

/** Requests the harness actually received, for asserting the forwarded query. */
const seen = [];

function stubFetch(behaviour) {
  seen.length = 0;
  globalThis.fetch = async (url, options) => {
    seen.push({ url: String(url), init: options ?? {} });
    return behaviour(String(url), options);
  };
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function probe(GET, behaviour, search = "") {
  stubFetch(behaviour);
  const request = { url: `http://localhost:3000/api/sessions${search}` };
  let outcome;
  try {
    const response = await GET(request);
    const text = await response.text();
    let body = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
    outcome = { status: response.status, body, raw: text };
  } catch (error) {
    // The route must never throw: an unhandled throw would surface to the
    // browser as a 500 with no explanation of which layer failed.
    outcome = { threw: true, message: String(error && error.message) };
  } finally {
    globalThis.fetch = realFetch;
  }
  return { ...outcome, requests: seen.map((s) => s.url) };
}

const { GET } = await import(ROUTE);

const results = {
  success: await probe(
    GET,
    async () => jsonResponse(200, { items: [{ id: "s-1", name: "Kitchen sweep" }] }),
  ),
  successWithFilter: await probe(
    GET,
    async () => jsonResponse(200, { items: [{ id: "s-2" }] }),
    "?world_id=w-1",
  ),
  backendClientError: await probe(
    GET,
    async () => jsonResponse(422, { detail: "world_id is not a valid identifier" }),
  ),
  backendServerError: await probe(
    GET,
    async () => jsonResponse(500, { detail: "database is locked" }),
  ),
  backendUnparseableError: await probe(
    GET,
    async () => new Response("<html>502 Bad Gateway</html>", { status: 502 }),
  ),
  // AbortSignal.timeout fires as a rejection from fetch.
  timeout: await probe(GET, async () => {
    throw Object.assign(new Error("The operation was aborted due to timeout"), {
      name: "TimeoutError",
    });
  }),
  // Connection refused / DNS failure surfaces as a TypeError from fetch.
  unreachable: await probe(GET, async () => {
    throw new TypeError("fetch failed");
  }),
};

process.stdout.write(JSON.stringify(results, null, 2));
