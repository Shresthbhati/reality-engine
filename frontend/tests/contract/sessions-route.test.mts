/**
 * Contract tests for the Next.js sessions proxy.
 *
 * These drive the real route handler (`src/app/api/sessions/route.ts`) with a
 * stubbed global `fetch`, so they assert the shipped behaviour rather than the
 * shape of the source text.
 *
 * The bug this pins: the proxy used to answer a backend failure with
 * `{ error, sessions: [], count: 0 }`. That is a fallback-success envelope --
 * an API outage was rendered by consumers as "this user has no sessions",
 * and because the real contract is `{ items: [...] }` the fabricated
 * `sessions` key did not even match what the backend sends.
 *
 * Run with: npm run test:contract
 */
import { test } from "node:test";
import assert from "node:assert/strict";

const BACKEND_URL = "http://localhost:8100";

type Handler = typeof import("../../src/app/api/sessions/route").GET;

async function loadHandler(): Promise<Handler> {
  const mod = await import("../../src/app/api/sessions/route");
  return mod.GET;
}

/** A NextRequest stand-in: the handler only reads `url` search params. */
function request(url: string) {
  return { url } as unknown as import("next/server").NextRequest;
}

/** Install a `fetch` stub, run the handler, and restore the real `fetch`. */
async function callWith(
  handler: Handler,
  url: string,
  impl: () => Promise<Response>
): Promise<Response> {
  const original = globalThis.fetch;
  globalThis.fetch = impl as unknown as typeof globalThis.fetch;
  try {

test("a successful backend response is forwarded verbatim as { items }", async () => {
  const handler = await loadHandler();
  const payload = {
    items: [
      {
        id: "ses_1",
        name: "Kitchen sweep",
        status: "captured",
        world_id: null,
        coordinate_reference_system: "WGS84",
        evidence_count: 0,
      },
    ],
  };

  const res = await callWith(handler, "http://localhost:3000/api/sessions", async () =>
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  );

  assert.equal(res.status, 200);
  const body = (await res.json()) as Record<string, unknown>;
  assert.deepEqual(body, payload, "the proxy must not rewrap the backend body");
  assert.ok(Array.isArray(body.items), "the real contract is an items list");
  assert.equal(
    body.sessions,
    undefined,
    "the proxy must not invent a `sessions` key the backend never emits"
  );
});

test("a genuinely empty session list stays an honest empty success", async () => {
  const handler = await loadHandler();
  const res = await callWith(handler, "http://localhost:3000/api/sessions", async () =>
    new Response(JSON.stringify({ items: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  );

  assert.equal(res.status, 200);
  assert.deepEqual(await res.json(), { items: [] });
});

test("the world_id filter is forwarded to the backend", async () => {
  const handler = await loadHandler();
  let seen = "";
  await callWith(
    handler,
    "http://localhost:3000/api/sessions?world_id=wld_7",
    async (input) => {
      seen = String(input);
      return new Response(JSON.stringify({ items: [] }), { status: 200 });
    }
  );

for (const status of [400, 401, 404, 409, 422, 500, 502, 503]) {
  test(`a backend ${status} stays an error, not an empty session list`, async () => {
    const handler = await loadHandler();
    const res = await callWith(handler, "http://localhost:3000/api/sessions", async () =>
      new Response(JSON.stringify({ detail: `backend said ${status}` }), { status })
    );

    assert.equal(res.status, status, "the backend status must be preserved");
    const body = (await res.json()) as Record<string, unknown>;
    assert.equal(body.detail, `backend said ${status}`, "the backend body must be preserved");
    assert.equal(body.sessions, undefined, "a failure must never carry a `sessions` list");
    assert.equal(body.items, undefined, "a failure must never carry an `items` list");
    assert.equal(body.count, undefined, "a failure must never carry a `count`");
  });
}

test("a backend timeout is a 503, not an empty success", async () => {
  const handler = await loadHandler();
  const res = await callWith(handler, "http://localhost:3000/api/sessions", async () => {
    // Mirrors AbortSignal.timeout firing inside the route.
    throw Object.assign(new Error("The operation was aborted due to timeout"), {
      name: "TimeoutError",
    });
  });

  assert.equal(res.status, 503);
  const body = (await res.json()) as Record<string, unknown>;
  assert.equal(body.available, false);
  assert.equal(body.sessions, undefined, "a timeout must not become an empty session list");
  assert.equal(body.items, undefined);
  assert.match(String(body.detail), /Failed to reach the Reality Engine API/);
});

test("an unreachable backend is a 503, not an empty success", async () => {
  const handler = await loadHandler();
  const res = await callWith(handler, "http://localhost:3000/api/sessions", async () => {
    throw new TypeError("fetch failed");
  });

  assert.equal(res.status, 503);
  const body = (await res.json()) as Record<string, unknown>;
  assert.equal(body.available, false);
  assert.equal(body.sessions, undefined);
  assert.equal(body.items, undefined);
});

  assert.equal(seen, `${BACKEND_URL}/api/sessions?world_id=wld_7`);
});

    return await handler(request(url));
  } finally {
    globalThis.fetch = original;
  }
}
