/**
 * DELETE + multipart upload contract harness.
 *
 * Both contracts were read from apps/api/routes_misc.py, not assumed:
 *
 *   DELETE /api/evidence/{id} -> 204 No Content (empty body); 404 if no row
 *   POST   /api/uploads       -> 201 { upload_id, evidence_id, job_id, checksum }
 *                                multipart/form-data, field `file`, optional
 *                                `session_id`; bounded by UPLOAD_MAX_BYTES
 *                                (default 1 GiB); empty body -> 400
 *
 * Invariants:
 *   1. A transport failure never answers 2xx, and never claims 204/201.
 *   2. A malformed or empty 2xx is NOT a successful deletion/ingestion.
 *   3. 404 stays 404 (item gone), distinct from 503 (cannot tell).
 *   4. Deletion is only ever reported on the documented 204.
 *   5. Ingestion is only ever reported on a 201 carrying an `evidence_id`.
 *
 * Usage: node tests/contract/evidence-mutations.contract.mjs
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
const violations = [];
const observations = {};

function stubFetch(behaviour) {
  globalThis.fetch = async (url, init) => behaviour(String(url), init ?? {});
}

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function probe(handler, behaviour, params, request) {
  stubFetch(behaviour);
  try {
    const res = await handler(request, { params: Promise.resolve(params ?? {}) });
    const text = await res.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = null;
    }
    return { status: res.status, body, rawLength: text.length };
  } catch (error) {
    return { threw: true, message: String(error?.message ?? error) };
  } finally {
    globalThis.fetch = realFetch;
  }
}

const fsPath = (rel) =>
  pathToFileURL(fileURLToPath(new URL(`../../src/app/api/${rel}`, import.meta.url))).href;

const { DELETE } = await import(fsPath("evidence/[evidenceId]/route.ts"));
const { POST: uploadPost } = await import(fsPath("uploads/route.ts"));

/** A request carrying a small multipart body, as the browser would send. */
function multipartRequest(bytes = 64) {
  return {
    url: "http://localhost:3000/api/uploads?session_id=s-1",
    headers: {
      get: (name) =>
        name.toLowerCase() === "content-type" ? "multipart/form-data; boundary=----X" : null,
    },
    nextUrl: { search: "?session_id=s-1" },
    arrayBuffer: async () => new ArrayBuffer(bytes),
  };
}

// ---------------------------------------------------------------- DELETE ---
const deleteReq = { url: "http://localhost:3000/api/evidence/ev-1" };

const DELETE_CASES = [
  {
    name: "204 No Content is the only success proof",
    backend: () => new Response(null, { status: 204 }),
    expectStatus: 204,
  },
  {
    name: "404 means the item is already gone",
    backend: () => jsonResponse(404, { detail: "Evidence not found" }),
    expectStatus: 404,
  },
  {
    name: "409 propagates",
    backend: () => jsonResponse(409, { detail: "evidence is referenced" }),
    expectStatus: 409,
  },
  { name: "422 propagates", backend: () => jsonResponse(422, { detail: "bad id" }), expectStatus: 422 },
  { name: "500 propagates", backend: () => jsonResponse(500, { detail: "db locked" }), expectStatus: 500 },
  {
    name: "502 propagates",
    backend: () => new Response("<html>502</html>", { status: 502 }),
    expectStatus: 502,
  },
  { name: "503 propagates", backend: () => jsonResponse(503, { detail: "down" }), expectStatus: 503 },
  {
    // A 200 instead of 204 is an unexpected contract: the deletion happened
    // nowhere we can prove, so it must NOT be reported as a success.
    name: "unexpected 200 is not a deletion",
    backend: () => jsonResponse(200, { ok: true }),
    expectStatus: 502,
  },
  {
    // A 204 is defined to carry no body; the proxy must short-circuit on the
    // status and never try to read one. (The test double cannot construct a
    // 204 WITH a body -- fetch/Response forbids it -- so this case exercises
    // the status short-circuit only.)
    name: "204 short-circuits without reading a body",
    backend: () => new Response(null, { status: 204 }),
    expectStatus: 204,
  },
  {
    name: "timeout is 503, never 404",
    backend: async () => {
      throw Object.assign(new Error("aborted"), { name: "TimeoutError" });
    },
    expectStatus: 503,
  },
  {
    name: "unreachable is 503, never 404",
    backend: async () => {
      throw new TypeError("fetch failed");
    },
    expectStatus: 503,
  },
];

for (const testCase of DELETE_CASES) {
  const out = await probe(DELETE, testCase.backend, { evidenceId: "ev-1" }, deleteReq);
  observations[`delete/${testCase.name}`] = out;
  if (out.threw) {
    violations.push(`delete/${testCase.name}: threw -- ${out.message}`);
    continue;
  }
  if (out.status !== testCase.expectStatus) {
    violations.push(
      `delete/${testCase.name}: expected ${testCase.expectStatus}, got ${out.status}`,
    );
  }
  if (
    (testCase.name.includes("timeout") || testCase.name.includes("unreachable")) &&
    out.status === 404
  ) {
    violations.push(`delete/${testCase.name}: transport failure reported as 404`);
  }
  if (out.status === 204 && out.rawLength !== 0) {
    violations.push(`delete/${testCase.name}: 204 must carry no body`);
  }
}

// ---------------------------------------------------------------- UPLOAD ---
const UPLOAD_CASES = [
  {
    name: "201 with evidence_id is a real ingestion",
    backend: () =>
      jsonResponse(201, {
        upload_id: "upl-1",
        evidence_id: "ev-9",
        job_id: "job-1",
        checksum: "abc",
      }),
    expectStatus: 201,
  },
  {
    name: "201 without evidence_id is NOT ingestion",
    backend: () => jsonResponse(201, { upload_id: "upl-1", job_id: "job-1" }),
    expectStatus: 502,
  },
  {
    name: "200 without evidence_id is not ingestion",
    backend: () => jsonResponse(200, { ok: true }),
    expectStatus: 502,
  },
  {
    name: "empty 201 body is not ingestion",
    backend: () => new Response("", { status: 201 }),
    expectStatus: 502,
  },
  {
    name: "malformed 201 body is not ingestion",
    backend: () =>
      new Response("{not json", {
        status: 201,
        headers: { "content-type": "application/json" },
      }),
    expectStatus: 502,
  },
  { name: "400 propagates", backend: () => jsonResponse(400, { detail: "Empty upload" }), expectStatus: 400 },
  {
    name: "404 propagates",
    backend: () => jsonResponse(404, { detail: "Session not found" }),
    expectStatus: 404,
  },
  { name: "413 propagates", backend: () => jsonResponse(413, { detail: "too large" }), expectStatus: 413 },
  { name: "422 propagates", backend: () => jsonResponse(422, { detail: "field required" }), expectStatus: 422 },
  { name: "500 propagates", backend: () => jsonResponse(500, { detail: "db locked" }), expectStatus: 500 },
  {
    name: "502 propagates",
    backend: () => new Response("<html>502</html>", { status: 502 }),
    expectStatus: 502,
  },
  { name: "503 propagates", backend: () => jsonResponse(503, { detail: "down" }), expectStatus: 503 },
  {
    name: "timeout is 503",
    backend: async () => {
      throw Object.assign(new Error("aborted"), { name: "TimeoutError" });
    },
    expectStatus: 503,
  },
  {
    name: "unreachable is 503",
    backend: async () => {
      throw new TypeError("fetch failed");
    },
    expectStatus: 503,
  },
];

for (const testCase of UPLOAD_CASES) {
  const out = await probe(uploadPost, testCase.backend, {}, multipartRequest());
  observations[`upload/${testCase.name}`] = out;
  if (out.threw) {
    violations.push(`upload/${testCase.name}: threw -- ${out.message}`);
    continue;
  }
  if (out.status !== testCase.expectStatus) {
    violations.push(
      `upload/${testCase.name}: expected ${testCase.expectStatus}, got ${out.status}`,
    );
  }
  if (out.status === 201 && testCase.expectStatus === 201) {
    if (!out.body || typeof out.body.evidence_id !== "string") {
      violations.push(`upload/${testCase.name}: 201 reported as success without an evidence_id`);
    }
  }
}

// Local (pre-backend) rejections must not reach the API at all.
{
  const out = await probe(
    uploadPost,
    () => {
      violations.push("upload/non-multipart: the API was contacted for a non-multipart body");
      return jsonResponse(201, { evidence_id: "ev-1" });
    },
    {},
    {
      url: "http://localhost:3000/api/uploads",
      headers: { get: () => "application/json" },
      nextUrl: { search: "" },
      arrayBuffer: async () => new ArrayBuffer(8),
    },
  );
  observations["upload/non-multipart content type"] = out;
  if (out.status !== 415) {
    violations.push(`upload/non-multipart: expected 415, got ${out.status}`);
  }
}

{
  const out = await probe(
    uploadPost,
    () => {
      violations.push("upload/empty file: the API was contacted with an empty body");
      return jsonResponse(201, { evidence_id: "ev-1" });
    },
    {},
    multipartRequest(0),
  );
  observations["upload/empty file"] = out;
  if (out.status !== 400) {
    violations.push(`upload/empty file: expected 400, got ${out.status}`);
  }
}

process.stdout.write(JSON.stringify({ violations, observations }, null, 2));

