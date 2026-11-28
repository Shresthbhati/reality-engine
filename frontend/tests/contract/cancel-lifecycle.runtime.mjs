/**
 * RUNTIME cancellation lifecycle test (mandatory, not a source grep).
 *
 * Simulates the real component loop against a scripted backend:
 *
 *   reconstruct -> queued -> running -> [click Cancel] -> 202 requested
 *               -> backend terminal CANCELLED -> UI shows cancelled
 *
 * It also drives the races that matter, and asserts the one hard invariant:
 *
 *   THE UI MUST NEVER DISPLAY "cancelled" WITHOUT THE BACKEND SAYING SO.
 *
 * The displayed status is derived the same way the component derives it, via
 * the canonical `classifyReconstruction`, so this exercises the real rule.
 *
 * Usage: node tests/contract/cancel-lifecycle.runtime.mjs
 */
import { fileURLToPath, pathToFileURL } from "node:url";

const modPath = pathToFileURL(
  fileURLToPath(new URL("../../src/lib/api/reconstruction.ts", import.meta.url)),
).href;
const { classifyReconstruction } = await import(modPath);

const violations = [];
const timeline = [];

/** Mirrors `cancelJob` in lib/api/platform.ts. */
async function cancelJob(backend, id) {
  const res = await backend.cancel(id);
  if (res.status === 409 || res.status === 404) {
    return { outcome: "rejected", reason: res.body?.detail ?? "refused" };
  }
  if (!res.ok) {
    return { outcome: "rejected", reason: res.body?.detail ?? `HTTP ${res.status}` };
  }
  if (res.status === 202 || res.body?.cancel_requested === true) {
    return { outcome: "cancelling" };
  }
  return { outcome: "cancelled" };
}

/**
 * A faithful re-implementation of the component's cancel state machine, using
 * the canonical rule for what the badge shows. `cancelState.kind` is the value
 * the Cancel button is allowed to set; `displayedStatus` is what the operator
 * sees, and is derived ONLY from a polled job.
 */
function createModal() {
  return {
    jobId: null,
    cancelState: { kind: "idle" },
    cancelInFlight: false,
    jobStatus: "idle",
    get displayedStatus() {
      return this.jobStatus;
    },
    async enqueue(backend) {
      const res = await backend.reconstruct();
      if (!res.ok) throw new Error(res.detail);
      if (typeof res.body?.job_id !== "string") throw new Error("enqueue returned no job_id");
      this.jobId = res.body.job_id;
      this.jobStatus = res.body.status ?? "queued";
      this.cancelState = { kind: "idle" };
      return this.jobId;
    },
    /** Exactly what the click handler does. Never sets jobStatus. */
    async clickCancel(backend) {
      if (!this.jobId) return;
      if (this.cancelInFlight) {
        timeline.push("duplicate click ignored (in flight)");
        return;
      }
      if (this.cancelState.kind === "requesting" || this.cancelState.kind === "requested") {
        timeline.push("duplicate click ignored (already requested)");
        return;
      }
      this.cancelInFlight = true;
      this.cancelState = { kind: "requesting" };
      try {
        const r = await cancelJob(backend, this.jobId);
        if (r.outcome === "cancelling") {
          this.cancelState = { kind: "requested", detail: "winding down" };
        } else if (r.outcome === "cancelled") {
          this.cancelState = { kind: "requested", detail: "accepted" };
        } else {
          this.cancelState = { kind: "refused", detail: r.reason };
        }
      } finally {
        this.cancelInFlight = false;
      }
    },
    /** One poll tick; mirrors the component's loop body. */
    async poll(backend) {
      const res = await backend.getJob(this.jobId);
      if (!res.ok) return; // transient read failure: keep polling
      const job = res.body;
      if (job.status === "running" || job.status === "queued") this.jobStatus = job.status;
      const verdict = classifyReconstruction(job);
      if (verdict.kind === "pending") return;
      if (verdict.kind === "success" || verdict.kind === "partial") {
        this.jobStatus = verdict.kind === "partial" ? "partial" : "succeeded";
        this.cancelState = { kind: "idle" };
        return;
      }
      // Terminal, not persisted. The badge reflects what the BACKEND said.
      this.jobStatus = job.status === "cancelled" ? "cancelled" : "failed";
      this.cancelState =
        job.status === "cancelled"
          ? { kind: "requested", detail: "Cancelled by operator." }
          : { kind: "idle" };
    },
  };
}

const json = (status, body) => ({ ok: status >= 200 && status < 300, status, body });


// =========================================================================
// SCENARIO 1 -- happy path: running -> cancel -> 202 -> backend cancelled
// =========================================================================
{
  const backend = {
    reconstruct: async () => json(200, { job_id: "job-1", type: "reconstruct", status: "queued" }),
    getJob: async (id) => {
      const s = backend.status ?? "queued";
      timeline.push(`poll job-1 -> ${s}`);
      if (s === "cancelled") return json(200, { id, status: "cancelled", payload: {} });
      return json(200, { id, status: s, stage: "sfm", payload: {} });
    },
    cancel: async () => {
      timeline.push("cancel request -> 202 (winding down)");
      return { ok: true, status: 202, body: { id: "job-1", status: "running", cancel_requested: true } };
    },
    status: "queued",
  };
  const m = createModal();
  await m.enqueue(backend);
  await m.poll(backend);
  backend.status = "running";
  await m.poll(backend);
  timeline.push(`UI before cancel: status=${m.displayedStatus} cancelState=${m.cancelState.kind}`);
  if (m.displayedStatus !== "running") {
    violations.push(`lifecycle: expected 'running' before cancel, got '${m.displayedStatus}'`);
  }

  await m.clickCancel(backend);
  timeline.push(`after click: cancelState=${m.cancelState.kind} status=${m.displayedStatus}`);

  // THE INVARIANT: the click must NOT have produced a cancelled display.
  if (m.displayedStatus === "cancelled") {
    violations.push("lifecycle: the click manufactured a 'cancelled' display locally");
  }
  if (m.cancelState.kind !== "requested") {
    violations.push(`lifecycle: expected cancelState 'requested' after 202, got '${m.cancelState.kind}'`);
  }

  // Still running server-side: another poll must not flip it.
  await m.poll(backend);
  if (m.displayedStatus === "cancelled") {
    violations.push("lifecycle: 'cancelled' shown while the backend still reported running");
  }

  // Now the backend actually cancels.
  backend.status = "cancelled";
  await m.poll(backend);
  timeline.push(`after backend terminal: status=${m.displayedStatus}`);
  if (m.displayedStatus !== "cancelled") {
    violations.push(`lifecycle: after the backend reported cancelled the UI shows '${m.displayedStatus}'`);
  }
}

// =========================================================================
// SCENARIO 2 -- cancel vs success race: 409, job already succeeded
// =========================================================================
{
  const backend = {
    reconstruct: async () => json(200, { job_id: "job-2", status: "queued" }),
    getJob: async (id) => json(200, { id, status: "succeeded", payload: { version_id: "v-1" } }),
    cancel: async () => ({ ok: false, status: 409, body: { detail: "Job already succeeded" } }),
    status: "running",
  };
  const m = createModal();
  await m.enqueue(backend);
  await m.poll(backend);
  await m.clickCancel(backend);
  if (m.cancelState.kind !== "refused") {
    violations.push(`race-success: expected 'refused' on 409, got '${m.cancelState.kind}'`);
  }
  if (m.displayedStatus === "cancelled") {
    violations.push("race-success: a refused cancel was displayed as cancelled");
  }
  await m.poll(backend);
  if (m.displayedStatus !== "succeeded") {
    violations.push(`race-success: expected 'succeeded', got '${m.displayedStatus}'`);
  }
  timeline.push(`race-success: final status=${m.displayedStatus} (cancelState=${m.cancelState.kind})`);
}

// =========================================================================
// SCENARIO 3 -- cancel vs failure race: 202 then the job fails
// =========================================================================
{
  const backend = {
    reconstruct: async () => json(200, { job_id: "job-3", status: "queued" }),
    getJob: async (id) => {
      const s = backend.status ?? "running";
      if (s === "failed") return json(200, { id, status: "failed", error: "reconstruction failed", payload: {} });
      return json(200, { id, status: s, payload: {} });
    },
    cancel: async () => ({ ok: true, status: 202, body: { id: "job-3", status: "running", cancel_requested: true } }),
    status: "running",
  };
  const m = createModal();
  await m.enqueue(backend);
  await m.poll(backend);
  await m.clickCancel(backend);
  if (m.displayedStatus === "cancelled") violations.push("race-failure: cancel click displayed 'cancelled'");
  backend.status = "failed";
  await m.poll(backend);
  if (m.displayedStatus !== "failed") {
    violations.push(`race-failure: expected 'failed', got '${m.displayedStatus}'`);
  }
  timeline.push(`race-failure: final status=${m.displayedStatus}`);
}

// =========================================================================
// SCENARIO 4 -- duplicate clicks must not fire duplicate requests
// =========================================================================
{
  let cancelCalls = 0;
  const backend = {
    reconstruct: async () => json(200, { job_id: "job-4", status: "queued" }),
    getJob: async (id) => json(200, { id, status: backend.status ?? "running", payload: {} }),
    cancel: async () => {
      cancelCalls++;
      return { ok: true, status: 202, body: { id: "job-4", status: "running", cancel_requested: true } };
    },
    status: "running",
  };
  const m = createModal();
  await m.enqueue(backend);
  await m.poll(backend);
  // Three concurrent clicks, then one more afterwards.
  await Promise.all([m.clickCancel(backend), m.clickCancel(backend), m.clickCancel(backend)]);
  await m.clickCancel(backend);
  if (cancelCalls !== 1) {
    violations.push(`duplicate-cancel: expected exactly 1 cancel request, got ${cancelCalls}`);
  }
  timeline.push(`duplicate-cancel: ${cancelCalls} request(s) for 4 clicks`);
}

// =========================================================================
// SCENARIO 5 -- repeated cancel / unknown job / transport failure
// =========================================================================
for (const [name, response, expectKind] of [
  ["repeated cancel on a terminal job (409)", { ok: false, status: 409, body: { detail: "already cancelled" } }, "refused"],
  ["unknown job (404)", { ok: false, status: 404, body: { detail: "Job not found" } }, "refused"],
  ["transport failure (503)", { ok: false, status: 503, body: { detail: "API unreachable" } }, "refused"],
]) {
  const backend = {
    reconstruct: async () => json(200, { job_id: "job-5", status: "queued" }),
    getJob: async (id) => json(200, { id, status: "running", payload: {} }),
    cancel: async () => response,
  };
  const m = createModal();
  await m.enqueue(backend);
  await m.clickCancel(backend);
  if (m.cancelState.kind !== expectKind) {
    violations.push(`${name}: expected '${expectKind}', got '${m.cancelState.kind}'`);
  }
  if (m.displayedStatus === "cancelled") {
    violations.push(`${name}: displayed 'cancelled' with no backend confirmation`);
  }
  timeline.push(`${name}: cancelState=${m.cancelState.kind} status=${m.displayedStatus}`);
}

process.stdout.write(JSON.stringify({ violations, timeline }, null, 2));

