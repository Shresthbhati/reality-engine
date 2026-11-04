import { test } from "node:test";
import assert from "node:assert/strict";
import {
  formatBytes,
  summarizeJobs,
  jobDurationText,
} from "../../src/lib/dashboard/dashboard-model.ts";
import type { JobDto } from "../../src/lib/api/types";

test("formatBytes formats byte sizes accurately", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1024), "1.00 KB");
  assert.equal(formatBytes(1536), "1.50 KB");
  assert.equal(formatBytes(1048576), "1.00 MB");
  assert.equal(formatBytes(15728640), "15.0 MB");
  assert.equal(formatBytes(1073741824), "1.00 GB");
  assert.equal(formatBytes(null), "0 B");
  assert.equal(formatBytes(-10), "0 B");
});

test("summarizeJobs categorizes active, failed, and recent jobs", () => {
  const jobs: JobDto[] = [
    {
      id: "j-1",
      type: "reconstruction",
      entity_type: "world",
      entity_id: "w-1",
      status: "running",
      stage: "dense_reconstruction",
      attempts: 1,
      error: null,
      worker_id: "wkr-1",
      created_at: "2026-11-04T10:00:00Z",
      started_at: "2026-11-04T10:00:05Z",
    },
    {
      id: "j-2",
      type: "reconstruction",
      entity_type: "world",
      entity_id: "w-2",
      status: "failed",
      stage: "feature_matching",
      attempts: 2,
      error: "insufficient matching pairs found",
      worker_id: "wkr-1",
      created_at: "2026-11-04T09:30:00Z",
      started_at: "2026-11-04T09:30:05Z",
      completed_at: "2026-11-04T09:31:15Z",
    },
    {
      id: "j-3",
      type: "export_gltf",
      entity_type: "world",
      entity_id: "w-1",
      status: "succeeded",
      stage: null,
      attempts: 1,
      error: null,
      worker_id: null,
      created_at: "2026-11-04T09:00:00Z",
      started_at: "2026-11-04T09:00:02Z",
      completed_at: "2026-11-04T09:00:10Z",
    },
  ];

  const summary = summarizeJobs(jobs);
  assert.equal(summary.counts.total, 3);
  assert.equal(summary.counts.running, 1);
  assert.equal(summary.counts.failed, 1);
  assert.equal(summary.counts.succeeded, 1);

  assert.equal(summary.active.length, 1);
  assert.equal(summary.active[0].id, "j-1");

  assert.equal(summary.failed.length, 1);
  assert.equal(summary.failed[0].id, "j-2");
  assert.equal(summary.failed[0].error, "insufficient matching pairs found");

  assert.equal(summary.recent.length, 3);
});

test("summarizeJobs handles empty or null jobs array", () => {
  const empty = summarizeJobs([]);
  assert.equal(empty.counts.total, 0);
  assert.equal(empty.active.length, 0);
  assert.equal(empty.failed.length, 0);

  const nil = summarizeJobs(null);
  assert.equal(nil.counts.total, 0);
});

test("jobDurationText computes elapsed or completed duration", () => {
  assert.equal(
    jobDurationText("2026-11-04T10:00:00Z", "2026-11-04T10:00:45Z"),
    "45s"
  );
  assert.equal(
    jobDurationText("2026-11-04T10:00:00Z", "2026-11-04T10:02:30Z"),
    "2m 30s"
  );
  assert.equal(jobDurationText(null, null), null);
});
