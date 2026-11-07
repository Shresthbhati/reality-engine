# Hosting decision brief (plan T-6.1) — DECISION REQUIRED

Status: **proposed, not decided.** The choice changes cost, operations and what has to be built next, so it is the owner's call
(CLAUDE.md section 51). This brief gives the options, the numbers that are known, the numbers that are not, and a recommendation.

## What the software needs (observed, 2026-11-07)

| Part | Needs | Today |
|------|-------|-------|
| API | CPU, ~1-2 GB RAM, SQLite/Postgres, TLS | `uvicorn apps.api.main:app`; DB via `DATABASE_URL`; alembic head 0004 |
| Worker | One CUDA GPU (24 GB class; the layout stack loads Depth-Anything-V2, UperNet, Grounding DINO, Mask R-CNN one after another), COLMAP, a few GB RAM | `python -m apps.api.worker` (T-6.2); API runs with `REALITY_EMBEDDED_WORKER=0`. Claims are atomic, running jobs heartbeat, a restart loses no job (`tests/test_job_queue_durability.py`) |
| Pipeline artifacts | Object storage | `REALITY_ARTIFACT_STORE=s3` + `REALITY_S3_*` (T-6.3, fake-client tests only; **never run against a real MinIO/S3**) |
| Evidence blobs (uploaded photos) | Storage shared by API and worker | **local directory** `STORAGE_ROOT` (`apps/api/storage.py`) |
| WorldStore (world versions) | Storage shared by API and worker, with a sequence lock | **local directory** `WORLDSTORE_ROOT` (`worldstore/store.py`) |
| Accounts / tenants / usage | Postgres | T-6.4 / T-6.5 done on SQLite-tested code; **Postgres never exercised** |

### The finding that shapes the options

T-6.3 moved only the pipeline-artifacts store to S3. Photos and world versions are still plain directories. **An API box and a
separate GPU box cannot share them**, so "1 API box + 1 GPU worker" is not deployable yet. Either both processes run on one
machine (or share a network volume), or the other two stores are moved to object storage first (registry item M6-2).

## Options

| | A. One GPU box runs everything | B. API box + GPU box, shared network volume | C. API box + GPU box, all stores on object storage | D. API box + serverless GPU per job |
|---|---|---|---|---|
| Shape | API, worker, DB, files on one GPU VM | small VM for API/DB; GPU VM; NFS/EFS-style volume mounted on both | as B, with evidence + WorldStore on S3 (needs M6-2) | as C, GPU billed per second; worker is *pushed* work, ours *pulls* |
| Needs new code | none | none (ops only) | **yes**: evidence store adapter + WorldStore on object storage (the sequence lock is the hard part) | C's code **plus** a push adapter and cold-start handling (model load ~tens of s) |
| Monthly cost (indicative) | GPU VM 24/7 | API VM + GPU VM 24/7 + volume | as B, volume replaced by bucket | API VM + GPU seconds actually used |
| Single point of failure | yes (everything) | volume + each box | each box only | API box only |
| Idle cost | full GPU price | full GPU price | GPU can be stopped when the queue is empty | ~zero GPU |
| Fit for 3-10 pilot firms | **good** | OK, more to operate | good once built | best at low volume, most to build |
| Fit at steady load | poor | OK | good | good while jobs are short and spiky |

### Indicative monthly cost — NOT verified, check current prices before committing

These are order-of-magnitude figures from general knowledge, **not looked up for this brief**; treat them as ranges to be replaced.

- Small API VM (2-4 vCPU, 4-8 GB): roughly USD 20-50. Managed Postgres, smallest tier: roughly USD 15-60.
- One 24 GB-class GPU (L4/A10/RTX-4090 class) on a rental GPU cloud: roughly USD 250-600 if left on 24/7; roughly USD 0.4-1.2 per hour if started only while jobs run.
- Object storage: roughly USD 0.006-0.023 per GB-month; egress is free on some providers and about USD 0.09/GB on others (matters for exports and the 3D viewer).
- Unknown and decisive: **GPU-minutes per room.** `GET /api/usage` (T-6.5) now records job wall-clock per tenant; ten real jobs would turn
  "per-job cost = minutes x hourly rate" from a guess into a number. Until then Option D's saving is unquantified.

## Recommendation

**Option A for the first pilot firms, with the work to reach C queued behind it.**

1. Pilot on one rented 24 GB GPU machine running API, worker, Postgres and local disk. No new code, no shared-storage problem, the lowest
   bill, and it is enough for a handful of firms who capture a few rooms a week. Take nightly volume snapshots (the data is the product).
2. Build **M6-2** (evidence blobs and WorldStore on object storage) when the second paying firm signs or the box is the bottleneck, then move to
   Option C and stop the GPU when the queue is empty.
3. Revisit serverless (Option D) only after real GPU-minutes per room are known from usage metering.

Why not B: a network volume under a lock-based store (WorldStore's sequence lock) is a correctness risk that C avoids by design.

## Consequences if accepted

- Needed before strangers use it, regardless of option: a domain + TLS, backups with a tested restore, an email provider for invites (today the owner
  copies the token), and `REALITY_SIGNUP=open` only behind an approved sign-up path.
- **Depth-Anything-V2-Metric-Indoor-Small licence is unconfirmed** for paid use — a blocker for any paid release on any option.
- T-6.8 (security pass on hosted staging) needs a staging host, so it cannot start before this decision.

## Decision log

| Date | Decision | By |
|------|----------|----|
| — | open | — |
