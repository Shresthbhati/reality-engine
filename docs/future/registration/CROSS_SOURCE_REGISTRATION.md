# Cross-Source Registration

Status: MISSING (2026-09-14)

## Purpose

Express captures from different sensors/sessions (phone RGB, depth
camera, phone-video VIO, GPS-tagged session, LiDAR) in **one shared
world frame**, so a WorldIR world can be compiled from more than one
source without silently choosing one session's frame as truth.

## Current state

Each session compiles into its own world in its own frame (see
`docs/architecture/COORDINATE_FRAMES.md`). Multi-source sessions exist
at the manifest/evidence level, but there is no inter-source
registration stage: sources are never aligned to each other.

## Approach

Ordered by confidence, each producing a **recorded** estimate, never a
silent default:

1. **GNSS anchors.** If two sources both carry GPS with plausible
   accuracy, align by position (translation only, no yaw assumption).
2. **Shared visible geometry (ICP).** Align source B's point cloud /
   mesh to source A's via point-to-plane ICP (scipy or Open3D when
   available). Requires overlap; verify overlap fraction before
   trusting the result.
3. **Shared landmarks / co-observed objects.** Align by detected
   object correspondence sets (needs multi-view identity, P2).
4. **Manual anchor.** User-supplied transform (CLI `reality register`),
   accepted as-is with provenance `user_supplied`.

## Model

`T_world_sourceB = T_world_sourceA @ T_A_B`, where `T_A_B` comes from
the alignment method with:

- `method`, `inlier_fraction`, `rmse` (ICP), `source_points`,
  `target_points`, `iterations`
- covariance or residual distribution when estimable
- overlap verification: reject (BLOCKED, not downgraded) when overlap
  < threshold or rmse >> voxel size.

## Rules

- The world frame of a multi-source world is **declared**
  (`world.metadata["frame"] = "world:arbitrary:sessionA"` or a GNSS
  frame id), never implicit.
- Registration results are evidence artifacts (persisted, hashed,
  provenance-complete) like everything else.
- GPS = unknown stays unknown; a 0,0 anchor is a bug (see
  `docs/future/sensor-ingestion/DEPTH_INGESTION.md` sibling rules and
  CLAUDE.md §44).

## Failure modes

- No overlap → BLOCKED, do not fabricate alignment.
- Degenerate geometry (corridor) → ICP converges to wrong optimum;
  report rmse + inlier stats, mark EXPERIMENTAL confidence.
- Different scales (depth metric vs SfM arbitrary) → scale must be
  resolved first (metric prior or known-size object) before ICP.

## Acceptance criteria

- Deterministic test: two synthetic clouds with known rigid offset →
  recovered transform within tolerance, provenance complete.
- Rejection tests: no overlap and degenerate cases → BLOCKED with
  diagnostics, never a silent best-effort transform.

## Priority

P1 (after VIO + depth; needs metric geometry from at least one source).
