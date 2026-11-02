"""World version lifecycle policy (WORLDOS_LIFECYCLE_HEALTH mission,
Mission 1): turns manual `compact_manifest_chain` calls
(`worldstore/tiles.py`) into a governed, observable policy -- WHEN a
delta chain should be compacted, not a new storage mechanism. Uses
ONLY existing WorldStore/tile primitives; no new schema, no new store.

Deliberately does NOT compact automatically inside `save_version_*` --
a caller decides when to CHECK the policy (e.g. after every save, or on
a timer), and `maybe_compact` decides whether that check actually does
anything. This keeps every existing save path's behavior unchanged
(backwards compatible) unless a caller explicitly opts into calling
this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from worldstore.store import WorldStore
from worldstore.tiles import (
    TileManifestDelta,
    _delta_path,
    _manifest_path,
    compact_manifest_chain,
    delta_chain_depth,
)


@dataclass(frozen=True)
class CompactionPolicy:
    """Deterministic, configurable thresholds. Any one tripping
    triggers compaction -- `CompactionDecision.reason` reports exactly
    which one, so the policy is observable, not a black box.

    Rationale for each threshold (read amplification / write
    amplification tradeoff, per the mission's own framing):
      - `max_chain_depth`: bounds READ amplification -- every
        `_resolve_manifest`/`open_version` call for a version this deep
        walks this many delta files before it can answer anything.
      - `max_accumulated_delta_bytes`: bounds the actual bytes a chain
        walk has to read and parse, which chain DEPTH alone doesn't
        capture if individual deltas are large (e.g. many tiles
        rebuilt in one save).
      - `max_deltas_before_forced_compaction`: a hard ceiling
        independent of the other two, so a pathological case (many
        tiny deltas, each individually under the byte threshold) still
        can't grow the chain unboundedly.
    Compaction itself has a WRITE cost (rewrites a full manifest,
    O(tile count)) -- these thresholds exist precisely to avoid paying
    that cost on every single save (which would erase the whole point
    of delta saves) while still bounding read cost.
    """
    max_chain_depth: int = 20
    max_accumulated_delta_bytes: int = 1_000_000
    max_deltas_before_forced_compaction: int = 50


DEFAULT_POLICY = CompactionPolicy()


def _accumulated_delta_bytes(store: WorldStore, version_id: str) -> int:
    """Sum of delta file sizes from `version_id` back to (not
    including) the nearest full manifest -- the actual bytes a future
    chain-walk would have to read for this version, not just a hop
    count."""
    total = 0
    current_id = version_id
    while True:
        if _manifest_path(store, current_id).exists():
            return total
        d_path = _delta_path(store, current_id)
        if not d_path.exists():
            return total  # unresolvable chain -- let compact_manifest_chain surface the real error
        total += d_path.stat().st_size
        delta = TileManifestDelta.from_dict(json.loads(d_path.read_text(encoding="utf-8")))
        current_id = delta.parent


@dataclass(frozen=True)
class CompactionDecision:
    version_id: str
    depth: int
    accumulated_delta_bytes: int
    should_compact: bool
    reason: Optional[str]


def evaluate_compaction(
    store: WorldStore, version_id: str, policy: CompactionPolicy = DEFAULT_POLICY,
) -> CompactionDecision:
    """Pure, deterministic policy evaluation -- no side effects, safe
    to call as often as a caller likes (e.g. to observe/log chain
    growth without acting on it). Backwards compatible: a version with
    a full manifest (whole-world-tiled or already-compacted) has
    `depth=0` and never trips any threshold."""
    depth = delta_chain_depth(store, version_id)
    accumulated = _accumulated_delta_bytes(store, version_id)
    reason = None
    if depth >= policy.max_deltas_before_forced_compaction:
        reason = f"chain depth {depth} >= hard ceiling {policy.max_deltas_before_forced_compaction}"
    elif depth >= policy.max_chain_depth:
        reason = f"chain depth {depth} >= max_chain_depth {policy.max_chain_depth}"
    elif accumulated >= policy.max_accumulated_delta_bytes:
        reason = f"accumulated delta bytes {accumulated} >= max_accumulated_delta_bytes {policy.max_accumulated_delta_bytes}"
    return CompactionDecision(
        version_id=version_id, depth=depth, accumulated_delta_bytes=accumulated,
        should_compact=reason is not None, reason=reason,
    )


@dataclass(frozen=True)
class CompactionOutcome:
    decision: CompactionDecision
    attempted: bool
    compacted: bool
    depth_after: Optional[int]
    error: Optional[str]


def maybe_compact(
    store: WorldStore, version_id: str, policy: CompactionPolicy = DEFAULT_POLICY,
) -> CompactionOutcome:
    """Evaluates the policy and compacts only if warranted -- never
    blindly compacts every version.

    FAILURE SAFE by construction, not by a try/except band-aid:
    `compact_manifest_chain` resolves the ENTIRE effective manifest in
    memory first, and only then does a single atomic file replace. If
    resolution raises (corrupted delta, missing artifact, etc.), NO
    write happens at all -- `version_id` remains exactly as resolvable
    via its existing delta chain as it was before this call. The
    try/except here exists so a caller running this as a background
    lifecycle sweep doesn't have one bad version's compaction failure
    interrupt sweeping the rest; the previous valid version is provably
    unaffected either way. This wraps EVALUATION too, not just the
    compact call -- `evaluate_compaction`/`delta_chain_depth` can
    themselves raise on a corrupted chain (e.g. a bad delta file), and
    that failure must be just as non-fatal and observable as a failure
    during the actual compact step.
    """
    try:
        decision = evaluate_compaction(store, version_id, policy)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        placeholder = CompactionDecision(
            version_id=version_id, depth=-1, accumulated_delta_bytes=-1, should_compact=False, reason=None,
        )
        return CompactionOutcome(decision=placeholder, attempted=False, compacted=False, depth_after=None, error=str(exc))

    if not decision.should_compact:
        return CompactionOutcome(
            decision=decision, attempted=False, compacted=False, depth_after=decision.depth, error=None,
        )
    try:
        compact_manifest_chain(store, version_id)
        return CompactionOutcome(decision=decision, attempted=True, compacted=True, depth_after=0, error=None)
    except Exception as exc:  # noqa: BLE001 -- a lifecycle sweep must not propagate one version's failure
        return CompactionOutcome(
            decision=decision, attempted=True, compacted=False, depth_after=decision.depth, error=str(exc),
        )
