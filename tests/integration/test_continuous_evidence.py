"""Continuous evidence: the SAME world keeps evolving as evidence arrives, in any order.

The product is not "N images -> a model"; it is a world that becomes better
supported as evidence accumulates. These tests feed real South Building
photographs (real COLMAP + real MiDaS, nothing mocked) in different orders and
check the invariants that must hold regardless of order:

  * one world across the whole sequence (never a new world per upload)
  * versions are numbered 1..n, and every earlier version stays retrievable
  * no evidence is ever dropped: after every step the world knows every
    photograph posted so far
  * a photograph that cannot be registered yet is KEPT (not discarded) and can
    be registered once later evidence bridges to it (A -> C -> B)
  * an unrelated photograph is kept, flagged as not contributing spatially,
    and does not break what was already registered

Each printed trace line is the measured state, so a failure shows real numbers.
Batches (consecutive views of one walk; neighbours share hundreds of verified
matches, A and C only connect through B):

    A = [15:18]   B = [18:21]   C = [21:24]   U = [0:1] (wide-baseline, unrelated)
"""

from __future__ import annotations

import pytest

from tests.integration.test_progressive_product_journey import (  # noqa: F401 -- shared harness + marks
    IMGS,
    _post,
    _settle,
    _worldir,
    app_client,
    pytestmark,
)

A, B, C, U = IMGS[15:18], IMGS[18:21], IMGS[21:24], IMGS[0:1]


def _run(tmp_path, steps, after=None):
    """Post each batch to the SAME world; return the measured trace.
    ``after(client, world_id, trace)`` runs inside the live app once all steps settled."""
    trace, posted, wid = [], [], None
    tmp_path.mkdir(parents=True, exist_ok=True)
    with app_client(tmp_path) as c:
        for name, batch in steps:
            r = _post(c, batch, world_id=wid)
            wid = wid or r["world_id"]
            assert r["world_id"] == wid, "an upload must join the existing world, never make a new one"
            posted += [e["id"] for e in r["evidence"]]
            st = _settle(c, wid)
            m = st["model"]
            known = {e["id"]: e for e in st["evidence"]}
            row = {
                "step": name, "state": st["state"], "level": m["level"], "model_state": m["model_state"],
                "used": m["images_used"], "registered": m["images_registered"],
                "version": next(v["number"] for v in st["versions"] if v["is_current"]),
                "n_versions": len(st["versions"]), "version_id": m["version_id"],
                "entities": len(_worldir(c, wid)["entities"]),
                "known": known, "posted": list(posted),
            }
            trace.append(row)
            print(f"[{name}] v{row['version']} level={row['level']} {row['model_state']} "
                  f"used={row['used']} registered={row['registered']} entities={row['entities']}")
            # invariants that must hold after EVERY step, in ANY order
            assert set(posted) <= set(known), "evidence was dropped from the world"
            assert row["used"] == len(posted), (row["used"], len(posted))
            assert row["n_versions"] == len(trace), "each meaningful refinement is exactly one new version"
        # every earlier version is still there and still readable (immutability)
        for row in trace:
            assert _worldir(c, wid, row["version_id"])["entities"], row["step"]
        if after:
            after(c, wid, trace)
    return trace


def test_A_then_B_then_C_refines_one_world_without_reset(tmp_path):
    t = _run(tmp_path, [("A", A), ("B", B), ("C", C)])
    assert [r["version"] for r in t] == [1, 2, 3]
    assert [r["used"] for r in t] == [3, 6, 9]
    # supporting evidence only grows: registration never collapses back after more overlapping views
    regs = [r["registered"] for r in t]
    assert regs == sorted(regs), f"registration regressed across a reset-like step: {regs}"
    assert regs[-1] >= 7, regs


def test_A_then_C_then_B_the_late_bridge_registers_what_was_waiting(tmp_path):
    """C shares nothing with A, so at step 2 it cannot join A's model. It must
    be KEPT, and once B (the bridge) arrives, everything registers together."""
    t = _run(tmp_path, [("A", A), ("C", C), ("B", B)])
    ids_c = set(t[1]["posted"][3:])
    waiting = [i for i in ids_c if not t[1]["known"][i]["registered"]]
    print("C photos not registered at step 2:", len(waiting))
    assert waiting, "expected C to be unregistered before the bridge (otherwise this scenario tests nothing)"
    assert all(t[1]["known"][i] for i in ids_c)                      # kept, not discarded
    final = t[-1]
    assert final["registered"] >= 7, final["registered"]             # the bridge let them in
    assert final["registered"] > t[1]["registered"], "the late bridge must add registered evidence"


def test_unrelated_photo_is_kept_flagged_and_harmless(tmp_path):
    t = _run(tmp_path, [("A", A), ("B", B), ("U", U), ("C", C)])
    u_id = t[2]["posted"][-1]
    u = t[2]["known"][u_id]
    assert u["registered"] is not True, "an unrelated photo must not be claimed as spatially registered"
    assert t[2]["registered"] >= t[1]["registered"], "an unrelated photo must not un-register earlier evidence"
    assert t[3]["registered"] >= t[2]["registered"]
    assert t[3]["used"] == 4 * 3 - 2 and u_id in t[3]["known"]       # 3+3+1+3 = 10, still remembered


def test_orderings_reach_equivalent_final_support(tmp_path):
    abc = _run(tmp_path / "abc", [("A", A), ("B", B), ("C", C)])[-1]
    acb = _run(tmp_path / "acb", [("A", A), ("C", C), ("B", B)])[-1]
    assert abc["used"] == acb["used"] == 9
    assert abs(abc["registered"] - acb["registered"]) <= 1, (abc["registered"], acb["registered"])


@pytest.mark.xfail(strict=True, reason="NOT IMPLEMENTED: conflicting evidence is not yet represented as "
                                       "an explicit conflict in WorldIR (see PENDING_IMPLEMENTATION.md)")
def test_conflicting_evidence_is_explicit_not_silently_resolved(tmp_path):
    def check(c, wid, trace):
        assert "conflicts" in _worldir(c, wid)["metadata"]

    _run(tmp_path, [("A", A), ("B", B)], after=check)
