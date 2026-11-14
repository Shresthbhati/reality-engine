# Reality Engine benchmark

A repeatable benchmark that says where Reality Engine succeeds, where it hallucinates, and what needs engineering work.
Everything here is a **proposed definition**. No metric has been measured against the engine yet; the default
dashboard says so (`python scripts/run_reality_benchmark.py`).

## What is in the box

| File | Purpose |
|---|---|
| `registry.py` | 12 metric categories (weights sum to 100), 118 metrics with targets, 6 critical gates, 24 scenarios |
| `scorer.py` | turns measurements into metric results, gates, a weighted scorecard and a verdict |
| `manifests.py` | empty per-scenario manifests (datasets, camera poses, ground truth, tolerances, pass/fail) |
| `result.schema.json` | the scorecard / `metric_results` schema |
| `scripts/run_reality_benchmark.py` | dashboard, scoring from result files, manifest emitter |
| `tests/test_reality_benchmark.py` | registry coherence and the scorer's honesty rules |

## Rules the scorer enforces

* **Unmeasured is never scored.** No value means `NOT_RUN`; a metric that needs ground truth that does not exist is `BLOCKED`.
* **Too few samples, or no threshold, is `PROVISIONAL`**, never `PASS` (scene-specific metrics need a threshold from the scenario manifest `tolerances`).
* **Critical gates beat the weighted score.** Any invalid commit reported as successful, corrupted commit, unresolved reference, invalid critical mesh, false-success state, or fabricated architectural entity fails the scenario however good detection is.
* **Small objects are reported on their own** (category E, per entity group) and a scenario that contains them is not complete until they are measured.
* **Verdict**: `FAIL` if any gate or measured metric fails; `PASS` only when every applicable metric passed; otherwise `INCOMPLETE`. The weighted score covers measured categories only and `coverage` says how much of the weight that is.
* Weights: A 15, B 20, C 15, D 12, E 10, G 10, H 5, I 5, J 2.5, K 2.5, L 3. Materials (F) had no proposed weight: reported, not weighted.

## Datasets and ground truth

Two datasets per scenario: **A** controlled synthetic scenes and **B** real captures. Generated images can disagree across
viewpoints, so they cannot establish real-world accuracy on their own. Quantitative geometry, multi-view and topology
metrics need ground truth: surveyed dimensions, verified camera poses, object identities. Without it they stay `BLOCKED`.
For one surveyed room see `benchmarks/measured_rooms/`.

## Use

```bash
python scripts/run_reality_benchmark.py                        # awaiting-execution dashboard
python scripts/run_reality_benchmark.py --emit-manifests DIR   # 24 empty manifests to fill in
python scripts/run_reality_benchmark.py --results DIR --json   # score result files
```

A result file: `{"scenario_id": "001", "ground_truth_available": true, "metric_results": [...]}`.
