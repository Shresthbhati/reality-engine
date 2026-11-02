# 01 — Principles (the engineering constitution)

Each rule says what it means in code and how to check it. Reviewers reject
changes that break a rule. IDs (P1…P14) are referenced across all docs.

| ID | Rule | In code this means | How to check |
|----|------|--------------------|--------------|
| P1 | **One place, one world.** | Photos of the same room register into one world/space. A new session of a known place attaches; it does not fork. | Two photo sets of one room → one `Space`, not two. Scenario A9 in [18](18_VALIDATION_AND_BENCHMARKS.md). |
| P2 | **One object, one identity.** | Many `Observation`s → one `Entity`. Never merge on label or distance alone; never transitively (A≈B, B≈C does not imply A≈C). | Duplicate-entity rate and false-merge rate on scenarios A2, A4, A5. |
| P3 | **ML proposes, geometry verifies.** | Model output becomes an `Observation` in the ledger. Only the association + validation stages write the authoritative `WorldIR`. | Model backends never import WorldIR writers (import-boundary test). |
| P4 | **Observed ≠ generated.** | Every surface/property carries an `EvidenceState`. Inferred or prior-generated geometry is never labelled `OBSERVED`. | Schema validator rejects entities without an evidence state. |
| P5 | **Keep uncertainty.** | `AMBIGUOUS` and `INSUFFICIENT_EVIDENCE` are valid outcomes. Conflicts are stored as `CONFLICTED` with both sides kept. | Scenario A20: conflict preserved, not averaged. |
| P6 | **Improve with evidence.** | New photos register into the existing world and update only the affected region (incremental). | Scenario A12: prior geometry unchanged outside the affected region. |
| P7 | **Provenance for every decision.** | Every entity/property links to observation IDs → image IDs → model + version. | Entity inspector shows source photos for any selected entity. |
| P8 | **Scale up, same contracts.** | Room, apartment and building all use the same `World → Building → Floor → Space → Entity` tree. No room-only shortcuts. | No class/table that only a room can use. |
| P9 | **All important entities matter.** | The ontology is hierarchical and extensible (sockets, switches, plants, fixtures included). Unknown objects get the `unknown` class, never a forced one. | Adding a class is a data change in the ontology, not a code change in the pipeline. |
| P10 | **Measure accuracy; never fake precision.** | Metric values need a scale source; otherwise the unit is `relative`. Show uncertainty; round to what the evidence supports. | A measurement without a scale source is stored as relative (already enforced in `WorldMeasurement` — keep it). |
| P11 | **Engineer performance.** | One heavy model resident at a time on 6 GB VRAM; cache per image; reprocess only affected regions. | Peak VRAM + runtime recorded per benchmark run. |
| P12 | **Persistent and manageable.** | Create, open, rename, duplicate, archive, restore, delete (trash), version and export all work. | Lifecycle e2e test (Phase 1). |
| P13 | **Understandable interface.** | Users see "Calculating camera positions", not "COLMAP mapper". Advanced detail is one click away, not in the way. | UX review against [14](14_FRONTEND_UX.md). |
| P14 | **Fail honestly.** | Failed/partial jobs show as failed/partial with a reason. No fake progress %, no placeholder data, no swallowed errors. | No hard-coded demo data or empty `catch {}` in the frontend; job states in the UI match the DB. |

## Two working rules (apply to every task)

- **Reuse before build.** Check [20_REUSE_MAP.md](20_REUSE_MAP.md) and mature
  OSS first. A new algorithm needs a one-line reason why existing ones don't fit.
- **Verify before status.** No task is DONE and no capability is VERIFIED
  without pasted execution evidence (test output, API response, screenshot).

## Engineering defaults

- Python 3.11 for engine + API; strict TypeScript for the frontend.
- Deterministic: seeded RNG, stable IDs, deterministic serialization.
- Every external tool/model sits behind an interface, with a fallback where
  practical, a pinned version and a licence record in `.agent/LICENSES.yaml`.
- Models under non-commercial licences may be benchmarked but **never** become
  the default in a shipped path.
