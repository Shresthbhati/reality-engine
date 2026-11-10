# Status

Generated from `docs/status.yaml` by `scripts/render_status.py`. Do not edit by hand.

Baseline: 595d3226

| ID | Capability | Status | Gate | Evidence | Tests | Updated | Next |
|---|---|---|---|---|---|---|---|
| S5-01 | Photo → World through the API | PARTIAL | G1 | call graph 1.1; tests/test_golden_path_api_to_glb.py in CI | tests/test_golden_path_api_to_glb.py | 2026-11-10 | P101–P108 |
| S5-02 | Stage contract / observability | MISSING | G1 |  |  | 2026-11-10 | P101–P105 (finding F2) |
| S5-03 | Resumable jobs | FOUNDATION | G1 | leases + heartbeat, no checkpoints (F3) |  | 2026-11-10 | P104, P109 |
| S5-04 | Honest job states | CONTRACT-VERIFIED | G0 | jobs.py:79–91; reconstruction_state.py |  | 2026-11-10 | T007 (unknown state in UI) |
| S5-05 | Universal pipeline | UNWIRED | G1 | F1 |  | 2026-11-10 | P112–P113 |
| S5-06 | Accuracy gate | IMPL-UNVERIFIED | G2 | F5; master plan marks it defective |  | 2026-11-10 | E201–E205. Defective per master plan: fix before trusting. |
| S5-07 | Measured-room dataset | MISSING | G2 |  |  | 2026-11-10 | E207–E210 [H]. No benchmarks/measured_rooms/ (audit claim [A], not re-checked by T010). |
| S5-08 | Closed windows / glass | IMPL-UNVERIFIED | G2 | tests/test_closed_windows.py (synthetic) | tests/test_closed_windows.py | 2026-11-10 | G306–G307 |
| S5-09 | Feed-forward geometry | MISSING | G2 |  |  | 2026-11-10 | G301–G305. Registry has MASt3R only, non-commercial licence. |
| S5-10 | VIO / IMU | FOUNDATION | G3 | evidence/sensors.py parsers, no consumer |  | 2026-11-10 | C601–C606 |
| S5-11 | Provenance classes | CONTRACT-VERIFIED | G3 | F10 |  | 2026-11-10 | W401 |
| S5-12 | Not-observed vs removed | MISSING | G3 |  |  | 2026-11-10 | W402–W405 (F10, F11) |
| S5-13 | Session → World adoption | IMPL-UNVERIFIED | G3 | routes_adoptions.py; tests/test_world_adoption_api.py in CI | tests/test_world_adoption_api.py | 2026-11-10 | W406–W410 |
| S5-14 | Typed edit commands + undo | IMPL-UNVERIFIED | G3 | engine/commands/; WorldEditCommand model |  | 2026-11-10 | U509 |
| S5-15 | CAD/BIM export | IMPL-UNVERIFIED | G2 | exporters/; IfcOpenShell + ezdxf extras |  | 2026-11-10 | E211–E212 [H] |
| S5-16 | Specialist tools (MEP, clash, takeoff, life-safety, thermal) | IMPL-UNVERIFIED | G0 | F8, F9; not in CI |  | 2026-11-10 | T003–T006, U512 |
| S5-17 | Gaussian splats | PARTIAL | G4 | engine/pipeline/gaussian_splats.py (I/O only) |  | 2026-11-10 | DEFERRED to P7 |
| S5-18 | Tenant isolation | IMPL-UNVERIFIED | G4 | apps/api/tenancy.py; tests/test_security_dast.py in-process | tests/test_security_dast.py | 2026-11-10 | S701–S704 |
| S5-19 | Billing | IMPL-UNVERIFIED | G4 | billing_service.py simulated without keys |  | 2026-11-10 | S710 [H] |
| S5-20 | Distributed presence | FOUNDATION | G4 | F14 |  | 2026-11-10 | S711 (document limit; build when multi-worker) |
| S5-21 | One status source | PARTIAL | G0 | F15; docs/status.yaml and docs/STATUS.md added by T010 | tests/test_status_ledger.py | 2026-11-10 | T011 (banner superseded docs). Master plan status word was DOCUMENTATION-DEBT. |
