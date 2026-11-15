# Unlisted suites: timing audit (T002)

Base commit: `595d3226246d23f65b76f5bf4600b1a5751cba40`. Python 3.14.7, Windows 11, CPU only.
Each file was run alone: `python -m pytest tests/<file> -q --tb=line --durations=3`.
"Wall seconds" is the shell-measured time for the whole invocation, including interpreter start and collection. The last output line is copied verbatim and shows pytest's own test time. Skipped is 0 for every file because no "skipped" count appeared in any summary line. Every run exited with code 0.

| File | Passed | Failed | Skipped | Wall seconds | Last output line (verbatim) |
|---|---|---|---|---|---|
| tests/test_gaussian_splats.py | 7 | 0 | 0 | 68.42 | 7 passed, 487 warnings in 58.87s |
| tests/test_temporal_delta.py | 4 | 0 | 0 | 31.40 | 4 passed, 483 warnings in 25.39s |
| tests/test_mep_topology.py | 7 | 0 | 0 | 37.70 | 7 passed, 485 warnings in 31.54s |
| tests/test_material_takeoff_boq.py | 5 | 0 | 0 | 16.61 | 5 passed, 484 warnings in 10.17s |
| tests/test_life_safety_compliance.py | 4 | 0 | 0 | 11.88 | 4 passed, 484 warnings in 7.10s |
| tests/test_thermal_envelope_ashrae.py | 4 | 0 | 0 | 13.29 | 4 passed, 483 warnings in 7.42s |
| tests/test_universal_spatial_pipeline_rules11_12.py | 3 | 0 | 0 | 6.24 | 3 passed in 2.90s |
| tests/test_no_fake_backend_in_production.py | 2 | 0 | 0 | 6.18 | 2 passed in 3.02s |
| tests/test_clash_detection_bcf.py | 5 | 0 | 0 | 11.36 | 5 passed, 484 warnings in 7.91s |
| tests/test_collab_telepresence.py | 4 | 0 | 0 | 11.01 | 4 passed, 488 warnings in 8.15s |
| tests/test_section_elevation_drawings.py | 5 | 0 | 0 | 8.22 | 5 passed, 484 warnings in 4.67s |

Totals: 50 passed, 0 failed, 0 skipped. None of the eleven needed network or a GPU.
Over 60 s wall: `test_gaussian_splats.py` only (68.42 s). Under 60 s wall: the other ten.
