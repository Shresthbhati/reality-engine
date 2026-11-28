"""Diagnose the 5 failing tests in test_spatial_budget.py."""
import datetime
import numpy as np

print("=== DIAGNOSTIC ===\n")

# Issue A: gsd_m vs gsd_mm
print("[A] Checking DetailBudget attribute names...")
import perception.quuality.detail_budget as db
b = db.DetailBudget(
    point_count=10,
    mean_gsd_mm_per_px=2.0,
    mean_distance_m=5.0,
    resolution_px=2000,
    view_count=4,
    occlusion_ratio=0.0,
    blur_score=0.1,
    texture_amplitude=0.5,
)
print(f"    Has gsd_m: {hasattr(b, 'gsd_m')}")
print(f"    Has gsd_mm: {hasattr(b, 'gsd_mm')}")
if hasattr(b, 'gsd_mm'):
    print(f"    gsd_mm value: {b.gsd_mm}")
print()

# Issue B: float32 round-trip
print("[B] Checking float32 round-trip in _split_partitions_by_meters...")
import perception.quuality.spatial_budget as sb
raw_f32 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
raw_f64 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
o32 = sb._split_partitions_by_meters(raw_f32, 0.3)
o64 = sb._split_partitions_by_meters(raw_f64, 0.3)
print(f"    float32 input -> output dtype: {o32.point_ids.dtype}")
print(f"    float64 input -> output dtype: {o64.point_ids.dtype}")
print()

# Issue C: changed_after at module load
print("[C] checking changed_after timing...")
# Compute what changed_after would be at module load
now = datetime.datetime.now(datetime.timezone.utc)
changed_after = now - datetime.timedelta(seconds=1)
task_started = datetime.datetime(2026, 1, 15, 11, 59, 59, tzinfo=datetime.timezone.utc)
scan_rec_times = [
    datetime.datetime(2026, 1, 15, 12, 0, 0, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 1, 15, 12, 0, 1, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 1, 15, 12, 0, 2, tzinfo=datetime.timezone.utc),
    datetime.datetime(2026, 1, 15, 12, 0, 3, tzinfo=datetime.timezone.utc),
]
print(f"    Module-load changed_after: {changed_after}")
print(f"    Test task_started:          {task_started}")
print(f"    Test scan_rec times:        {scan_rec_times[0]} ... {scan_rec_times[-1]}")
print(f"    changed_after > all scan_rec times? {all(changed_after > t for t in scan_rec_times)}")
print(f"    -> If True: ALL rows match 'changed after', not just 1")
print(f"    -> The test expects exactly 1 row (the last one at 12:00:03)")
print()

# Issue D: Check if _test_single_point_projection_returns_real_gsd uses gsd_m
print("[D] Scanning test file for gsd_m usage...")
with open('tests/test_spatial_budget.py') as f:
    for i, line in enumerate(f, 1):
        if 'gsd_m' in line and 'gsd_mm' not in line:
            print(f"    Line {i}: {line.rstrip()}")
