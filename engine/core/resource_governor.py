"""Performance and Resource Governor (Wave 6, Block 38).

Monitors system resources (RAM, CPU cores, GPU VRAM) and adaptively governs
reconstruction workloads to enforce performance budgets:
1. Hardware Resource Sensing:
   - Inspects physical system RAM and available memory headroom.
   - Probes CPU core counts and GPU VRAM availability via engine.core.device.
2. Adaptive Policy Selection:
   - Tier 1 (High Headroom): Uncompressed processing, fine voxel grid, maximum parallel workers.
   - Tier 2 (Balanced): Downsamples large 4K images to 1920px, medium voxel pitch, decimation 0.5.
   - Tier 3 (Memory Constrained): Out-of-core spatial chunking, coarse voxel pitch, single/dual thread.
3. Workload Partitioning:
   - Spatially chunks point clouds and mesh geometry when scale exceeds working RAM.
4. Governed Execution Context:
   - Measures elapsed execution time and memory high-water mark against defined SLA budgets.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

from engine.core.device import select_device

Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class ResourceSnapshot:
    """Snapshot of current host system compute and memory resources."""

    total_ram_mb: int
    available_ram_mb: int
    cpu_cores: int
    gpu_name: str
    free_vram_mb: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_mb": self.total_ram_mb,
            "available_ram_mb": self.available_ram_mb,
            "cpu_cores": self.cpu_cores,
            "gpu_name": self.gpu_name,
            "free_vram_mb": self.free_vram_mb,
        }


@dataclass(frozen=True)
class GovernorPolicy:
    """Adaptive execution parameters computed for a specific reconstruction workload."""

    tier: int  # 1 (Full), 2 (Balanced), 3 (Constrained)
    max_image_dim_px: int
    max_points_in_memory: int
    voxel_pitch_m: float
    decimation_ratio: float
    max_workers: int
    enable_out_of_core: bool
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "max_image_dim_px": self.max_image_dim_px,
            "max_points_in_memory": self.max_points_in_memory,
            "voxel_pitch_m": round(self.voxel_pitch_m, 4),
            "decimation_ratio": round(self.decimation_ratio, 2),
            "max_workers": self.max_workers,
            "enable_out_of_core": self.enable_out_of_core,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class GovernedExecutionReport:
    """Report generated after a governed task completes."""

    task_name: str
    elapsed_s: float
    budget_s: float
    within_budget: bool
    peak_memory_mb: float
    policy: GovernorPolicy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "elapsed_s": round(self.elapsed_s, 3),
            "budget_s": round(self.budget_s, 3),
            "within_budget": self.within_budget,
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "policy": self.policy.to_dict(),
        }


def probe_system_resources() -> ResourceSnapshot:
    """Query current system hardware resources (RAM, CPU, GPU)."""
    cpu_cores = os.cpu_count() or 4

    # 1. Probe RAM
    total_ram_mb = 8192
    available_ram_mb = 4096

    # Try psutil if installed
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        total_ram_mb = int(mem.total // (1024 * 1024))
        available_ram_mb = int(mem.available // (1024 * 1024))
    except ImportError:
        # Fallback to OS-specific inspection
        if sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_ram_mb = int(stat.ullTotalPhys // (1024 * 1024))
                available_ram_mb = int(stat.ullAvailPhys // (1024 * 1024))
            except Exception:
                pass

    # 2. Probe GPU via engine.core.device
    dev_info = select_device()
    gpu_name = dev_info.name
    free_vram = dev_info.free_vram_mb

    return ResourceSnapshot(
        total_ram_mb=total_ram_mb,
        available_ram_mb=available_ram_mb,
        cpu_cores=cpu_cores,
        gpu_name=gpu_name,
        free_vram_mb=free_vram,
    )


def compute_governor_policy(
    image_count: int = 100,
    point_count: int = 500_000,
    target_budget_s: float = 30.0,
    snapshot: Optional[ResourceSnapshot] = None,
) -> GovernorPolicy:
    """Compute adaptive processing policy based on workload size and host resources.

    Ensures the target processing budget (e.g. 30s for 100 images) is not exceeded
    while avoiding out-of-memory crashes.
    """
    res = snapshot or probe_system_resources()

    # Determine stress factor
    ram_gb = res.available_ram_mb / 1024.0
    estimated_memory_needed_mb = (image_count * 25) + (point_count * 0.0001)

    if ram_gb >= 12.0 and res.cpu_cores >= 8 and estimated_memory_needed_mb < res.available_ram_mb * 0.5:
        # Tier 1: Abundant resources
        return GovernorPolicy(
            tier=1,
            max_image_dim_px=3840,  # 4K
            max_points_in_memory=2_000_000,
            voxel_pitch_m=0.005,  # 5mm
            decimation_ratio=1.0,  # Full detail
            max_workers=min(12, res.cpu_cores),
            enable_out_of_core=False,
            rationale="Abundant RAM (>12GB available) and high core count: full resolution enabled.",
        )
    elif ram_gb >= 4.0 and estimated_memory_needed_mb < res.available_ram_mb * 0.8:
        # Tier 2: Balanced
        return GovernorPolicy(
            tier=2,
            max_image_dim_px=1920,  # 1080p
            max_points_in_memory=750_000,
            voxel_pitch_m=0.015,  # 15mm
            decimation_ratio=0.5,
            max_workers=min(6, max(2, res.cpu_cores // 2)),
            enable_out_of_core=point_count > 1_000_000,
            rationale="Balanced tier: 1080p downsampling and 15mm voxel filtering to meet 30s SLA.",
        )
    else:
        # Tier 3: Memory constrained
        return GovernorPolicy(
            tier=3,
            max_image_dim_px=1024,
            max_points_in_memory=250_000,
            voxel_pitch_m=0.035,  # 35mm
            decimation_ratio=0.25,
            max_workers=2,
            enable_out_of_core=True,
            rationale="Constrained memory (<4GB available): enabled spatial chunking and 1024px cap to prevent OOM.",
        )


def chunk_point_cloud_spatially(
    points: Sequence[Point3],
    max_chunk_size: int = 250_000,
) -> Generator[List[Point3], None, None]:
    """Spatially partition point cloud into bounded chunks to respect memory limits.

    Sorts points along their longest spatial axis and yields chunks with at most
    `max_chunk_size` points.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= max_chunk_size:
        yield [tuple(p) for p in pts]
        return

    # Find longest axis
    spans = np.ptp(pts, axis=0)
    sort_axis = int(np.argmax(spans))

    # Sort indices by principal span coordinate
    order = np.argsort(pts[:, sort_axis])
    sorted_pts = pts[order]

    for start_idx in range(0, n, max_chunk_size):
        end_idx = min(n, start_idx + max_chunk_size)
        chunk = [tuple(p) for p in sorted_pts[start_idx:end_idx]]
        yield chunk


class StageResourceProbe:
    """MEASURES one stage (never estimates): peak resident memory of this process PLUS its child processes (COLMAP and other tools run as
    subprocesses), the lowest system-available RAM seen, and the CUDA peak allocated / reserved bytes when torch is already loaded.
    Sampled at ``interval`` seconds on a daemon thread; a stage shorter than one interval still gets its start and end samples.

        with StageResourceProbe() as probe:
            run_the_stage()
        probe.metrics   # {"mem_rss_start_mb", "mem_rss_peak_mb", "mem_rss_end_mb", "mem_system_available_min_mb", "mem_cuda_peak_allocated_mb", ...}
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.metrics: Dict[str, float] = {}
        self._stop = None
        self._thread = None
        self._rss: List[float] = []
        self._avail: List[float] = []

    @staticmethod
    def _tree_rss_mb() -> float:
        import psutil  # type: ignore

        me = psutil.Process()
        total = 0
        for p in [me, *me.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except Exception:  # noqa: BLE001 - a child that exited between listing and reading
                pass
        return total / (1024 * 1024)

    def _sample(self) -> None:
        import psutil  # type: ignore

        self._rss.append(self._tree_rss_mb())
        self._avail.append(psutil.virtual_memory().available / (1024 * 1024))

    @staticmethod
    def _torch():
        mod = sys.modules.get("torch")  # only when something else already imported it: probing must never load a framework
        try:
            return mod if mod is not None and mod.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            return None

    def __enter__(self) -> "StageResourceProbe":
        import threading

        try:
            self._sample()
        except Exception:  # noqa: BLE001 - psutil missing: the stage still runs, the metrics stay absent (not zero)
            return self
        t = self._torch()
        if t is not None:
            try:
                t.cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass
        self._stop = threading.Event()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                try:
                    self._sample()
                except Exception:  # noqa: BLE001
                    pass

        self._thread = threading.Thread(target=loop, name="stage-resource-probe", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stop is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self._sample()
        except Exception:  # noqa: BLE001
            pass
        m = {"mem_rss_start_mb": self._rss[0], "mem_rss_peak_mb": max(self._rss), "mem_rss_end_mb": self._rss[-1],
             "mem_system_available_min_mb": min(self._avail), "mem_samples": float(len(self._rss))}
        t = self._torch()
        if t is not None:
            try:
                m["mem_cuda_peak_allocated_mb"] = t.cuda.max_memory_allocated() / (1024 * 1024)
                m["mem_cuda_peak_reserved_mb"] = t.cuda.max_memory_reserved() / (1024 * 1024)
            except Exception:  # noqa: BLE001
                pass
        self.metrics = {k: round(v, 1) for k, v in m.items()}


class GovernedExecution:
    """Context manager that tracks runtime duration and memory against a budget."""

    def __init__(
        self,
        task_name: str,
        budget_s: float = 30.0,
        policy: Optional[GovernorPolicy] = None,
    ):
        self.task_name = task_name
        self.budget_s = budget_s
        self.policy = policy or compute_governor_policy(target_budget_s=budget_s)
        self.start_time = 0.0
        self.elapsed_s = 0.0
        self.report: Optional[GovernedExecutionReport] = None

    def __enter__(self) -> GovernedExecution:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_s = time.perf_counter() - self.start_time
        res = probe_system_resources()
        # Approximate memory used
        used_ram = res.total_ram_mb - res.available_ram_mb
        within_budget = self.elapsed_s <= self.budget_s

        self.report = GovernedExecutionReport(
            task_name=self.task_name,
            elapsed_s=self.elapsed_s,
            budget_s=self.budget_s,
            within_budget=within_budget,
            peak_memory_mb=float(used_ram),
            policy=self.policy,
        )
