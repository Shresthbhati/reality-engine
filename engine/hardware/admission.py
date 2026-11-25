"""Memory-aware stage admission (Workstream 4).

Before a stage runs, ``StageAdmission.admit`` reads available system RAM and free VRAM and compares them with the stage's
declared estimate (``STAGE_NEEDS``; estimates, to be replaced by measured values from ``job_stage_runs`` -- see
``scripts/profile_stage_rss.py``). The outcome is always explicit and recorded on the stage result:

* ``admit``  -- enough memory; the stage runs.
* ``degrade`` -- not enough for the normal path but enough for the DECLARED fallback; the fallback and its quality effect are
  recorded as a stage warning (the stage becomes ``degraded`` -> job ``partial``), never a silent weaker result.
* ``defer`` -- not enough right now; wait (collecting garbage) up to ``max_wait_s`` for memory to free up.
* ``fail``  -- still not enough after the wait; the stage fails with the measured numbers in its error. A stage is never
  silently skipped.

Modes (``RE_MEMORY_ADMISSION``): ``enforce`` (default), ``record`` (measure and record only, never refuse), ``off``.
Reading memory never imports torch (which itself costs hundreds of MB): VRAM comes from an already-imported torch, else
``nvidia-smi``, else unknown.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

MB = 1024 * 1024


@dataclass(frozen=True)
class StageNeed:
    ram_mb: float = 0.0
    vram_mb: float = 0.0
    #: optional declared fallback the STAGE honours (``ctx.state["admission"][stage]["fallback"]``); used when the normal path does not fit
    fallback: str | None = None
    fallback_ram_mb: float = 0.0
    fallback_vram_mb: float = 0.0
    quality_effect: str | None = None


#: ESTIMATES (MB) for the coarse in-process stages. ``reconstruct`` carries the multi-view rung and the ladder's fine stages
#: (the fine-stage rows only report), and a real run peaked at 3.69 GB RSS in total. Replace with measured numbers.
STAGE_NEEDS: dict[str, StageNeed] = {
    "evidence_resolve": StageNeed(ram_mb=150),
    "evidence_quality": StageNeed(ram_mb=400),
    "reconstruct": StageNeed(ram_mb=2500),
    "compile_validate": StageNeed(ram_mb=600),
    "changeset_commit": StageNeed(ram_mb=200),
}


@dataclass
class MemorySnapshot:
    avail_ram_mb: float | None
    rss_mb: float | None
    free_vram_mb: float | None
    vram_source: str = "unknown"


@dataclass
class Admission:
    stage: str
    decision: str  # admit | degrade | fail   (``defer`` is internal: it resolves to one of these)
    reason: str
    fallback: str | None = None
    quality_effect: str | None = None
    waited_s: float = 0.0
    snapshot: MemorySnapshot | None = None
    metrics: dict[str, float] = field(default_factory=dict)


# ---- readers (never raise; None = unknown) ----------------------------------------------------------------------------

def _win_counters():
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    class MEMSTATUS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

    return ctypes, PMC, MEMSTATUS


def read_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / MB
    except Exception:  # noqa: BLE001 -- psutil missing or failing: fall through to the platform readers
        pass
    try:
        if sys.platform == "win32":
            ctypes, PMC, _ = _win_counters()
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / MB
            return None
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / MB
    except Exception:  # noqa: BLE001
        return None


def read_available_ram_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / MB
    except Exception:  # noqa: BLE001
        pass
    try:
        if sys.platform == "win32":
            ctypes, _, MEMSTATUS = _win_counters()
            st = MEMSTATUS()
            st.dwLength = ctypes.sizeof(MEMSTATUS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullAvailPhys / MB
            return None
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def read_free_vram_mb() -> tuple[float | None, str]:
    """(free MB, source). Never imports torch; an already-imported CUDA torch wins, else nvidia-smi, else unknown."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / MB, "torch"
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3, check=True).stdout.strip().splitlines()
        return float(out[0].strip()), "nvidia-smi"
    except Exception:  # noqa: BLE001 -- no NVIDIA driver / tool: unknown, not zero
        return None, "unknown"


def snapshot(want_vram: bool = True) -> MemorySnapshot:
    """``want_vram=False`` skips the VRAM probe (``nvidia-smi`` spawns a process): only stages that declare a VRAM need pay for it."""
    free_vram, src = read_free_vram_mb() if want_vram else (None, "not probed")
    return MemorySnapshot(read_available_ram_mb(), read_rss_mb(), free_vram, src)


def release_resources() -> None:
    """After a stage: collect garbage and return cached CUDA blocks (only if torch is already loaded -- never import it here)."""
    gc.collect()
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- releasing is best effort; it must never fail a stage
        pass


class RssSampler:
    """Samples process RSS while a stage runs, so the recorded peak is the stage's, not the process lifetime's."""

    def __init__(self, interval_s: float = 0.1, reader: Callable[[], float | None] = read_rss_mb) -> None:
        self._interval, self._reader = interval_s, reader
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        v = self._reader()
        if v is not None and (self.peak_mb is None or v > self.peak_mb):
            self.peak_mb = v

    def __enter__(self) -> "RssSampler":
        self._sample()

        def loop() -> None:
            while not self._stop.wait(self._interval):
                self._sample()

        self._thread = threading.Thread(target=loop, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()


# ---- policy -----------------------------------------------------------------------------------------------------------

def _shortfall(need_ram: float, need_vram: float, snap: MemorySnapshot, reserve_mb: float) -> list[str]:
    out = []
    if need_ram > 0 and snap.avail_ram_mb is not None and snap.avail_ram_mb - reserve_mb < need_ram:
        out.append(f"RAM: need {need_ram:.0f} MB + {reserve_mb:.0f} MB reserve, {snap.avail_ram_mb:.0f} MB available")
    if need_vram > 0:
        if snap.free_vram_mb is None:
            out.append(f"VRAM: need {need_vram:.0f} MB but no GPU memory reading is available ({snap.vram_source})")
        elif snap.free_vram_mb < need_vram:
            out.append(f"VRAM: need {need_vram:.0f} MB, {snap.free_vram_mb:.0f} MB free")
    return out


class StageAdmission:
    def __init__(self, *, mode: str = "enforce", needs: dict[str, StageNeed] | None = None, reserve_mb: float = 768.0,
                 max_wait_s: float = 30.0, poll_s: float = 2.0, reader: Callable[[], MemorySnapshot] | None = None,
                 sleep: Callable[[float], None] = time.sleep, release: Callable[[], None] = release_resources,
                 sampler_interval_s: float = 0.1) -> None:
        self.mode, self.needs, self.reserve_mb = mode, STAGE_NEEDS if needs is None else needs, reserve_mb
        self.max_wait_s, self.poll_s, self._reader, self._sleep, self.release = max_wait_s, poll_s, reader, sleep, release
        self.sampler_interval_s = sampler_interval_s

    @classmethod
    def from_env(cls, **kw) -> "StageAdmission | None":
        mode = (os.environ.get("RE_MEMORY_ADMISSION") or "enforce").strip().lower()
        if mode == "off":
            return None
        if mode not in ("enforce", "record"):
            mode = "enforce"

        def num(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except ValueError:
                return default

        return cls(mode=mode, reserve_mb=num("RE_MEMORY_RESERVE_MB", 768.0), max_wait_s=num("RE_ADMISSION_MAX_WAIT_S", 30.0), **kw)

    def after_stage(self, stage: str, rss_before: float | None, rss_after: float | None, failed: bool) -> bool:
        """Release resources after a stage. A full ``gc.collect`` holds the GIL (it stalls the API's event loop for hundreds of ms on
        a large heap), so it runs only when it can matter: a failed stage, a heavy stage (declared need >= 500 MB RAM or any VRAM) or
        measured RSS growth over 150 MB. Returns whether it released."""
        need = self.needs.get(stage, StageNeed())
        grew = rss_before is not None and rss_after is not None and rss_after - rss_before > 150
        if failed or grew or need.ram_mb >= 500 or need.vram_mb > 0:
            self.release()
            return True
        return False

    def admit(self, stage: str) -> Admission:
        need = self.needs.get(stage, StageNeed())
        self._read = self._reader or (lambda: snapshot(want_vram=need.vram_mb > 0 or need.fallback_vram_mb > 0))
        snap = self._read()
        waited = 0.0
        problems = _shortfall(need.ram_mb, need.vram_mb, snap, self.reserve_mb)
        metrics = {"admission_need_ram_mb": need.ram_mb, "admission_need_vram_mb": need.vram_mb}
        if self.mode == "record" or not problems:
            adm = Admission(stage, "admit", "within budget" if not problems else f"record mode, would refuse: {'; '.join(problems)}")
        else:
            adm = None
            while adm is None:
                if need.fallback and not _shortfall(need.fallback_ram_mb, need.fallback_vram_mb, snap, self.reserve_mb):
                    adm = Admission(stage, "degrade", f"normal path does not fit ({'; '.join(problems)}); using fallback {need.fallback!r}",
                                    need.fallback, need.quality_effect or "reduced quality; see fallback")
                    break
                if waited >= self.max_wait_s:
                    adm = Admission(stage, "fail", f"insufficient memory after waiting {waited:.0f}s: {'; '.join(problems)}")
                    break
                self.release()                       # a stage that just ended may still hold memory
                self._sleep(self.poll_s)
                waited += self.poll_s
                snap = self._read()
                problems = _shortfall(need.ram_mb, need.vram_mb, snap, self.reserve_mb)
                if not problems:
                    adm = Admission(stage, "admit", f"admitted after waiting {waited:.0f}s for memory")
        adm.waited_s, adm.snapshot = waited, snap
        for k, v in (("avail_ram_before_mb", snap.avail_ram_mb), ("rss_before_mb", snap.rss_mb), ("free_vram_before_mb", snap.free_vram_mb)):
            if v is not None:
                metrics[k] = round(v, 1)
        metrics["admission_waited_s"] = waited
        adm.metrics = metrics
        return adm
