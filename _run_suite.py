"""Scratch runner: launch 4 parallel pytest chunk processes over tests/test_*.py."""
import glob
import subprocess
import sys

files = sorted(glob.glob("tests/test_*.py"))
chunks = [files[i::4] for i in range(4)]
procs = []
logs = []
for i, chunk in enumerate(chunks):
    log = open(f"_suite_{i}.log", "w", encoding="utf-8")
    logs.append(log)
    procs.append(subprocess.Popen(
        [sys.executable, "-u", "-m", "pytest", "-q",
         "-p", "no:cacheprovider", "-m", "not slow", "--durations=10"] + chunk,
        stdout=log,
        stderr=subprocess.STDOUT,
    ))
rc = 0
for i, p in enumerate(procs):
    rc |= p.wait()
    logs[i].close()
    print(f"chunk {i} rc={p.returncode}")
sys.exit(rc)

