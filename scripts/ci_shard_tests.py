"""Split EVERY test file into N balanced shards, so CI can run the whole suite instead of a hand-kept allowlist.

    python scripts/ci_shard_tests.py --shard 3 --of 8      # prints shard 3's test files, one per line
    python scripts/ci_shard_tests.py --of 8 --check        # verifies the shards partition tests/ exactly

Deterministic (size-weighted greedy, ties broken by path) so a shard's content is the same on every runner. A new
``tests/**/test_*.py`` file is picked up automatically: nothing can be forgotten by omission.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def all_test_files(root: Path = Path("tests")) -> list[str]:
    return sorted(p.as_posix() for p in root.rglob("test_*.py"))


def shards(files: list[str], n: int, base: Path = Path(".")) -> list[list[str]]:
    """Greedy balance by file size (a cheap proxy for run time): biggest first, always onto the lightest shard.
    ``files`` are posix paths relative to ``base`` (default: the current directory)."""
    if n < 1:
        raise ValueError("need at least one shard")
    size = {f: max((base / f).stat().st_size, 1) for f in files}
    loads = [0] * n
    out: list[list[str]] = [[] for _ in range(n)]
    for f in sorted(files, key=lambda f: (-size[f], f)):
        i = min(range(n), key=lambda k: (loads[k], k))
        out[i].append(f)
        loads[i] += size[f]
    return [sorted(s) for s in out]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--of", type=int, required=True, help="number of shards")
    ap.add_argument("--shard", type=int, help="1-based shard to print")
    ap.add_argument("--check", action="store_true", help="verify the shards cover every test file exactly once")
    args = ap.parse_args(argv)
    files = all_test_files()
    parts = shards(files, args.of)
    if args.check:
        flat = [f for s in parts for f in s]
        if sorted(flat) != files or len(flat) != len(set(flat)):
            print("shards do not partition the test files", file=sys.stderr)
            return 1
        print(f"OK: {len(files)} test files in {args.of} shards ({', '.join(str(len(s)) for s in parts)})")
        return 0
    if not args.shard or not 1 <= args.shard <= args.of:
        ap.error("--shard must be between 1 and --of")
    # Not print(): on Windows it would emit "\r\n" and leave a stray "\r" on every path in the manifest.
    sys.stdout.buffer.write(("\n".join(parts[args.shard - 1]) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
