"""Seeded deterministic RNG (spec sec 90 DETERMINISM: deterministic seed).

Wraps Python's Mersenne Twister so every consumer that needs randomness
takes an explicit seed and produces byte-identical sequences across runs
and platforms -- required for golden tests and replay.
"""

from __future__ import annotations

import random


class DeterministicRNG:
    """A named, seeded RNG stream.

    `name` lets a simulation derive independent, reproducible sub-streams
    (e.g. one per solver) from a single world seed without solvers
    stepping on each other's draws.
    """

    def __init__(self, seed: int, name: str = "default"):
        self.seed = seed
        self.name = name
        self._derived_seed = self._derive_seed(seed, name)
        self._rng = random.Random(self._derived_seed)

    @staticmethod
    def _derive_seed(seed: int, name: str) -> int:
        # Stable, platform-independent derivation (no hash() -- that's
        # salted per-process in CPython and would break determinism).
        h = 1469598103934665603  # FNV offset basis
        for ch in f"{seed}:{name}":
            h ^= ord(ch)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return h

    def child(self, name: str) -> "DeterministicRNG":
        """Derive an independent, reproducible sub-stream."""
        return DeterministicRNG(self.seed, f"{self.name}/{name}")

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return self._rng.uniform(lo, hi)

    def randint(self, lo: int, hi: int) -> int:
        return self._rng.randint(lo, hi)

    def choice(self, seq):
        return self._rng.choice(seq)

    def reset(self) -> None:
        self._rng = random.Random(self._derived_seed)
