from engine.core.clock import DeterministicClock


def test_time_is_exact_tick_times_dt():
    clock = DeterministicClock(dt=0.1, seed=42)
    for _ in range(1000):
        clock.advance()
    assert clock.tick == 1000
    assert clock.time == 1000 * 0.1


def test_run_yields_n_plus_one_contexts():
    clock = DeterministicClock(dt=0.5, seed=1)
    contexts = list(clock.run(5))
    assert len(contexts) == 6
    assert [c.tick for c in contexts] == [0, 1, 2, 3, 4, 5]
    assert [c.time for c in contexts] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]


def test_determinism_same_seed_same_sequence():
    a = list(DeterministicClock(dt=0.25, seed=7).run(10))
    b = list(DeterministicClock(dt=0.25, seed=7).run(10))
    assert a == b


def test_seed_is_carried_but_does_not_affect_time():
    a = DeterministicClock(dt=0.1, seed=1)
    b = DeterministicClock(dt=0.1, seed=2)
    a.advance()
    b.advance()
    assert a.time == b.time
    assert a.context().seed != b.context().seed


def test_rejects_nonpositive_dt():
    import pytest

    with pytest.raises(ValueError):
        DeterministicClock(dt=0.0)
