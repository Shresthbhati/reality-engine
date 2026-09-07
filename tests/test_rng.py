from engine.core.rng import DeterministicRNG


def test_same_seed_same_sequence():
    a = DeterministicRNG(seed=123, name="fracture")
    b = DeterministicRNG(seed=123, name="fracture")
    seq_a = [a.uniform() for _ in range(20)]
    seq_b = [b.uniform() for _ in range(20)]
    assert seq_a == seq_b


def test_different_names_diverge():
    a = DeterministicRNG(seed=123, name="fracture")
    b = DeterministicRNG(seed=123, name="fire")
    assert [a.uniform() for _ in range(5)] != [b.uniform() for _ in range(5)]


def test_child_streams_are_independent_but_reproducible():
    parent1 = DeterministicRNG(seed=1)
    parent2 = DeterministicRNG(seed=1)
    child1 = parent1.child("solver_a")
    child2 = parent2.child("solver_a")
    assert [child1.uniform() for _ in range(10)] == [child2.uniform() for _ in range(10)]


def test_reset_replays_same_sequence():
    rng = DeterministicRNG(seed=99)
    first = [rng.uniform() for _ in range(10)]
    rng.reset()
    second = [rng.uniform() for _ in range(10)]
    assert first == second
