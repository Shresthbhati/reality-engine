from types import SimpleNamespace as NS

from apps.api.jobs import _drop_exact_duplicates


def _ev(i, checksum):
    return NS(id=i, checksum=checksum)


def test_identical_checksums_keep_first_and_report_rest():
    rows = [(_ev("a", "h1"), "pa"), (_ev("b", "h2"), "pb"), (_ev("c", "h1"), "pc")]
    kept, dropped = _drop_exact_duplicates(rows)
    assert [e.id for e, _ in kept] == ["a", "b"]
    assert dropped == [{"evidence_id": "c", "reason": "byte-identical duplicate of a"}]


def test_missing_checksum_is_never_treated_as_duplicate():
    rows = [(_ev("a", None), "pa"), (_ev("b", None), "pb")]
    kept, dropped = _drop_exact_duplicates(rows)
    assert len(kept) == 2 and not dropped
