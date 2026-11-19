from __future__ import annotations

from perception.detection.extensible_prompter import (
    ExtensiblePrompter,
    ScaleFilter,
    get_extensible_prompter,
)


def test_extensible_prompter_indexes_universal_ontology() -> None:
    prompter = ExtensiblePrompter()
    all_prompts = prompter.get_prompts()
    assert len(all_prompts) > 50

    # Verify MEP & electrical objects are indexed
    assert "fuse box" in all_prompts or "conduit" in all_prompts
    assert "smoke detector" in all_prompts

    # Verify architecture details are indexed
    assert "door handle" in all_prompts or "skirting board" in all_prompts

    # Verify outdoor infrastructure is indexed
    assert "bollard" in all_prompts or "manhole cover" in all_prompts


def test_extensible_prompter_scale_filtering() -> None:
    prompter = ExtensiblePrompter()

    micro_prompts = prompter.get_prompts(scale=ScaleFilter.MICRO)
    # Small items like door handle, screw, switch should be micro
    assert any("handle" in p or "switch" in p or "socket" in p or "detector" in p for p in micro_prompts)

    macro_prompts = prompter.get_prompts(scale=ScaleFilter.MACRO)
    # Structural elements should be macro
    assert any("wall" in p or "floor" in p or "ceiling" in p or "building" in p for p in macro_prompts)


def test_extensible_prompter_resolves_finegrained_and_parts() -> None:
    prompter = ExtensiblePrompter()

    # Direct resolution of finegrained entity
    target_fuse = prompter.resolve_prediction("fuse box")
    assert target_fuse is not None
    assert target_fuse.canonical_label == "fuse_box"

    # Direct resolution of subpart
    target_part = prompter.resolve_prediction("chair leg")
    assert target_part is not None
    assert "leg" in target_part.canonical_label

    # Case insensitive resolution
    target_smoke = prompter.resolve_prediction("SMOKE DETECTOR")
    assert target_smoke is not None
    assert target_smoke.canonical_label == "smoke_detector"


def test_singleton_extensible_prompter() -> None:
    p1 = get_extensible_prompter()
    p2 = get_extensible_prompter()
    assert p1 is p2
