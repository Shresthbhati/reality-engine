"""RE-PERC-000016: text on labels becomes attribute evidence with basis 'ocr'; ambiguity stays unknown."""
import numpy as np
import pytest

from perception.detection.ocr import Florence2Unavailable, match_attribute, read_text

LOCKS = {"keyhole": ["key lock"], "keypad": ["numpad", "number pad"], "unknown": []}


def test_single_match_is_taken_and_marked_ocr():
    r = match_attribute("Master KEYPAD 3000", LOCKS)
    assert r["value"] == "keypad" and r["basis"] == "ocr" and r["p"] is None


def test_synonym_matches_its_value():
    assert match_attribute("number-pad entry", LOCKS)["value"] == "keypad"


def test_no_match_and_conflict_stay_unknown_with_reason():
    none = match_attribute("EXIT", LOCKS)
    both = match_attribute("keyhole or keypad", LOCKS)
    assert none["value"] is None and "no allowed" in none["reason"]
    assert both["value"] is None and both["matches"] == ["keyhole", "keypad"]


def test_whole_words_only():
    assert match_attribute("keypads", LOCKS)["value"] is None


def test_read_text_reads_rendered_text():
    pytest.importorskip("torch", reason="torch is not installed (optional extra); Florence-2 needs it")
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 120), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((20, 20), "KEYPAD", fill="black", font=font)
    try:
        out = read_text(np.asarray(img))
    except Florence2Unavailable as exc:
        pytest.skip(str(exc))
    assert "KEYPAD" in out.upper()
