"""EXIF orientation and decoded-pixel cap at image intake (lifecycle gap #2)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from evidence.image_check import ImageTooLarge, inspect_image, open_oriented


def _rotated_jpeg(path, orientation=6):
    # Stored 200x100 landscape: left half red, right half blue.
    arr = np.zeros((100, 200, 3), np.uint8)
    arr[:, :100] = (255, 0, 0)
    arr[:, 100:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = orientation
    exif[0x010F] = "TestCam"
    Image.fromarray(arr).save(path, "JPEG", exif=exif, quality=95)
    return path


def test_orientation_6_is_applied_and_recorded(tmp_path):
    p = _rotated_jpeg(tmp_path / "phone.jpg")
    f = inspect_image(p)
    assert f.ok
    assert (f.stored_width, f.stored_height) == (200, 100)
    assert (f.width, f.height) == (100, 200)  # oriented, not stored
    assert f.exif_orientation == 6
    assert f.to_dict()["exif_orientation"] == 6

    img = open_oriented(p)
    assert img.size == (100, 200)
    top = np.asarray(img.convert("RGB"))[:50].mean(axis=(0, 1))
    assert top[0] > 200 and top[2] < 60  # stored-left (red) is now on top (CW rotate)


def test_no_orientation_tag_unchanged(tmp_path):
    p = tmp_path / "plain.png"
    Image.fromarray(np.full((100, 200, 3), 128, np.uint8)).save(p)
    f = inspect_image(p)
    assert (f.width, f.height) == (200, 100) == (f.stored_width, f.stored_height)
    assert f.exif_orientation is None


def test_oversize_refused_with_reason_not_crash(tmp_path, monkeypatch):
    p = tmp_path / "big.png"
    Image.fromarray(np.zeros((400, 400, 3), np.uint8)).save(p)
    monkeypatch.setenv("REALITY_ENGINE_MAX_IMAGE_PIXELS", "100000")  # 160k px > cap
    f = inspect_image(p)
    assert not f.ok and "decoded-pixel cap" in f.reason and "160,000" in f.reason
    assert f.max_pixels == 100_000
    with pytest.raises(ImageTooLarge):
        open_oriented(p)


def test_header_only_check_never_decodes(tmp_path, monkeypatch):
    # A huge header-declared size must be refused from the header alone.
    p = tmp_path / "huge.png"
    Image.new("1", (30000, 30000)).save(p)  # ~900 MP, tiny file
    f = inspect_image(p)
    assert not f.ok and ("cap" in f.reason or "bomb" in f.reason)
