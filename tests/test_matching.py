"""reconstruction.matching: the IMatcher interface, SIFT backend, LightGlue backend, and registry honesty."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from reconstruction.matching import MatcherUnavailableError, available_matchers, get_matcher
from reconstruction.matching.interface import IMatcher
from reconstruction.scene_identity import SiftPairVerifier


def _textured(seed: int, size=(480, 640)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (rng.random(size) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    for _ in range(120):  # blobs and rectangles give SIFT/DISK stable, distinctive keypoints
        x, y = int(rng.integers(20, size[1] - 20)), int(rng.integers(20, size[0] - 20))
        cv2.circle(img, (x, y), int(rng.integers(3, 14)), int(rng.integers(0, 255)), -1)
        cv2.rectangle(img, (x, y), (x + int(rng.integers(5, 30)), y + int(rng.integers(5, 30))), int(rng.integers(0, 255)), 2)
    return img


@pytest.fixture()
def pair(tmp_path):
    a = _textured(1)
    h, w = a.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 6.0, 1.0)
    m[:, 2] += (18, 9)
    b = cv2.warpAffine(a, m, (w, h))
    other = _textured(99)  # an unrelated scene
    paths = {}
    for name, img in (("a", a), ("b", b), ("c", other)):
        p = tmp_path / f"{name}.png"
        cv2.imwrite(str(p), img)
        paths[name] = p
    return paths


def test_sift_matcher_satisfies_the_interface_and_verifies_a_true_pair(pair):
    m = get_matcher("sift", pair)
    assert isinstance(m, IMatcher) and m.name == "sift@opencv" and not m.non_commercial
    r = m.match("a", "b")
    assert r.inliers >= 30 and r.inlier_ratio > 0.5
    assert r.points_a.shape == r.points_b.shape == (r.inliers, 2)
    assert r.matcher == "sift@opencv" and r.runtime_s > 0


def test_sift_unrelated_pair_has_far_fewer_inliers(pair):
    m = get_matcher("sift", pair)
    assert m.match("a", "c").inliers < m.match("a", "b").inliers / 3


def test_pair_verifier_still_returns_the_same_evidence_shape(pair):
    ev = SiftPairVerifier(pair)("a", "b")
    direct = get_matcher("sift", pair).match("a", "b")
    assert (ev.a, ev.b, ev.matches, ev.inliers) == ("a", "b", direct.matches, direct.inliers)


def test_lightglue_matcher_satisfies_the_interface_and_verifies_a_true_pair(pair):
    m = get_matcher("lightglue", pair)
    assert isinstance(m, IMatcher) and m.name == "lightglue@disk" and not m.non_commercial
    r = m.match("a", "b")
    assert r.inliers >= 100 and r.inlier_ratio > 0.7
    assert r.points_a.shape == r.points_b.shape == (r.inliers, 2)
    assert r.matcher == "lightglue@disk" and r.runtime_s > 0


def test_lightglue_unrelated_pair_has_far_fewer_inliers(pair):
    m = get_matcher("lightglue", pair)
    r_ab = m.match("a", "b")
    r_ac = m.match("a", "c")
    assert r_ac.inliers < r_ab.inliers / 5


def test_registry_is_honest_about_unbuilt_and_non_commercial_matchers(monkeypatch):
    monkeypatch.delenv("REALITY_ALLOW_NC_MODELS", raising=False)
    matchers = available_matchers()
    assert matchers["sift"] == "available"
    assert matchers["lightglue"] == "available"
    with pytest.raises(MatcherUnavailableError, match="not implemented"):
        get_matcher("loftr", {})
    with pytest.raises(MatcherUnavailableError, match="non-commercial"):
        get_matcher("mast3r", {})
    monkeypatch.setenv("REALITY_ALLOW_NC_MODELS", "1")
    with pytest.raises(MatcherUnavailableError, match="not implemented"):  # the flag lifts the licence gate, not the missing backend
        get_matcher("mast3r", {})
    with pytest.raises(MatcherUnavailableError, match="unknown matcher"):
        get_matcher("nope", {})


def test_learned_matchers_never_masquerade_as_classical_fallbacks(monkeypatch, pair):
    """xfeat/dedode report the true state; absent upstream they raise instead of silently running AKAZE/BRISK."""
    from reconstruction.matching import dedode, xfeat

    monkeypatch.setattr(xfeat, "probe", lambda: (False, "xfeat repo not cached"))
    monkeypatch.setattr(dedode, "probe", lambda: (False, "dedode weights not cached"))
    ms = available_matchers()
    assert ms["xfeat"] == "xfeat repo not cached" and ms["dedode"] == "dedode weights not cached"
    for name in ("xfeat", "dedode"):
        with pytest.raises(MatcherUnavailableError):
            get_matcher(name, pair)


def test_classical_matchers_are_registered_under_their_true_names(pair):
    ms = available_matchers()
    assert ms["orb"] == "available"  # present in every OpenCV build
    for name, expected in (("orb", "orb@opencv"), ("akaze", "akaze@opencv"), ("brisk", "brisk@opencv")):
        if ms[name] != "available":  # some OpenCV builds (e.g. 5.x) lack AKAZE/BRISK: must say so, not fake it
            with pytest.raises(MatcherUnavailableError, match="OpenCV"):
                get_matcher(name, pair)
            continue
        m = get_matcher(name, pair)
        assert m.name == expected
        r = m.match("a", "b")
        assert r.matcher == expected and "xfeat" not in r.matcher and "dedode" not in r.matcher
        assert r.inliers > 0
