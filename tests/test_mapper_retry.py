"""The mapper is seeded (COLMAP's default is a fresh random seed per run) and retried when registration is weak."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from reconstruction.backend import colmap_backend as cb
from reconstruction.backend.interface import ReconstructionResult
from reconstruction.colmap_session import ColmapSession

N_INPUT = 10


def _backend(tmp_path, registered_per_attempt, monkeypatch):
    """A backend whose COLMAP steps are recorded and whose mapper 'registers' the scripted number of cameras."""
    backend = cb.ColmapReconstructionBackend(session=ColmapSession(tmp_path / "session"))
    steps, parses = [], []

    def fake_step(step, args, env):
        steps.append((step, list(args)))

    def fake_parse(sparse_dir, evidence_id_by_name, n_input, env):
        k = len(parses)
        parses.append(k)
        n = registered_per_attempt[min(k, len(registered_per_attempt) - 1)]
        (sparse_dir / "0").mkdir(exist_ok=True)
        (sparse_dir / "0" / "attempt.txt").write_text(str(k))          # which attempt a kept model came from
        poses = [SimpleNamespace(evidence_id=f"ev{i}") for i in range(n)]
        pts = [SimpleNamespace(track_id=f"t{i}") for i in range(100 + k)]
        return ReconstructionResult(points=pts, camera_poses=poses, registration_status="partial"), [sparse_dir / "0"], "0"

    monkeypatch.setattr(backend, "_run_step", fake_step)
    monkeypatch.setattr(backend, "_parse_sparse_dir", fake_parse)
    items = []
    for i in range(N_INPUT):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (64, 48)).save(p)
        items.append(SimpleNamespace(id=f"ev{i}", source_uri=p.as_uri(), metadata={}))
    return backend, steps, items


def _mapper_runs(steps):
    return [args for step, args in steps if step == "mapper"]


def _seed(args):
    return args[args.index("--Mapper.random_seed") + 1]


def test_a_good_first_attempt_runs_the_mapper_once_with_a_pinned_seed(tmp_path, monkeypatch):
    backend, steps, items = _backend(tmp_path, [9], monkeypatch)
    result = backend._reconstruct_in_session(items, env={}, gpu_flag="0")
    runs = _mapper_runs(steps)
    assert len(runs) == 1 and _seed(runs[0]) == "0"                        # reproducible: never COLMAP's random -1
    assert len(result.camera_poses) == 9
    assert [a["registered"] for a in backend.last_run_info["mapper_attempts"]] == [9]


def test_weak_registration_retries_with_the_next_seed_then_relaxed_thresholds(tmp_path, monkeypatch):
    backend, steps, items = _backend(tmp_path, [3, 4, 9], monkeypatch)
    result = backend._reconstruct_in_session(items, env={}, gpu_flag="0")
    runs = _mapper_runs(steps)
    assert [_seed(r) for r in runs] == ["0", "1", "2"]
    assert "--Mapper.init_min_tri_angle" not in runs[1] and "--Mapper.init_min_tri_angle" in runs[2]   # relaxed only last
    assert len(result.camera_poses) == 9
    assert [a["registered"] for a in backend.last_run_info["mapper_attempts"]] == [3, 4, 9]


def test_when_every_attempt_is_weak_the_best_one_is_kept(tmp_path, monkeypatch):
    backend, steps, items = _backend(tmp_path, [3, 6, 4], monkeypatch)
    result = backend._reconstruct_in_session(items, env={}, gpu_flag="0")
    assert len(_mapper_runs(steps)) == 3 and len(result.camera_poses) == 6
    kept = (backend.session.staging / "sparse" / "0" / "attempt.txt").read_text()
    assert kept == "1"                                                      # the model on disk is the 6-camera attempt
    assert not list((backend.session.staging).glob("sparse_full_try*"))    # scratch attempts are cleaned up
