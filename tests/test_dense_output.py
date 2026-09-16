"""Tests for reconstruction/backend/dense_output.py (P6-01 remainder):
parsing COLMAP's dense stereo output (patch_match_stereo's
* geometric-consistency-fused point clouds, fused.ply) into this
repo's canonical ReconstructedPoint type, with the honesty rules the
repo requires:

- absent attributes (normals, colors, track ids) are absent -- no
  fabricated values, no silently-dropped semantics;
- a non-POINTCLOUD or non-binary PLY is a recorded parse failure,
  never a half-parsed cloud;
- the dense backend slot is honest about unavailability (patch_match
  needs the CUDA build; a CPU-only colmap reports the slot UNAVAILABLE,
  not a silent CPU fallback pretending to be MVS).
"""

import struct

import pytest

from reconstruction.backend.dense_output import (
    DenseOutputError,
    parse_fused_ply,
)
from reconstruction.backend.interface import ReconstructedPoint


class TestCrlfHeaders:
    def test_crlf_header_parses(self):
        # COLMAP (Windows) writes CRLF line endings; the original
        # parser partitioned on 'end_header\n' only and rejected every
        # real fused.ply (caught by the real-capture integration test
        # 2026-09-16; synthetic LF fixtures had hidden it).
        n = 3
        header = (
            "ply\r\n"
            "format binary_little_endian 1.0\r\n"
            f"element vertex {n}\r\n"
            "property float x\r\n"
            "property float y\r\n"
            "property float z\r\n"
            "end_header\r\n"
        ).encode("ascii")
        body = b"".join(
            struct.pack("<3f", float(i), 0.0, 0.0) for i in range(n)
        )
        pts = parse_fused_ply(header + body)
        assert len(pts) == n


def _binary_point_ply(points, *, with_colors=False, fmt="binary_little_endian"):
    """Build a minimal fused.ply-style binary PLY (vertex-only --
    COLMAP's fused output is a vertex cloud, no faces)."""
    props = ["float x", "float y", "float z"]
    if with_colors:
        props += ["uchar red", "uchar green", "uchar blue"]
    header_lines = [
        "ply",
        f"format {fmt} 1.0",
        f"element vertex {len(points)}",
    ]
    header_lines += [f"property {p}" for p in props]
    header_lines.append("end_header")
    header = ("\n".join(header_lines) + "\n").encode("ascii")
    body = bytearray()
    for pt in points:
        body += struct.pack("<3f", pt[0], pt[1], pt[2])
        if with_colors:
            body += struct.pack("<3B", pt[3], pt[4], pt[5])
    return header + bytes(body)


class TestParseFusedPly:
    def test_positions_parsed_to_canonical_points(self):
        data = _binary_point_ply([(0.0, 0.0, 1.0), (1.5, -2.0, 3.25)])
        points = parse_fused_ply(data, source_evidence_ids=["ev1", "ev2"])
        assert len(points) == 2
        assert all(isinstance(p, ReconstructedPoint) for p in points)
        assert points[0].position == pytest.approx((0.0, 0.0, 1.0))
        assert points[1].position == pytest.approx((1.5, -2.0, 3.25))

    def test_track_ids_are_dense_not_sfm_tracks(self):
        # Dense fused points have no SfM track; the track id must say
        # what it is (dense:<index>), never borrow a sparse concept.
        data = _binary_point_ply([(0.0, 0.0, 1.0)])
        points = parse_fused_ply(data, source_evidence_ids=["ev1"])
        assert points[0].track_id == "dense:0"

    def test_colors_in_pointcloud_are_dropped_with_note(self):
        # ReconstructedPoint carries position+track, not color; colors
        # present in fused.ply are real data -- silently discarding
        # them without record would be data loss, but fabricating a
        # color slot is worse. The parse returns points plus a facts
        # dict that records what was present-but-unused.
        data = _binary_point_ply([(0.0, 0.0, 1.0, 255, 128, 0)], with_colors=True)
        points, facts = parse_fused_ply(
            data, source_evidence_ids=["ev1"], return_facts=True
        )
        assert points[0].position == pytest.approx((0.0, 0.0, 1.0))
        assert facts["had_colors"] is True
        assert facts["n_points"] == 1

    def test_truncated_body_raises_recorded_error(self):
        data = _binary_point_ply([(0.0, 0.0, 1.0), (1.0, 1.0, 1.0)])
        with pytest.raises(DenseOutputError, match="truncated"):
            parse_fused_ply(data[:-4], source_evidence_ids=["ev1"])

    def test_ascii_ply_rejected_not_guessed(self):
        text = (
            b"ply\nformat ascii 1.0\nelement vertex 1\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"end_header\n0 0 1\n"
        )
        with pytest.raises(DenseOutputError, match="binary_little_endian"):
            parse_fused_ply(text, source_evidence_ids=["ev1"])

    def test_mesh_ply_rejected(self):
        # A fused.ply is a point cloud; a meshed PLY (with faces) fed
        # here is a caller error and must be named, not silently
        # vertex-dumped.
        data = _binary_point_ply([(0.0, 0.0, 1.0)])
        data += (
            b"element face 1\n"
            b"property list uchar int vertex_indices\n"
            b"end_header\n"
        )
        # NOTE: this stream has two end_headers; build a proper one instead.
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")
        body = struct.pack("<3f", 0.0, 0.0, 1.0) + struct.pack("<B3I", 3, 0, 0, 0)
        with pytest.raises(DenseOutputError, match="face"):
            parse_fused_ply(header + body, source_evidence_ids=["ev1"])
