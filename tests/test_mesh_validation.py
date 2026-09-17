"""Test matrix for reconstruction/meshing/validation.py: known-bad meshes
each assert the validator catches the specific defect with the right
count, plus a known-good closed mesh asserting a clean report."""

from reconstruction.meshing.mesh import MeshData
from reconstruction.meshing.validation import validate_mesh

TETRA_VERTS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
TETRA_FACES = (
    (0, 1, 2),
    (0, 1, 3),
    (1, 2, 3),
    (0, 2, 3),
)


def test_known_good_closed_tetrahedron_is_clean():
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    report = validate_mesh(mesh)
    assert report.n_degenerate_triangles == 0
    assert report.n_non_manifold_edges == 0
    assert report.n_boundary_edges == 0  # closed manifold -- no holes
    assert report.n_connected_components == 1
    assert report.candidate_self_intersection_pairs == ()


def test_degenerate_triangle_detected():
    # Third face's vertices are collinear -> zero area.
    verts = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),  # collinear with vertex 0 and 1
    )
    faces = ((0, 1, 2), (0, 1, 3))
    mesh = MeshData(vertices=verts, faces=faces)
    report = validate_mesh(mesh)
    assert report.n_degenerate_triangles == 1
    assert report.degenerate_triangle_indices == (1,)


def test_non_manifold_edge_detected():
    # Three triangles fanned around the shared edge (0,1) -- that edge is
    # referenced by 3 faces instead of the manifold-expected 1 or 2.
    verts = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    faces = ((0, 1, 2), (0, 1, 3), (0, 1, 4))
    mesh = MeshData(vertices=verts, faces=faces)
    report = validate_mesh(mesh)
    assert report.n_non_manifold_edges == 1
    assert report.non_manifold_edges == ((0, 1),)


def test_disconnected_components_detected():
    # Two separate triangles sharing no vertices -> 2 components.
    verts = (
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (10.0, 0.0, 0.0), (11.0, 0.0, 0.0), (10.0, 1.0, 0.0),
    )
    faces = ((0, 1, 2), (3, 4, 5))
    mesh = MeshData(vertices=verts, faces=faces)
    report = validate_mesh(mesh)
    assert report.n_connected_components == 2


def test_hole_boundary_edges_detected():
    # Tetrahedron missing one face -> the 3 edges of the missing face are
    # each referenced by exactly 1 triangle (boundary/hole edges).
    faces = TETRA_FACES[:3]  # drop the (0, 2, 3) face
    mesh = MeshData(vertices=TETRA_VERTS, faces=faces)
    report = validate_mesh(mesh)
    assert report.has_holes
    assert report.n_boundary_edges == 3
    assert set(report.boundary_edges) == {(0, 2), (0, 3), (2, 3)}


def test_bad_normal_detected():
    verts = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    faces = ((0, 1, 2),)
    normals = (
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),  # zero-length -- bad
        (float("nan"), 0.0, 0.0),  # NaN -- bad
    )
    mesh = MeshData(vertices=verts, faces=faces, normals=normals)
    report = validate_mesh(mesh)
    assert report.n_bad_normals == 2
    assert set(report.bad_normal_vertex_indices) == {1, 2}


def test_candidate_self_intersection_overlapping_bboxes():
    # Two non-adjacent triangles whose bounding boxes overlap in space.
    verts = (
        (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0),
        (1.0, 1.0, -1.0), (3.0, 1.0, -1.0), (1.0, 3.0, 1.0),
    )
    faces = ((0, 1, 2), (3, 4, 5))
    mesh = MeshData(vertices=verts, faces=faces)
    report = validate_mesh(mesh)
    assert report.candidate_self_intersection_pairs == ((0, 1),)


def test_report_to_dict_has_expected_keys():
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    d = validate_mesh(mesh).to_dict()
    for key in (
        "n_triangles", "n_vertices", "n_degenerate_triangles",
        "n_non_manifold_edges", "n_boundary_edges", "has_holes",
        "n_bad_normals", "n_connected_components", "extent_m",
    ):
        assert key in d


# -- bad-scale check (declared expectation; never inferred) ----------------

def test_no_expected_extent_means_scale_fields_absent_not_guessed():
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    report = validate_mesh(mesh)
    assert report.expected_extent_m is None
    assert report.scale_ratio_per_axis is None
    assert report.scale_tolerance is None
    assert report.scale_within_tolerance is None


def test_matching_expected_extent_is_within_tolerance():
    # Tetra extent is exactly (1, 1, 1); declare the same -> ratios 1.0.
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    report = validate_mesh(mesh, expected_extent_m=(1.0, 1.0, 1.0))
    assert report.scale_ratio_per_axis == (1.0, 1.0, 1.0)
    assert report.scale_within_tolerance is True
    assert report.scale_tolerance == 1.10  # default recorded


def test_half_scale_reconstruction_is_caught():
    # A mesh built at half the declared size: every axis ratio is 0.5.
    scaled = tuple(tuple(c * 0.5 for c in v) for v in TETRA_VERTS)
    mesh = MeshData(vertices=scaled, faces=TETRA_FACES)
    report = validate_mesh(mesh, expected_extent_m=(1.0, 1.0, 1.0))
    assert report.scale_ratio_per_axis == (0.5, 0.5, 0.5)
    assert report.scale_within_tolerance is False


def test_mm_vs_m_unit_mixup_signature_is_caught():
    # A mesh measured in mm but interpreted as m: ratio ~1000 per axis.
    scaled = tuple(tuple(c * 1000.0 for c in v) for v in TETRA_VERTS)
    mesh = MeshData(vertices=scaled, faces=TETRA_FACES)
    report = validate_mesh(mesh, expected_extent_m=(1.0, 1.0, 1.0))
    assert report.scale_ratio_per_axis == (1000.0, 1000.0, 1000.0)
    assert report.scale_within_tolerance is False


def test_per_axis_tolerance_one_bad_axis_fails_the_check():
    # Two axes fine, one axis off by 1.5x -> overall check fails.
    verts = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.5, 0.0),
        (0.0, 0.0, 1.0),
    )
    mesh = MeshData(vertices=verts, faces=TETRA_FACES)
    report = validate_mesh(mesh, expected_extent_m=(1.0, 1.0, 1.0))
    rx, ry, rz = report.scale_ratio_per_axis
    assert (rx, ry, rz) == (1.0, 1.5, 1.0)
    assert report.scale_within_tolerance is False


def test_custom_tolerance_is_recorded_and_respected():
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    report = validate_mesh(
        mesh, expected_extent_m=(1.0, 1.0, 1.0), scale_tolerance=2.0
    )
    assert report.scale_tolerance == 2.0
    # Half-scale fits inside a 2.0x tolerance.
    assert report.scale_within_tolerance is True


def test_bad_scale_fields_roundtrip_through_to_dict():
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    d = validate_mesh(mesh, expected_extent_m=(1.0, 1.0, 1.0)).to_dict()
    assert d["expected_extent_m"] == [1.0, 1.0, 1.0]
    assert d["scale_ratio_per_axis"] == [1.0, 1.0, 1.0]
    assert d["scale_tolerance"] == 1.10
    assert d["scale_within_tolerance"] is True
    d_none = validate_mesh(mesh).to_dict()
    assert d_none["expected_extent_m"] is None
    assert d_none["scale_within_tolerance"] is None


def test_invalid_expectations_rejected_loudly():
    mesh = MeshData(vertices=TETRA_VERTS, faces=TETRA_FACES)
    import pytest

    with pytest.raises(ValueError):
        validate_mesh(mesh, expected_extent_m=(1.0, 1.0))  # not a 3-tuple
    with pytest.raises(ValueError):
        validate_mesh(mesh, expected_extent_m=(1.0, 0.0, 1.0))  # non-positive
    with pytest.raises(ValueError):
        validate_mesh(mesh, expected_extent_m=(1.0, 1.0, 1.0), scale_tolerance=0.0)
