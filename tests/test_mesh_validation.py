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
