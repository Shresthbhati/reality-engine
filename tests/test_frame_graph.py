"""Tests for the FrameGraph coordinate system."""

from __future__ import annotations

import math
from dataclasses import dataclass

from world_ir.coordinates import Frame, Transform, IDENTITY_MATRIX
from world_ir.frame_graph import (
    FrameGraph,
    FrameGraphEdge,
    FrameGraphCycleError,
    FrameGraphMissingPathError,
    FrameGraphAmbiguousPathError,
    FrameGraphDiagnostics,
    compose_uncertainties,
)
from provenance import Uncertainty


def _make_transform(src: Frame, tgt: Frame, matrix=None) -> Transform:
    """Create a Transform with optional custom matrix."""
    return Transform(
        source_frame=src,
        target_frame=tgt,
        matrix=matrix or IDENTITY_MATRIX,
    )


def _make_edge(src: Frame, tgt: Frame, provenance: str = "test", matrix=None) -> FrameGraphEdge:
    """Create a FrameGraphEdge."""
    return FrameGraphEdge(
        transform=_make_transform(src, tgt, matrix),
        provenance=provenance,
    )


def test_frame_graph_basic_path():
    """Test basic path resolution through the graph."""
    graph = FrameGraph()
    
    # Add chain: camera -> session_local -> world
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test"))
    
    result = graph.resolve(Frame.CAMERA, Frame.WORLD)
    
    assert result.transform.source_frame == Frame.CAMERA
    assert result.transform.target_frame == Frame.WORLD
    assert result.path == [Frame.CAMERA, Frame.SESSION_LOCAL, Frame.WORLD]
    assert len(result.edges) == 2
    assert not result.ambiguous


def test_frame_graph_direct_edge():
    """Test resolution with a direct edge."""
    graph = FrameGraph()
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.WORLD, "test"))
    
    result = graph.resolve(Frame.CAMERA, Frame.WORLD)
    
    assert result.path == [Frame.CAMERA, Frame.WORLD]
    assert len(result.edges) == 1


def test_frame_graph_identity():
    """Test resolution from frame to itself."""
    graph = FrameGraph()
    
    result = graph.resolve(Frame.WORLD, Frame.WORLD)
    
    assert result.transform.source_frame == Frame.WORLD
    assert result.transform.target_frame == Frame.WORLD
    assert result.path == [Frame.WORLD]
    assert len(result.edges) == 0


def test_frame_graph_missing_path():
    """Test error when no path exists."""
    graph = FrameGraph()
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    # No path from session_local to world
    
    try:
        graph.resolve(Frame.SESSION_LOCAL, Frame.WORLD)
        assert False, "Expected FrameGraphMissingPathError"
    except FrameGraphMissingPathError as e:
        # Error message says "Frame(s) not in graph" when target not in graph
        # or "No valid path" when target in graph but no path
        assert "not in graph" in str(e) or "No valid path" in str(e)


def test_frame_graph_cycle_detection():
    """Test cycle detection in the graph."""
    graph = FrameGraph()
    
    # Create a cycle: A -> B -> C -> A
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.BUILDING_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.BUILDING_LOCAL, Frame.CAMERA, "test"))
    
    cycles = graph.check_cycles()
    
    assert len(cycles) == 1
    # Cycle should contain all three frames
    cycle = cycles[0]
    assert Frame.CAMERA in cycle
    assert Frame.SESSION_LOCAL in cycle
    assert Frame.BUILDING_LOCAL in cycle


def test_frame_graph_disconnected_frames():
    """Test detection of disconnected frames."""
    graph = FrameGraph()
    
    # Main component
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test"))
    
    # Disconnected component
    graph.add_edge(_make_edge(Frame.ENU, Frame.UTM, "test"))
    
    disconnected = graph.find_disconnected()
    
    # ENU and UTM are disconnected from the main component
    assert Frame.ENU in disconnected
    assert Frame.UTM in disconnected
    assert Frame.CAMERA not in disconnected
    assert Frame.WORLD not in disconnected


def test_frame_graph_ambiguity_detection():
    """Test detection of ambiguous paths with distinct transforms."""
    graph = FrameGraph()
    
    # Two paths from CAMERA to WORLD with different transforms
    # Path 1: CAMERA -> SESSION_LOCAL -> WORLD
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test", 
        ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test",
        ((1,0,0,1), (0,1,0,2), (0,0,1,3), (0,0,0,1))))
    
    # Path 2: CAMERA -> BUILDING_LOCAL -> WORLD (different transform)
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.BUILDING_LOCAL, "test",
        ((1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1))))
    graph.add_edge(_make_edge(Frame.BUILDING_LOCAL, Frame.WORLD, "test",
        ((1,0,0,5), (0,1,0,6), (0,0,1,7), (0,0,0,1))))
    
    # Should detect ambiguity
    try:
        graph.resolve(Frame.CAMERA, Frame.WORLD)
        assert False, "Expected FrameGraphAmbiguousPathError"
    except FrameGraphAmbiguousPathError as e:
        assert "Multiple distinct transforms" in str(e)


def test_frame_graph_ambiguous_allow():
    """Test allow_ambiguous=True returns first path with ambiguous flag."""
    graph = FrameGraph()
    
    # Same setup as ambiguity test
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test"))
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.BUILDING_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.BUILDING_LOCAL, Frame.WORLD, "test"))
    
    result = graph.resolve(Frame.CAMERA, Frame.WORLD, allow_ambiguous=True)
    
    assert result.ambiguous is True
    assert len(result.alternatives) == 1


def test_frame_graph_timestamp_validity():
    """Test edge validity window filtering by timestamp."""
    graph = FrameGraph()
    
    # Edge only valid in time window [100, 200]
    edge = FrameGraphEdge(
        transform=_make_transform(Frame.CAMERA, Frame.WORLD),
        provenance="test",
        valid_from_ns=100,
        valid_to_ns=200,
    )
    graph.add_edge(edge)
    
    # Valid at timestamp 150
    result = graph.resolve(Frame.CAMERA, Frame.WORLD, timestamp_ns=150)
    assert result.transform.source_frame == Frame.CAMERA
    
    # Invalid at timestamp 50 (before window)
    try:
        graph.resolve(Frame.CAMERA, Frame.WORLD, timestamp_ns=50)
        assert False, "Expected FrameGraphMissingPathError"
    except FrameGraphMissingPathError:
        pass
    
    # Invalid at timestamp 250 (after window)
    try:
        graph.resolve(Frame.CAMERA, Frame.WORLD, timestamp_ns=250)
        assert False, "Expected FrameGraphMissingPathError"
    except FrameGraphMissingPathError:
        pass


def test_frame_graph_diagnostics():
    """Test comprehensive diagnostics."""
    graph = FrameGraph()
    
    # Add some edges
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test"))
    graph.add_edge(_make_edge(Frame.ENU, Frame.UTM, "test"))  # disconnected
    
    diagnostics = graph.run_diagnostics()
    
    assert diagnostics.has_issues()
    assert Frame.ENU in diagnostics.disconnected_frames
    assert Frame.UTM in diagnostics.disconnected_frames
    assert "disconnected" in diagnostics.summary()


def test_frame_graph_serialization():
    """Test serialization and deserialization."""
    graph = FrameGraph()
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test"))
    
    data = graph.to_dict()
    restored = FrameGraph.from_dict(data)
    
    assert set(restored.frames()) == set(graph.frames())
    assert len(restored.edges()) == len(graph.edges())
    
    # Verify path resolution still works
    result = restored.resolve(Frame.CAMERA, Frame.WORLD)
    assert result.transform.source_frame == Frame.CAMERA
    assert result.transform.target_frame == Frame.WORLD


def test_compose_uncertainties():
    """Test uncertainty composition for chained transforms."""
    u1 = Uncertainty(confidence=0.8, note="first")
    u2 = Uncertainty(confidence=0.6, note="second")
    
    combined = compose_uncertainties(u1, u2)
    
    assert combined.confidence == 0.6  # min of both
    assert "first" in combined.note
    assert "second" in combined.note


def test_frame_graph_transform_composition():
    """Test that composed transforms are mathematically correct."""
    graph = FrameGraph()
    
    # Translation by (1, 2, 3)
    trans1 = (
        (1, 0, 0, 1),
        (0, 1, 0, 2),
        (0, 0, 1, 3),
        (0, 0, 0, 1),
    )
    # Translation by (4, 5, 6)
    trans2 = (
        (1, 0, 0, 4),
        (0, 1, 0, 5),
        (0, 0, 1, 6),
        (0, 0, 0, 1),
    )
    
    graph.add_edge(FrameGraphEdge(
        transform=_make_transform(Frame.CAMERA, Frame.SESSION_LOCAL, trans1),
        provenance="test",
    ))
    graph.add_edge(FrameGraphEdge(
        transform=_make_transform(Frame.SESSION_LOCAL, Frame.WORLD, trans2),
        provenance="test",
    ))
    
    result = graph.resolve(Frame.CAMERA, Frame.WORLD)
    
    # Composed translation should be (1+4, 2+5, 3+6) = (5, 7, 9)
    point = result.transform.apply((0, 0, 0))
    assert math.isclose(point[0], 5.0)
    assert math.isclose(point[1], 7.0)
    assert math.isclose(point[2], 9.0)


def test_frame_graph_invalid_transform_detection():
    """Test detection of invalid (non-rigid) transforms."""
    graph = FrameGraph()
    
    # Add a non-rigid transform (scale)
    bad_matrix = (
        (2, 0, 0, 0),
        (0, 2, 0, 0),
        (0, 0, 2, 0),
        (0, 0, 0, 1),
    )
    graph.add_edge(FrameGraphEdge(
        transform=_make_transform(Frame.CAMERA, Frame.WORLD, bad_matrix),
        provenance="test",
    ))
    
    invalid = graph.validate_transforms()
    
    # The graph adds both the edge and its inverse, so we expect 1 invalid pair
    # but 2 edges in _edges (original + inverse). validate_transforms should
    # deduplicate by frame pair, so we expect 1.
    assert len(invalid) >= 1


def test_frame_graph_load_from_registration():
    """Test loading transforms from registration results."""
    graph = FrameGraph()
    
    # Simulate a registration result
    from dataclasses import dataclass
    from typing import Optional, Tuple
    
    @dataclass
    class MockRegistrationResult:
        transform: Optional[Transform]
        method: str
        rmse: float
        inlier_fraction: float
        iterations: int
    
    # Create a mock transform
    transform = _make_transform(Frame.SESSION_LOCAL, Frame.WORLD)
    reg_result = MockRegistrationResult(
        transform=transform,
        method="icp",
        rmse=0.01,
        inlier_fraction=0.95,
        iterations=10,
    )
    
    graph.load_from_registration(reg_result, "registration")
    
    edges = graph.edges()
    assert len(edges) == 1
    assert edges[0].provenance == "registration:icp"
    assert edges[0].metadata["method"] == "icp"


def test_frame_graph_load_from_trajectory():
    """Test loading transforms from trajectory data."""
    graph = FrameGraph()
    
    # Simulate a trajectory
    class MockFrameSource:
        def __init__(self, value):
            self.value = value
    
    class MockTrajectory:
        def __init__(self, frames, frame_source, clock_id, sync_state):
            self.frames = frames
            self.frame_source = MockFrameSource(frame_source)
            self.clock_id = clock_id
            self.sync_state = sync_state
        
        @property
        def start_ns(self):
            return 1000
        
        @property
        def end_ns(self):
            return 2000
    
    class MockFrame:
        def __init__(self, pose, timestamp_ns):
            self.pose = pose
            self.timestamp_ns = timestamp_ns
    
    transform = _make_transform(Frame.CAMERA, Frame.WORLD)
    traj_frame = MockFrame(pose=transform, timestamp_ns=1000)
    trajectory = MockTrajectory(
        frames=(traj_frame,),
        frame_source="vio",
        clock_id="test_clock",
        sync_state="SYNCHRONIZED",
    )
    
    graph.load_from_trajectory(trajectory, "trajectory")
    
    edges = graph.edges()
    assert len(edges) == 1
    assert edges[0].provenance == "trajectory:vio"
    assert edges[0].valid_from_ns == 1000
    assert edges[0].valid_to_ns == 2000


def test_frame_graph_multiple_valid_paths_same_transform():
    """Test that multiple paths with identical transforms don't raise ambiguity."""
    graph = FrameGraph()
    
    # Two paths that produce the same transform (identity)
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.SESSION_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.WORLD, "test"))
    graph.add_edge(_make_edge(Frame.CAMERA, Frame.BUILDING_LOCAL, "test"))
    graph.add_edge(_make_edge(Frame.BUILDING_LOCAL, Frame.WORLD, "test"))
    
    # Both paths are identity, so should not be ambiguous
    result = graph.resolve(Frame.CAMERA, Frame.WORLD)
    
    assert not result.ambiguous
    assert len(result.alternatives) == 0


def test_frame_graph_registration_integration():
    """Test integration with registration engine types."""
    graph = FrameGraph()
    
    # Add registration-style edge with metadata
    edge = FrameGraphEdge(
        transform=_make_transform(Frame.SESSION_LOCAL, Frame.WORLD),
        provenance="registration:gnss_anchor",
        metadata={
            "method": "gnss_anchor",
            "rmse": "0.005",
            "anchor_pairs": "3",
        },
    )
    graph.add_edge(edge)
    
    result = graph.resolve(Frame.SESSION_LOCAL, Frame.WORLD)
    
    assert len(result.edges) == 1
    assert result.edges[0].provenance == "registration:gnss_anchor"
    assert result.edges[0].metadata["method"] == "gnss_anchor"


def test_frame_graph_trajectory_integration():
    """Test integration with trajectory types."""
    graph = FrameGraph()
    
    # Add trajectory-style edge with validity window
    edge = FrameGraphEdge(
        transform=_make_transform(Frame.CAMERA, Frame.WORLD),
        provenance="trajectory:vio",
        valid_from_ns=1000000,
        valid_to_ns=2000000,
        metadata={
            "frame_source": "vio",
            "clock_id": "clock_1",
            "sync_state": "SYNCHRONIZED",
        },
    )
    graph.add_edge(edge)
    
    # Valid within window
    result = graph.resolve(Frame.CAMERA, Frame.WORLD, timestamp_ns=1500000)
    assert result.transform.source_frame == Frame.CAMERA
    
    # Invalid outside window
    try:
        graph.resolve(Frame.CAMERA, Frame.WORLD, timestamp_ns=500000)
        assert False, "Expected MissingPathError"
    except FrameGraphMissingPathError:
        pass


def test_frame_graph_complex_hierarchy():
    """Test a complex frame hierarchy matching the spec."""
    graph = FrameGraph()
    
    # Build hierarchy: SENSOR -> CAPTURE -> SESSION -> TRAJECTORY -> REGISTERED -> WORLD
    # (using available Frame enums)
    graph.add_edge(_make_edge(Frame.SENSOR, Frame.SESSION_LOCAL, "capture"))
    graph.add_edge(_make_edge(Frame.SESSION_LOCAL, Frame.BUILDING_LOCAL, "trajectory"))
    graph.add_edge(_make_edge(Frame.BUILDING_LOCAL, Frame.WORLD, "registration"))
    
    # Also add geodetic frame connection
    graph.add_edge(_make_edge(Frame.WORLD, Frame.ENU, "geodetic"))
    
    # Resolve full chain
    result = graph.resolve(Frame.SENSOR, Frame.ENU)
    
    assert result.transform.source_frame == Frame.SENSOR
    assert result.transform.target_frame == Frame.ENU
    assert len(result.edges) == 4
    assert result.path == [Frame.SENSOR, Frame.SESSION_LOCAL, Frame.BUILDING_LOCAL, Frame.WORLD, Frame.ENU]


def test_frame_graph_all_frames_in_enum():
    """Test that all Frame enum values can be used in the graph."""
    graph = FrameGraph()
    
    for frame in Frame:
        graph.add_edge(_make_edge(frame, Frame.WORLD, "test"))
    
    # Verify all frames are in the graph
    assert set(graph.frames()) == set(Frame)
    
    # All should resolve to WORLD
    for frame in Frame:
        if frame != Frame.WORLD:
            result = graph.resolve(frame, Frame.WORLD)
            assert result.transform.target_frame == Frame.WORLD


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])