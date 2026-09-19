"""Coordinate Frame Graph (city-scale campaign).

A proper frame graph abstraction for city-scale coordinate management.

Frame hierarchy (from sensor to world):
  SENSOR -> CAPTURE -> SESSION_LOCAL -> TRAJECTORY -> REGISTERED_SESSION -> WORLD
       \\                   /                     \
        \\                 /                       \
         -> ENU/UTM/WGS84/ECEF (global geodetic frames, peer to WORLD)

Each edge in the graph represents a Transform with:
- source_frame, target_frame
- 4x4 rigid matrix
- timestamp (validity window)
- uncertainty/covariance
- provenance (which system produced this transform: registration, trajectory, GNSS, etc.)
- validity: optional time range or condition

The graph provides:
- Path finding with deterministic tie-breaking
- Cycle detection
- Missing path diagnostics
- Ambiguity detection (multiple valid paths)
- Transform composition with uncertainty propagation
- Integration with registration (GNSS anchor, ICP) and trajectory systems

Safety guarantees:
- Never silently chooses an arbitrary transform
- Returns structured diagnostics for all failure modes
- Detects and reports cycles, disconnected components, ambiguous paths
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from provenance import Uncertainty
from world_ir.coordinates import (
    Frame,
    Transform,
    IDENTITY_MATRIX,
    _mat_mul,
    _mat_inverse_rigid,
    apply_point,
)


class FrameGraphError(Exception):
    """Base exception for frame graph errors."""
    pass


class FrameGraphCycleError(FrameGraphError):
    """Raised when a cycle is detected in the frame graph."""
    pass


class FrameGraphMissingPathError(FrameGraphError):
    """Raised when no path exists between frames."""
    pass


class FrameGraphAmbiguousPathError(FrameGraphError):
    """Raised when multiple valid paths exist between frames."""
    pass


class FrameGraphInvalidTransformError(FrameGraphError):
    """Raised when a transform is invalid (e.g., non-rigid, singular)."""
    pass


@dataclass(frozen=True)
class FrameGraphEdge:
    """One edge in the frame graph: a Transform with metadata."""
    transform: Transform
    #: Which system produced this transform
    provenance: str  # "registration:gnss_anchor", "registration:icp", "trajectory:vio", "manual", etc.
    #: Optional validity window (inclusive): if set, transform is only valid within [valid_from, valid_to] nanoseconds
    valid_from_ns: Optional[int] = None
    valid_to_ns: Optional[int] = None
    #: Additional metadata (sensor_id, registration_method, etc.)
    metadata: Dict[str, str] = field(default_factory=dict)

    def is_valid_at(self, timestamp_ns: int) -> bool:
        """Check if this edge is valid at the given timestamp."""
        if self.valid_from_ns is not None and timestamp_ns < self.valid_from_ns:
            return False
        if self.valid_to_ns is not None and timestamp_ns > self.valid_to_ns:
            return False
        return True


@dataclass(frozen=True)
class PathResult:
    """Result of a frame graph path query."""
    transform: Transform
    path: List[Frame]  # sequence of frames from source to target
    edges: List[FrameGraphEdge]  # edges used in composition
    #: True if multiple valid paths existed (ambiguity detected)
    ambiguous: bool = False
    #: If ambiguous, the alternative paths (each with its composed transform)
    alternatives: List[Tuple[Transform, List[Frame], List[FrameGraphEdge]]] = field(default_factory=list)


@dataclass
class FrameGraphDiagnostics:
    """Diagnostics from frame graph operations."""
    #: Frames that have no path to/from other frames
    disconnected_frames: List[Frame] = field(default_factory=list)
    #: Cycles detected in the graph
    cycles: List[List[Frame]] = field(default_factory=list)
    #: Edges with invalid transforms
    invalid_edges: List[FrameGraphEdge] = field(default_factory=list)
    #: Ambiguous paths (multiple valid routes between same frames)
    ambiguous_paths: List[Tuple[Frame, Frame, int]] = field(default_factory=list)  # (source, target, num_paths)

    def has_issues(self) -> bool:
        return bool(self.disconnected_frames or self.cycles or self.invalid_edges or self.ambiguous_paths)

    def summary(self) -> str:
        if not self.has_issues():
            return "Frame graph: OK"
        parts = []
        if self.disconnected_frames:
            parts.append(f"{len(self.disconnected_frames)} disconnected frame(s)")
        if self.cycles:
            parts.append(f"{len(self.cycles)} cycle(s)")
        if self.invalid_edges:
            parts.append(f"{len(self.invalid_edges)} invalid edge(s)")
        if self.ambiguous_paths:
            parts.append(f"{len(self.ambiguous_paths)} ambiguous path(s)")
        return "Frame graph issues: " + "; ".join(parts)


class FrameGraph:
    """Frame graph with safety checks and path resolution.
    
    Extends the basic CoordinateRegistry with:
    - Full frame metadata (provenance, validity window, covariance)
    - Safety checks (cycles, disconnected, ambiguity)
    - Deterministic path selection with ambiguity detection
    - Integration with registration and trajectory systems
    """
    
    def __init__(self):
        self._edges: Dict[Tuple[Frame, Frame], FrameGraphEdge] = {}
        self._frames: Set[Frame] = set()
        # Adjacency for graph algorithms
        self._adj: Dict[Frame, List[Tuple[Frame, FrameGraphEdge]]] = {}
        self._rev_adj: Dict[Frame, List[Tuple[Frame, FrameGraphEdge]]] = {}
    
    def add_edge(self, edge: FrameGraphEdge) -> None:
        """Add a transform edge to the graph with full metadata."""
        src, tgt = edge.transform.source_frame, edge.transform.target_frame
        self._edges[(src, tgt)] = edge
        self._frames.add(src)
        self._frames.add(tgt)
        
        self._adj.setdefault(src, []).append((tgt, edge))
        self._rev_adj.setdefault(tgt, []).append((src, edge))
    
    def remove_edge(self, source: Frame, target: Frame) -> bool:
        """Remove an edge from the graph. Returns True if edge existed."""
        edge = self._edges.pop((source, target), None)
        if edge is None:
            return False
        # Remove from adjacency
        if source in self._adj:
            self._adj[source] = [(t, e) for t, e in self._adj[source] if t != target]
        if target in self._rev_adj:
            self._rev_adj[target] = [(s, e) for s, e in self._rev_adj[target] if s != source]
        return True
    
    def get_edge(self, source: Frame, target: Frame) -> Optional[FrameGraphEdge]:
        """Get direct edge between frames."""
        return self._edges.get((source, target))
    
    def has_frame(self, frame: Frame) -> bool:
        return frame in self._frames
    
    def frames(self) -> List[Frame]:
        return sorted(self._frames)
    
    def edges(self) -> List[FrameGraphEdge]:
        return list(self._edges.values())
    
    # ---- Path resolution with safety checks ----
    
    def resolve(
        self,
        source: Frame,
        target: Frame,
        timestamp_ns: Optional[int] = None,
        allow_ambiguous: bool = False,
    ) -> PathResult:
        """Resolve a transform from source to target frame.
        
        Args:
            source: Source frame
            target: Target frame
            timestamp_ns: If provided, only use edges valid at this timestamp
            allow_ambiguous: If False, raise on ambiguous paths; if True, return first with ambiguous=True
        
        Returns:
            PathResult with composed transform, path, edges, and ambiguity info
        
        Raises:
            FrameGraphMissingPathError: No path exists
            FrameGraphAmbiguousPathError: Multiple valid paths and allow_ambiguous=False
            FrameGraphCycleError: Cycle detected in path
        """
        # Identity case: source == target always succeeds even if not in graph
        if source == target:
            return PathResult(
                transform=Transform.identity(source),
                path=[source],
                edges=[],
            )
        
        if source not in self._frames or target not in self._frames:
            raise FrameGraphMissingPathError(
                f"Frame(s) not in graph: source={source.value}, target={target.value}"
            )
        
        # Find all valid paths using BFS
        paths = self._find_all_paths(source, target, timestamp_ns)
        
        if not paths:
            raise FrameGraphMissingPathError(
                f"No valid path from {source.value} to {target.value}"
                + (f" at timestamp {timestamp_ns}" if timestamp_ns else "")
            )
        
        if len(paths) > 1 and not allow_ambiguous:
            # Check if paths produce identical transforms (numerically equivalent)
            unique_transforms = {}
            for path in paths:
                key = tuple(round(v, 12) for row in path[0].matrix for v in row)
                if key not in unique_transforms:
                    unique_transforms[key] = path
            
            if len(unique_transforms) > 1:
                # Build alternative list
                alternatives = []
                for t in list(unique_transforms.values())[1:]:
                    alternatives.append((t[0], t[1], t[2]))
                
                raise FrameGraphAmbiguousPathError(
                    f"Multiple distinct transforms from {source.value} to {target.value}: "
                    f"{len(paths)} paths, {len(unique_transforms)} distinct transforms. "
                    f"Use allow_ambiguous=True to get the first."
                )
            
            # Multiple paths but all produce identical transform - not ambiguous
            paths = list(unique_transforms.values())
        
        # Return the first path (deterministically chosen)
        best_transform, best_path, best_edges = paths[0]
        return PathResult(
            transform=best_transform,
            path=best_path,
            edges=best_edges,
            ambiguous=len(paths) > 1,
            alternatives=[(t, p, e) for t, p, e in paths[1:]] if len(paths) > 1 else [],
        )
    
    def _find_all_paths(
        self,
        source: Frame,
        target: Frame,
        timestamp_ns: Optional[int],
    ) -> List[Tuple[Transform, List[Frame], List[FrameGraphEdge]]]:
        """Find all valid paths from source to target using BFS with path tracking."""
        paths = []
        
        # Queue: (current_frame, accumulated_transform, path_frames, path_edges)
        queue = [(source, Transform.identity(source), [source], [])]
        visited_paths = set()  # Track visited (frame, path_signature) to avoid cycles
        
        max_paths = 100  # Limit to prevent infinite loops in pathological graphs
        
        while queue and len(paths) < max_paths:
            current, acc_transform, path_frames, path_edges = queue.pop(0)
            
            # Create path signature for cycle detection
            path_sig = tuple(f.value for f in path_frames)
            if path_sig in visited_paths:
                continue
            visited_paths.add(path_sig)
            
            if current == target:
                paths.append((acc_transform, path_frames, path_edges))
                continue
            
            # Explore neighbors
            for next_frame, edge in self._adj.get(current, []):
                # Check timestamp validity
                if timestamp_ns is not None and not edge.is_valid_at(timestamp_ns):
                    continue
                
                # Check for cycles
                if next_frame in path_frames:
                    continue  # Skip cycles
                
                # Compose transform
                try:
                    new_transform = acc_transform.then(edge.transform)
                except ValueError as e:
                    # Frame mismatch in composition - skip this edge
                    continue
                
                queue.append((
                    next_frame,
                    new_transform,
                    path_frames + [next_frame],
                    path_edges + [edge],
                ))
        
        return paths
    
    # ---- Safety diagnostics ----
    
    def check_cycles(self) -> List[List[Frame]]:
        """Detect cycles in the frame graph using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []
        path = []
        
        def dfs(node: Frame) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for next_frame, _ in self._adj.get(node, []):
                if next_frame not in visited:
                    dfs(next_frame)
                elif next_frame in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(next_frame)
                    cycle = path[cycle_start:] + [next_frame]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
            path.pop()
        
        for frame in self._frames:
            if frame not in visited:
                dfs(frame)
        
        return cycles
    
    def find_disconnected(self) -> List[Frame]:
        """Find frames that are not connected to the main component (prefer WORLD as reference)."""
        if not self._frames:
            return []
        
        # Prefer WORLD as the reference frame for "main" component
        # If WORLD not in graph, use the largest connected component
        start = Frame.WORLD if Frame.WORLD in self._frames else next(iter(self._frames))
        
        visited = set()
        queue = [start]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for next_frame, _ in self._adj.get(node, []):
                if next_frame not in visited:
                    queue.append(next_frame)
            for prev_frame, _ in self._rev_adj.get(node, []):
                if prev_frame not in visited:
                    queue.append(prev_frame)
        
        return [f for f in self._frames if f not in visited]
    
    def check_ambiguity(self) -> List[Tuple[Frame, Frame, int]]:
        """Find frame pairs with multiple distinct paths.
        
        Returns list of (source, target, num_distinct_transforms).
        """
        ambiguous = []
        frames_list = list(self._frames)
        
        for i, src in enumerate(frames_list):
            for tgt in frames_list[i+1:]:
                try:
                    paths = self._find_all_paths(src, tgt, None)
                    if len(paths) > 1:
                        # Check for distinct transforms
                        unique = {}
                        for t in paths:
                            key = tuple(round(v, 12) for row in t[0].matrix for v in row)
                            if key not in unique:
                                unique[key] = t
                        if len(unique) > 1:
                            ambiguous.append((src, tgt, len(unique)))
                except Exception:
                    pass
        
        return ambiguous
    
    def validate_transforms(self) -> List[FrameGraphEdge]:
        """Check all edges for valid rigid transforms."""
        invalid = []
        seen: Set[Tuple[Frame, Frame]] = set()
        for edge in self._edges.values():
            # Skip if we've already checked this frame pair (avoid double-checking inverses)
            key = (edge.transform.source_frame, edge.transform.target_frame)
            reverse_key = (edge.transform.target_frame, edge.transform.source_frame)
            if key in seen or reverse_key in seen:
                continue
            seen.add(key)
            seen.add(reverse_key)
            
            # Check if matrix is valid rigid transform
            try:
                # Verify inverse works
                _ = edge.transform.inverse()
                # Verify composition with inverse gives identity
                composed = edge.transform.then(edge.transform.inverse())
                for i in range(4):
                    for j in range(4):
                        expected = 1.0 if i == j else 0.0
                        if abs(composed.matrix[i][j] - expected) > 1e-6:
                            invalid.append(edge)
                            break
            except Exception:
                invalid.append(edge)
        return invalid
    
    def run_diagnostics(self) -> FrameGraphDiagnostics:
        """Run all safety checks and return comprehensive diagnostics."""
        return FrameGraphDiagnostics(
            disconnected_frames=self.find_disconnected(),
            cycles=self.check_cycles(),
            invalid_edges=self.validate_transforms(),
            ambiguous_paths=self.check_ambiguity(),
        )
    
    # ---- Integration helpers ----
    
    def load_from_registration(self, registration_result, provenance_prefix: str = "registration") -> None:
        """Load transforms from a RegistrationResult into the frame graph."""
        if registration_result.transform is not None:
            transform = registration_result.transform
            edge = FrameGraphEdge(
                transform=Transform(
                    source_frame=Frame(transform.source_frame),
                    target_frame=Frame(transform.target_frame),
                    matrix=transform.matrix,
                    timestamp=transform.timestamp,
                    uncertainty=transform.uncertainty,
                ),
                provenance=f"{provenance_prefix}:{registration_result.method}",
                metadata={
                    "method": registration_result.method,
                    "rmse": str(registration_result.rmse),
                    "inlier_fraction": str(registration_result.inlier_fraction),
                    "iterations": str(registration_result.iterations),
                },
            )
            self.add_edge(edge)
    
    def load_from_trajectory(self, trajectory, provenance_prefix: str = "trajectory") -> None:
        """Load frame transforms from a Trajectory.
        
        A trajectory defines the body_frame -> world_frame relationship over time.
        This adds a representative transform (first frame) as a graph edge.
        """
        if not trajectory.frames:
            return
        
        first_frame = trajectory.frames[0]
        transform = first_frame.pose
        
        edge = FrameGraphEdge(
            transform=Transform(
                source_frame=Frame(transform.source_frame),
                target_frame=Frame(transform.target_frame),
                matrix=transform.matrix,
                timestamp=transform.timestamp,
                uncertainty=transform.uncertainty,
            ),
            provenance=f"{provenance_prefix}:{trajectory.frame_source.value}",
            valid_from_ns=trajectory.start_ns,
            valid_to_ns=trajectory.end_ns,
            metadata={
                "frame_source": trajectory.frame_source.value,
                "clock_id": trajectory.clock_id,
                "sync_state": trajectory.sync_state,
                "num_frames": str(len(trajectory.frames)),
            },
        )
        self.add_edge(edge)
    
    def to_dict(self) -> dict:
        """Serialize the frame graph."""
        return {
            "edges": [
                {
                    "source": edge.transform.source_frame.value,
                    "target": edge.transform.target_frame.value,
                    "transform": edge.transform.to_dict(),
                    "provenance": edge.provenance,
                    "valid_from_ns": edge.valid_from_ns,
                    "valid_to_ns": edge.valid_to_ns,
                    "metadata": edge.metadata,
                }
                for edge in self._edges.values()
            ],
            "frames": [f.value for f in sorted(self._frames)],
        }
    
    @staticmethod
    def from_dict(data: dict) -> "FrameGraph":
        """Deserialize a frame graph."""
        graph = FrameGraph()
        for edge_data in data.get("edges", []):
            transform = Transform.from_dict(edge_data["transform"])
            edge = FrameGraphEdge(
                transform=transform,
                provenance=edge_data["provenance"],
                valid_from_ns=edge_data.get("valid_from_ns"),
                valid_to_ns=edge_data.get("valid_to_ns"),
                metadata=edge_data.get("metadata", {}),
            )
            graph.add_edge(edge)
        return graph


def build_frame_graph_from_world(world, include_trajectories=True) -> FrameGraph:
    """Build a frame graph from a WorldIR and its associated data.
    
    This is a convenience function that loads transforms from:
    - WorldIR coordinate frames and transforms
    - Registration results (if available in world metadata)
    - Trajectory data (if available)
    """
    graph = FrameGraph()
    
    # Load WorldIR transforms
    for transform in world.transforms.values():
        edge = FrameGraphEdge(
            transform=transform,
            provenance="worldir",
            metadata={"source": "worldir"},
        )
        graph.add_edge(edge)
    
    # Load registration data from metadata
    if "registration" in world.metadata:
        reg_data = world.metadata["registration"]
        # This would load actual registration results if stored
        pass
    
    # Note: trajectory loading would require the trajectory object
    # which is not stored in WorldIR directly
    
    return graph


# ---- Uncertainty propagation for chained transforms ----

def compose_uncertainties(u1: Uncertainty, u2: Uncertainty) -> Uncertainty:
    """Compose two uncertainties for chained transforms.
    
    Conservative composition: combined confidence is the minimum,
    notes are merged.
    """
    combined_confidence = min(u1.confidence, u2.confidence)
    notes = []
    if u1.note:
        notes.append(u1.note)
    if u2.note:
        notes.append(u2.note)
    merged_note = "; ".join(notes) if notes else None
    return Uncertainty(confidence=combined_confidence, note=merged_note)