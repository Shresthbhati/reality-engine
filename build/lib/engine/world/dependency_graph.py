"""System dependency graph and topological execution ordering.

Invariants:
  - DAG (directed acyclic graph) guaranteed; cycles rejected
  - Execution order deterministic and reproducible
  - Dependencies resolved before dependents
  - No circular dependencies allowed
"""

from __future__ import annotations

from typing import Callable, Optional, Set, List, Dict
from dataclasses import dataclass, field


@dataclass
class System:
    """A system in the dependency graph."""
    name: str
    execute: Callable[[], None]
    dependencies: Set[str] = field(default_factory=set)
    tier: int = 0  # Execution tier (lower = earlier)


class DependencyGraph:
    """Manages system dependencies and execution ordering.

    Systems are executed in topological order. Dependencies are validated
    to prevent cycles. Execution order is deterministic (consistent across runs).

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self):
        self._systems: Dict[str, System] = {}
        self._execution_order: List[str] = []
        self._dirty: bool = True

    def register_system(
        self,
        name: str,
        execute: Callable[[], None],
        depends_on: Optional[Set[str]] = None,
        tier: int = 0,
    ) -> None:
        """Register a system with dependencies.

        Args:
            name: System name (must be unique)
            execute: Callable to execute the system
            depends_on: Set of system names this depends on
            tier: Execution tier (lower = earlier)

        Raises:
            ValueError: If system already registered or dependency doesn't exist
        """
        if name in self._systems:
            raise ValueError(f"System '{name}' already registered")

        if depends_on is None:
            depends_on = set()

        # Validate dependencies exist (or will be registered)
        system = System(name=name, execute=execute, dependencies=depends_on, tier=tier)
        self._systems[name] = system
        self._dirty = True

    def update_dependencies(self, name: str, depends_on: Set[str]) -> None:
        """Update dependencies for an existing system.

        Args:
            name: System to update
            depends_on: New set of dependencies

        Raises:
            ValueError: If system doesn't exist or would create a cycle
        """
        if name not in self._systems:
            raise ValueError(f"System '{name}' not registered")

        # Check for cycles before updating
        self._check_for_cycle(name, depends_on)

        self._systems[name].dependencies = depends_on
        self._dirty = True

    def unregister_system(self, name: str) -> None:
        """Unregister a system.

        Args:
            name: System to remove

        Raises:
            ValueError: If system doesn't exist or is depended on by others
        """
        if name not in self._systems:
            raise ValueError(f"System '{name}' not registered")

        # Check if any other systems depend on this one
        for system in self._systems.values():
            if name in system.dependencies:
                raise ValueError(
                    f"Cannot unregister '{name}': system(s) depend on it"
                )

        del self._systems[name]
        self._dirty = True

    def get_execution_order(self) -> List[str]:
        """Get the topological execution order.

        Returns list of system names in execution order (dependencies before dependents).

        Raises:
            ValueError: If a cycle is detected
        """
        if not self._dirty:
            return list(self._execution_order)

        # Topological sort using Kahn's algorithm
        in_degree: Dict[str, int] = {}
        adjacency: Dict[str, List[str]] = {}

        # Initialize
        for name in self._systems:
            in_degree[name] = 0
            adjacency[name] = []

        # Build graph
        for system in self._systems.values():
            for dep in system.dependencies:
                if dep not in self._systems:
                    raise ValueError(f"Unknown dependency: '{dep}'")
                adjacency[dep].append(system.name)
                in_degree[system.name] += 1

        # Find all nodes with no incoming edges
        queue = [name for name in self._systems if in_degree[name] == 0]
        queue.sort()  # Deterministic order

        order = []
        while queue:
            # Process by tier first, then alphabetically for determinism
            queue.sort(key=lambda n: (self._systems[n].tier, n))
            current = queue.pop(0)
            order.append(current)

            # Remove edges from current
            for neighbor in sorted(adjacency[current]):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._systems):
            raise ValueError("Cycle detected in dependency graph")

        self._execution_order = order
        self._dirty = False
        return list(order)

    def execute_all(self) -> None:
        """Execute all systems in dependency order.

        Raises:
            ValueError: If a cycle is detected or execution fails
            RuntimeError: If a system raises an exception
        """
        order = self.get_execution_order()
        for name in order:
            system = self._systems[name]
            try:
                system.execute()
            except Exception as e:
                raise RuntimeError(f"System '{name}' execution failed: {e}") from e

    def execute_system(self, name: str) -> None:
        """Execute a single system (for testing).

        Does NOT execute dependencies; use execute_all() for proper ordering.

        Args:
            name: System to execute

        Raises:
            ValueError: If system doesn't exist
            RuntimeError: If system execution fails
        """
        if name not in self._systems:
            raise ValueError(f"System '{name}' not registered")

        system = self._systems[name]
        try:
            system.execute()
        except Exception as e:
            raise RuntimeError(f"System '{name}' execution failed: {e}") from e

    def get_system(self, name: str) -> Optional[System]:
        """Get a system by name."""
        return self._systems.get(name)

    def all_systems(self) -> List[str]:
        """Get all registered system names."""
        return list(self._systems.keys())

    def system_count(self) -> int:
        """Get number of registered systems."""
        return len(self._systems)

    def _check_for_cycle(self, name: str, depends_on: Set[str]) -> None:
        """Check if adding dependencies would create a cycle.

        Args:
            name: System being updated
            depends_on: New dependencies

        Raises:
            ValueError: If a cycle would be created
        """
        # DFS to detect cycle
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Get dependencies of this node
            if node == name:
                neighbors = depends_on
            else:
                neighbors = self._systems[node].dependencies if node in self._systems else set()

            for neighbor in neighbors:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        if has_cycle(name):
            raise ValueError(f"Adding dependencies to '{name}' would create a cycle")

    def clear(self) -> None:
        """Clear all systems (for testing)."""
        self._systems.clear()
        self._execution_order.clear()
        self._dirty = True
