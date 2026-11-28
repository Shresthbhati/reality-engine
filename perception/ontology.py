"""Reality Engine — Entity Ontology & Dynamic Schema Loader (Blueprint 17 / BUILD_SPEC B03).

Provides the versioned, hierarchical ontology loader for physical entities,
furnishings, architectural components, fixtures, parts, attributes, and open-vocabulary detector prompts.
Reads `config/ontology.yaml` as the canonical source of truth.
Unknown or unmapped entities are strictly resolved to `fallback/unknown_object`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml


FALLBACK_PATH = "fallback/unknown_object"

VALID_CAPABILITY_STATUSES = frozenset({
    "implemented_and_tested",
    "implemented_unvalidated",
    "partially_supported",
    "not_implemented",
    "not_applicable",
})


@dataclass(frozen=True)
class OntologyNode:
    path: str
    label: str
    parent: Optional[str] = None
    prompts: Tuple[str, ...] = ()
    geometry: str = "proxy_box"
    components: Tuple[str, ...] = ()
    mobility: str = "movable"
    min_pixels: int = 0
    typical_dims_m: Optional[Dict[str, Tuple[float, float]]] = None
    attach: Tuple[Dict[str, str], ...] = ()
    attributes: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    category: str = ""
    world_type: str = "unknown"
    size_m: Tuple[float, float] = (0.01, 500.0)
    small: bool = False
    hosts: Tuple[str, ...] = ()
    domain: str = ""
    capability_status: str = "not_implemented"
    required_evidence: Tuple[str, ...] = ()
    instance_identity_matters: bool = True
    can_be_part_of: Tuple[str, ...] = ()
    occlusion_conditions: Tuple[str, ...] = ()
    confusion_classes: Tuple[str, ...] = ()

    @property
    def group(self) -> str:
        """Top-level domain grouping (e.g. 'architecture', 'furniture', 'decor')."""
        return self.path.split("/")[0] if "/" in self.path else self.path.split(".")[0]

    @property
    def name(self) -> str:
        """Leaf name of the ontology node."""
        if "/" in self.path:
            return self.path.split("/")[-1]
        return self.path.split(".")[-1]

    def is_child_of(self, parent_path: str) -> bool:
        """Check if this node is an immediate or ancestor descendant of parent_path."""
        if self.path == parent_path:
            return True
        if self.path.startswith(parent_path.rstrip("/") + "/"):
            return True
        if self.path.startswith(parent_path.rstrip(".") + "."):
            return True
        return self.parent == parent_path


@dataclass
class Ontology:
    version: str
    nodes: Dict[str, OntologyNode] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict, init=False)
    alias_to_path: Dict[str, str] = field(default_factory=dict, init=False)
    _prompt_lookup: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _name_lookup: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.children = {p: [] for p in self.nodes}
        self.alias_to_path = {}
        self._prompt_lookup = {}
        self._name_lookup = {}

        for path, node in self.nodes.items():
            if node.parent and node.parent in self.children:
                self.children[node.parent].append(path)

            # Register aliases: lower-cased and spaces collapsed
            for alias in node.aliases:
                a_norm = " ".join(alias.strip().lower().split())
                if a_norm and a_norm not in self.alias_to_path:
                    self.alias_to_path[a_norm] = path

            # Automatically map dotted and slashed variants
            dotted = path.replace("/", ".").lower()
            slashed = path.replace(".", "/").lower()
            self.alias_to_path.setdefault(dotted, path)
            self.alias_to_path.setdefault(slashed, path)

            # First definition wins, like aliases and prompts: nodes that share a label (the hand-authored
            # ``furniture/seating/sofa`` and the taxonomy-derived ``furniture.sofa``) must not let a later duplicate
            # take the label over, or the richer node's min_pixels, prompts and dimensions silently stop applying.
            self._name_lookup.setdefault(node.name.lower(), path)
            self._name_lookup.setdefault(node.label.lower(), path)
            self._name_lookup[path.lower()] = path

            for prompt in node.prompts:
                p_clean = prompt.lower().strip()
                if p_clean not in self._prompt_lookup:
                    self._prompt_lookup[p_clean] = path

    def ancestors(self, path: str) -> List[str]:
        """Walk parent pointers upwards from the specified path."""
        out: List[str] = []
        curr = path
        visited: Set[str] = set()
        while curr:
            node = self.nodes.get(curr) or self.get(curr)
            if not node or not node.parent:
                break
            curr = node.parent
            if curr in visited:
                break
            visited.add(curr)
            out.append(curr)
        return out

    def get(self, path: str) -> Optional[OntologyNode]:
        """Lookup node by exact slash-separated or dotted path or alias."""
        if path in self.nodes:
            return self.nodes[path]
        norm = " ".join(path.strip().lower().split())
        if norm in self.alias_to_path and self.alias_to_path[norm] in self.nodes:
            return self.nodes[self.alias_to_path[norm]]
        dotted = path.replace("/", ".").lower()
        if dotted in self.alias_to_path and self.alias_to_path[dotted] in self.nodes:
            return self.nodes[self.alias_to_path[dotted]]
        slashed = path.replace(".", "/").lower()
        if slashed in self.alias_to_path and self.alias_to_path[slashed] in self.nodes:
            return self.nodes[self.alias_to_path[slashed]]
        return None

    def resolve_or_unknown(self, query: str) -> OntologyNode:
        """Resolve query (path, alias, label, name, or prompt) to an OntologyNode.
        
        If no matching node is found, returns the fallback node (`fallback/unknown_object`).
        """
        if not query:
            return self._fallback_node()

        q_clean = query.strip().lower()
        q_norm = " ".join(q_clean.split())

        # 1. Exact path match
        if query in self.nodes:
            return self.nodes[query]

        # 2. Alias lookup
        if q_norm in self.alias_to_path and self.alias_to_path[q_norm] in self.nodes:
            return self.nodes[self.alias_to_path[q_norm]]

        # 3. Case-insensitive name, label or path match
        if q_clean in self._name_lookup and self._name_lookup[q_clean] in self.nodes:
            return self.nodes[self._name_lookup[q_clean]]

        # 4. Prompt lookup match
        if q_clean in self._prompt_lookup and self._prompt_lookup[q_clean] in self.nodes:
            return self.nodes[self._prompt_lookup[q_clean]]

        # 5. Substring or prompt match
        for prompt, path in self._prompt_lookup.items():
            if q_clean == prompt or q_clean in prompt or prompt in q_clean:
                if path in self.nodes:
                    return self.nodes[path]

        return self._fallback_node()

    def _fallback_node(self) -> OntologyNode:
        if FALLBACK_PATH in self.nodes:
            return self.nodes[FALLBACK_PATH]
        return OntologyNode(
            path=FALLBACK_PATH,
            label="Unknown object",
            prompts=("object", "item"),
            geometry="proxy_box",
            mobility="movable",
            min_pixels=0,
        )

    def all_prompts(self) -> Dict[str, List[str]]:
        """Return dict of {path: list_of_prompts} for open-vocabulary detectors."""
        return {node.path: list(node.prompts) for node in self.nodes.values() if node.prompts}

    def flat_prompt_list(self) -> List[str]:
        """Return unique flattened list of all detector text queries."""
        seen = set()
        out = []
        for node in self.nodes.values():
            for p in node.prompts:
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    def by_group(self, group: str) -> List[OntologyNode]:
        """Filter nodes belonging to a specific top-level group."""
        return [node for node in self.nodes.values() if node.group == group]

    def validate(self) -> List[str]:
        """Validate internal consistency of the ontology graph."""
        issues: List[str] = []
        if FALLBACK_PATH not in self.nodes:
            issues.append(f"Missing mandatory fallback node: {FALLBACK_PATH}")

        # Cycle detection using DFS with visiting set
        VISITING, VISITED = 1, 2
        visit_state: Dict[str, int] = {}

        def dfs(p: str, stack: List[str]) -> bool:
            visit_state[p] = VISITING
            stack.append(p)
            node = self.nodes.get(p)
            if node and node.parent:
                parent_p = node.parent
                if parent_p in self.nodes:
                    if visit_state.get(parent_p) == VISITING:
                        cycle_path = " -> ".join(stack + [parent_p])
                        issues.append(f"Cycle detected in parent hierarchy: {cycle_path}")
                        return True
                    elif visit_state.get(parent_p) != VISITED:
                        if dfs(parent_p, stack):
                            return True
            stack.pop()
            visit_state[p] = VISITED
            return False

        for p in self.nodes:
            if p not in visit_state:
                dfs(p, [])

        for path, node in self.nodes.items():
            if node.parent and node.parent not in self.nodes and self.get(node.parent) is None:
                issues.append(f"Node {path} references non-existent parent: {node.parent}")
            if not node.prompts:
                issues.append(f"Node {path} has no detector prompts defined")
            if node.min_pixels < 0:
                issues.append(f"Node {path} has negative min_pixels: {node.min_pixels}")

            # Every component path exists
            for comp in node.components:
                if comp not in self.nodes and self.get(comp) is None:
                    issues.append(f"Node {path} references non-existent component: {comp}")

            # Every attach path exists
            for att in node.attach:
                for k, target_path in att.items():
                    if target_path not in self.nodes and self.get(target_path) is None:
                        issues.append(f"Node {path} attach {k} references non-existent target: {target_path}")

            # Every attribute list contains 'unknown'
            for attr_name, allowed_vals in node.attributes.items():
                if "unknown" not in allowed_vals:
                    issues.append(f"Node {path} attribute '{attr_name}' missing 'unknown' in allowed values: {allowed_vals}")

        return issues


_GLOBAL_ONTOLOGY: Optional[Ontology] = None


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_ontology(
    config_path: Optional[str | Path] = None,
    reload: bool = False,
) -> Ontology:
    """Load and parse the ontology YAML file. Caches the result globally."""
    global _GLOBAL_ONTOLOGY
    if _GLOBAL_ONTOLOGY is not None and not reload and config_path is None:
        return _GLOBAL_ONTOLOGY

    path = Path(config_path) if config_path else get_repo_root() / "config" / "ontology.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Ontology configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    version = str(data.get("version", "1.0.0"))
    raw_classes = data.get("classes", [])

    nodes: Dict[str, OntologyNode] = {}
    for item in raw_classes:
        p = item.get("path")
        if not p:
            continue

        raw_dims = item.get("typical_dims_m")
        dims: Optional[Dict[str, Tuple[float, float]]] = None
        if isinstance(raw_dims, dict):
            dims = {}
            for dim_k, dim_v in raw_dims.items():
                if isinstance(dim_v, (list, tuple)) and len(dim_v) == 2:
                    dims[dim_k] = (float(dim_v[0]), float(dim_v[1]))

        raw_size = item.get("size_m")
        size_m: Tuple[float, float] = (0.01, 500.0)
        if isinstance(raw_size, (list, tuple)) and len(raw_size) == 2:
            size_m = (float(raw_size[0]), float(raw_size[1]))

        raw_attach = item.get("attach") or []
        attach_list: List[Dict[str, str]] = []
        if isinstance(raw_attach, list):
            for a in raw_attach:
                if isinstance(a, dict):
                    attach_list.append({str(k): str(v) for k, v in a.items()})

        raw_attrs = item.get("attributes") or {}
        parsed_attrs: Dict[str, Tuple[str, ...]] = {}
        if isinstance(raw_attrs, dict):
            for ak, av in raw_attrs.items():
                if isinstance(av, (list, tuple)):
                    parsed_attrs[str(ak)] = tuple(str(x) for x in av)

        aliases = tuple(str(a) for a in item.get("aliases", []))

        cap_status = str(item.get("capability_status", "not_implemented"))
        if cap_status not in VALID_CAPABILITY_STATUSES:
            cap_status = "not_implemented"

        node = OntologyNode(
            path=p,
            label=str(item.get("label", p)),
            parent=item.get("parent"),
            prompts=tuple(item.get("prompts", [])),
            geometry=str(item.get("geometry", "proxy_box")),
            components=tuple(item.get("components", [])),
            mobility=str(item.get("mobility", "movable")),
            min_pixels=int(item.get("min_pixels", 0)),
            typical_dims_m=dims,
            attach=tuple(attach_list),
            attributes=parsed_attrs,
            aliases=aliases,
            category=str(item.get("category", "")),
            world_type=str(item.get("world_type", "unknown")),
            size_m=size_m,
            small=bool(item.get("small", False)),
            hosts=tuple(item.get("hosts", [])),
            domain=str(item.get("domain", "")),
            capability_status=cap_status,
            required_evidence=tuple(item.get("required_evidence", [])),
            instance_identity_matters=bool(item.get("instance_identity_matters", True)),
            can_be_part_of=tuple(item.get("can_be_part_of", [])),
            occlusion_conditions=tuple(item.get("occlusion_conditions", [])),
            confusion_classes=tuple(item.get("confusion_classes", [])),
        )
        nodes[p] = node

    onto = Ontology(version=version, nodes=nodes)
    if config_path is None:
        _GLOBAL_ONTOLOGY = onto
    return onto


def get_default_ontology() -> Ontology:
    """Convenience accessor for the global default ontology."""
    return load_ontology()


def resolve_entity_class(query: str) -> OntologyNode:
    """Resolve an entity classification query using the global default ontology."""
    return get_default_ontology().resolve_or_unknown(query)
