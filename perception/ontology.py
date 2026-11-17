"""Reality Engine — Entity Ontology & Dynamic Schema Loader (Blueprint 17).

Provides the versioned, hierarchical ontology loader for physical entities,
furnishings, architectural components, fixtures, and open-vocabulary detector prompts.
Reads `config/ontology.yaml` as the canonical source of truth.
Unknown or unmapped entities are strictly resolved to `fallback/unknown_object`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


FALLBACK_PATH = "fallback/unknown_object"


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
    #: attribute name -> allowed values; every list ends with "unknown" (an unseen attribute is never guessed).
    attributes: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()
    #: Registered entity-class data (formerly perception.entities.taxonomy). None on nodes that are not registered
    #: classes: the source never stated it, so there is no default.
    world_type: Optional[str] = None            # an existing world_ir EntityType value
    size_m: Optional[Tuple[float, float]] = None  # plausible largest dimension (min, max), metres
    small: bool = False                          # needs the high-detail inspection pass

    @property
    def group(self) -> str:
        """Top-level domain grouping (e.g. 'architecture', 'furniture', 'decor')."""
        return self.path.split("/")[0] if "/" in self.path else self.path

    @property
    def name(self) -> str:
        """Leaf name of the ontology node."""
        return self.path.split("/")[-1]

    def is_child_of(self, parent_path: str) -> bool:
        """Check if this node is an immediate or ancestor descendant of parent_path."""
        if self.path == parent_path:
            return True
        if self.path.startswith(parent_path.rstrip("/") + "/"):
            return True
        return self.parent == parent_path


@dataclass
class Ontology:
    version: str
    nodes: Dict[str, OntologyNode] = field(default_factory=dict)
    children: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=False)
    alias_to_path: Dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _prompt_lookup: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.children = {}
        self.alias_to_path = {}
        self._prompt_lookup = {}
        for path, node in self.nodes.items():
            if node.parent:
                self.children.setdefault(node.parent, []).append(path)
            # First node wins a name, so adding classes never re-points an existing lookup: an implicit name
            # (label/leaf/path) does not take a string an earlier node already uses as a detector prompt.
            for name in node.aliases:
                self.alias_to_path.setdefault(_norm(name), path)
            for name in (node.label, node.name, path, path.replace("/", ".")):
                if _norm(name) not in self._prompt_lookup:
                    self.alias_to_path.setdefault(_norm(name), path)
            for prompt in node.prompts:
                p_clean = prompt.lower().strip()
                if p_clean not in self._prompt_lookup:
                    self._prompt_lookup[p_clean] = path

    def ancestors(self, path: str) -> List[str]:
        """Parent chain, nearest first (cycle-safe: validate() reports cycles)."""
        out: List[str] = []
        seen = {path}
        cur = self.nodes[path].parent if path in self.nodes else None
        while cur and cur not in seen:
            out.append(cur)
            seen.add(cur)
            node = self.nodes.get(cur)
            cur = node.parent if node else None
        return out

    def get(self, path: str) -> Optional[OntologyNode]:
        """Lookup node by exact slash-separated path."""
        return self.nodes.get(path)

    def resolve_or_unknown(self, query: str) -> OntologyNode:
        """Resolve query (path, label, name, or prompt) to an OntologyNode.
        
        If no matching node is found, returns the fallback node (`fallback/unknown_object`).
        """
        if not query:
            return self._fallback_node()

        q_clean = query.strip().lower()

        # 1. Exact path match
        if query in self.nodes:
            return self.nodes[query]

        # 2. Alias, label, name, path or dotted-path match (case-insensitive, spaces collapsed)
        hit = self.alias_to_path.get(_norm(query))
        if hit is not None:
            return self.nodes[hit]

        # 3. Prompt lookup match
        if q_clean in self._prompt_lookup:
            return self.nodes[self._prompt_lookup[q_clean]]

        # 4. Substring or prompt match
        for prompt, path in self._prompt_lookup.items():
            if q_clean == prompt or q_clean in prompt or prompt in q_clean:
                return self.nodes[path]

        return self._fallback_node()

    def _fallback_node(self) -> OntologyNode:
        if FALLBACK_PATH in self.nodes:
            return self.nodes[FALLBACK_PATH]
        # Synthetic fallback if YAML lacks one
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

        for path, node in self.nodes.items():
            if node.parent and node.parent not in self.nodes:
                issues.append(f"Node {path} references non-existent parent: {node.parent}")
            if not node.prompts:
                issues.append(f"Node {path} has no detector prompts defined")
            if node.min_pixels < 0:
                issues.append(f"Node {path} has negative min_pixels: {node.min_pixels}")
            for comp in node.components:
                # a short part name is a child node; a full path may point at any node
                if comp not in self.nodes and f"{path}/{comp}" not in self.nodes:
                    issues.append(f"Node {path} lists component that does not exist: {comp}")
            for rel in node.attach:
                for kind, target in rel.items():
                    if target not in self.nodes:
                        issues.append(f"Node {path} attach {kind} targets non-existent node: {target}")
            for attr, values in node.attributes.items():
                if "unknown" not in values:
                    issues.append(f"Node {path} attribute {attr} has no 'unknown' value")
        owner: Dict[str, str] = {}
        for path, node in self.nodes.items():
            for alias in node.aliases:
                key = _norm(alias)
                if owner.setdefault(key, path) != path:
                    issues.append(f"Alias '{alias}' claimed by both {owner[key]} and {path}")
        done: set = set()
        for start in self.nodes:
            visiting: set = set()
            cur: Optional[str] = start
            while cur and cur in self.nodes and cur not in done:
                if cur in visiting:
                    issues.append(f"Parent cycle through {cur}")
                    break
                visiting.add(cur)
                cur = self.nodes[cur].parent
            done |= visiting
        return issues


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


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

        raw_attach = item.get("attach") or []
        attach_list: List[Dict[str, str]] = []
        if isinstance(raw_attach, list):
            for a in raw_attach:
                if isinstance(a, dict):
                    attach_list.append({str(k): str(v) for k, v in a.items()})

        raw_attrs = item.get("attributes") or {}
        raw_size = item.get("size_m")
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
            attributes={str(k): tuple(str(x) for x in v) for k, v in raw_attrs.items()},
            aliases=tuple(str(a) for a in item.get("aliases") or ()),
            world_type=item.get("world_type"),
            size_m=(float(raw_size[0]), float(raw_size[1])) if raw_size else None,
            small=bool(item.get("small", False)),
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
