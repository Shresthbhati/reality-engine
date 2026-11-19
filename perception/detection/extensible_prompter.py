"""Extensible Open-Vocabulary Detection Prompter & Ontology Bridge.

Bridges the canonical Reality Engine Ontology (`config/ontology.yaml` via
`perception.ontology`) to open-vocabulary perception models (Grounding DINO,
Florence-2, OWL-ViT).

Eliminates hard-coded prompt lists by dynamically deriving prompts for
macro-scale structures, meso-scale objects, and micro-scale architectural details
(sockets, switches, door handles, hinges, skirting boards, road furniture).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from perception.ontology import get_default_ontology


class DomainFilter(str, Enum):
    """Environmental domain filtering for targeted open-vocabulary prompting."""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    ALL = "all"


class ScaleFilter(str, Enum):
    """Physical scale filtering for staged coarse-to-fine perception."""

    MACRO = "macro"  # Structural elements, rooms, facades, roads (>= 1.5 m)
    MESO = "meso"    # Furniture, appliances, doors, windows, vehicles (0.2 m - 2.0 m)
    MICRO = "micro"  # Hardware, electrical sockets, switches, hinges, small decor (< 0.2 m)
    ALL = "all"


@dataclass(frozen=True)
class PromptTarget:
    """One detection target mapped directly to a canonical OntologyNode."""

    path: str
    canonical_label: str
    primary_prompt: str
    prompt_variants: tuple[str, ...]
    min_pixels: int
    is_small: bool
    group: str
    hosts: tuple[str, ...]
    typical_dims_m: dict[str, tuple[float, float]] | None = None


class ExtensiblePrompter:
    """Dynamically builds prompt sets and maps detector hits to canonical ontology nodes."""

    def __init__(self) -> None:
        self._ontology = get_default_ontology()
        self._targets: dict[str, PromptTarget] = {}
        self._prompt_to_target: dict[str, PromptTarget] = {}
        self._custom_targets: dict[str, PromptTarget] = {}
        self._build_targets()

    def _build_targets(self) -> None:
        """Indexes all nodes from the canonical ontology."""
        for path, node in self._ontology.nodes.items():
            if not node.prompts or path.startswith("fallback"):
                continue

            primary = node.prompts[0]
            variants = node.prompts[1:] if len(node.prompts) > 1 else ()

            target = PromptTarget(
                path=path,
                canonical_label=node.label,
                primary_prompt=primary,
                prompt_variants=variants,
                min_pixels=node.min_pixels,
                is_small=node.small,
                group=node.group,
                hosts=node.hosts,
                typical_dims_m=node.typical_dims_m,
            )
            self._targets[path] = target
            self._prompt_to_target[primary.lower().strip()] = target
            for v in variants:
                self._prompt_to_target[v.lower().strip()] = target

    def register_custom_target(self, target: PromptTarget) -> None:
        """Dynamically add an extensible target without modifying config files."""
        self._custom_targets[target.path] = target
        self._prompt_to_target[target.primary_prompt.lower().strip()] = target
        for v in target.prompt_variants:
            self._prompt_to_target[v.lower().strip()] = target

    def get_prompts(
        self,
        domain: DomainFilter = DomainFilter.ALL,
        scale: ScaleFilter = ScaleFilter.ALL,
        groups: Sequence[str] | None = None,
        only_small: bool = False,
    ) -> list[str]:
        """Collects deduplicated detection prompts matching the filter criteria."""
        all_targets = {**self._targets, **self._custom_targets}
        selected_prompts: set[str] = set()

        for target in all_targets.values():
            if only_small and not target.is_small:
                continue

            # Scale filter
            if scale == ScaleFilter.MICRO and not target.is_small:
                continue
            elif scale == ScaleFilter.MACRO and (target.is_small or target.group in ("furniture", "decor")):
                continue
            elif scale == ScaleFilter.MESO and target.is_small:
                continue

            # Group filter
            if groups is not None and target.group not in groups:
                continue

            # Domain filter
            outdoor_groups = {"exterior", "road", "street", "building"}
            indoor_groups = {"furniture", "room", "decor", "electrical", "plumbing", "hvac"}
            if domain == DomainFilter.INDOOR and target.group in outdoor_groups:
                continue
            if domain == DomainFilter.OUTDOOR and target.group in indoor_groups:
                continue

            selected_prompts.add(target.primary_prompt)
            for v in target.prompt_variants:
                selected_prompts.add(v)

        return sorted(selected_prompts)

    def resolve_prediction(self, raw_label: str) -> PromptTarget | None:
        """Resolves a raw open-vocabulary predicted string to its canonical target."""
        clean = raw_label.lower().strip()
        if clean in self._prompt_to_target:
            return self._prompt_to_target[clean]

        # Prioritize exact word matching or longer prompt match
        best_match = None
        best_len = 0
        for prompt, target in self._prompt_to_target.items():
            if prompt == clean:
                return target
            if prompt in clean or clean in prompt:
                if len(prompt) > best_len:
                    best_len = len(prompt)
                    best_match = target

        if best_match is not None:
            return best_match

        # Node name fallback
        node = self._ontology.resolve_or_unknown(clean)
        if node and not node.path.startswith("fallback"):
            return self._targets.get(node.path)

        return None


_GLOBAL_PROMPTER: ExtensiblePrompter | None = None


def get_extensible_prompter() -> ExtensiblePrompter:
    """Returns the singleton instance of the extensible prompter."""
    global _GLOBAL_PROMPTER
    if _GLOBAL_PROMPTER is None:
        _GLOBAL_PROMPTER = ExtensiblePrompter()
    return _GLOBAL_PROMPTER
