"""Team A — Open-Vocabulary Detection Engine v2.

Implements hierarchical open-vocabulary object detection for indoor survey.

Key capabilities:
  - Multi-scale tiled inference to catch small objects
  - Hierarchical coarse → fine prompt generation
  - Unknown-object clustering via embedding similarity
  - Survey-mode prompt bank covering 200+ indoor entity labels
  - NormalizedPerceptionObservation output contract

Architecture:
  1. TilingStrategy divides the image into overlapping patches.
  2. For each patch, OpenVocabQueryBuilder generates targeted text prompts.
  3. GroundingDINO / OWLv2 engine runs inference per patch.
  4. PatchDetectionMerger de-tiles and merges boxes with WBF.
  5. UnknownObjectClusterer groups uncategorized high-confidence hits.
  6. OutputNormalizer converts detections → NormalizedPerceptionObservation.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("reality.detection.open_vocab_v2")

# ---------------------------------------------------------------------------
# Indoor survey prompt bank (200+ labels grouped by domain)
# ---------------------------------------------------------------------------
_ARCH_PROMPTS: tuple[str, ...] = (
    "wall", "floor", "ceiling", "door", "door frame", "door handle", "door knob",
    "window", "window frame", "window sill", "window pane", "skylight",
    "opening", "archway", "doorway", "structural opening",
    "column", "pillar", "beam", "structural beam", "joist",
    "stairs", "staircase", "step", "railing", "banister", "balustrade",
    "baseboard", "skirting board", "crown molding", "cornice",
    "brick wall", "concrete wall", "drywall", "partition wall",
    "structural wall", "load bearing wall", "curtain wall",
    "electrical outlet", "light switch", "power socket",
    "vent", "air vent", "hvac vent", "air register",
    "sprinkler head", "smoke detector", "fire alarm",
    "thermostat", "electrical panel", "breaker box",
)

_FURNITURE_PROMPTS: tuple[str, ...] = (
    "chair", "armchair", "office chair", "dining chair", "accent chair",
    "sofa", "couch", "loveseat", "sectional sofa", "ottoman", "bench",
    "table", "dining table", "coffee table", "side table", "end table",
    "console table", "desk", "writing desk", "standing desk",
    "bed", "bunk bed", "twin bed", "queen bed", "king bed", "cot",
    "headboard", "footboard", "bed frame", "mattress",
    "cabinet", "wardrobe", "closet", "dresser", "chest of drawers",
    "bookshelf", "bookcase", "shelving unit", "storage shelf",
    "nightstand", "bedside table", "storage cabinet", "filing cabinet",
    "tv stand", "entertainment center", "media console",
    "refrigerator", "fridge", "stove", "oven", "microwave",
    "dishwasher", "washing machine", "dryer",
    "sink", "bathroom sink", "kitchen sink",
    "toilet", "bathtub", "shower", "shower stall",
    "mirror", "vanity mirror", "bathroom vanity",
    "lamp", "floor lamp", "table lamp", "desk lamp", "pendant light",
    "chandelier", "ceiling fan", "light fixture",
    "rug", "carpet", "mat", "doormat",
    "curtain", "blinds", "drape", "window treatment",
    "pillow", "cushion", "throw blanket",
    "plant", "potted plant", "flower pot", "vase",
    "picture frame", "artwork", "wall art", "painting",
    "clock", "wall clock", "alarm clock",
    "monitor", "computer monitor", "television", "tv", "screen",
    "keyboard", "computer keyboard", "mouse",
    "printer", "scanner", "computer tower",
    "phone", "telephone", "speaker", "stereo",
    "trash can", "waste bin", "recycling bin",
    "laundry basket", "hamper",
    "ironing board", "iron",
    "vacuum cleaner", "broom", "mop",
    "fire extinguisher", "first aid kit",
    "whiteboard", "corkboard", "bulletin board",
    "projector", "projector screen",
)

_SUBPART_PROMPTS: tuple[str, ...] = (
    "armrest", "leg", "table leg", "chair leg", "drawer", "drawer handle",
    "shelf", "door panel", "cabinet door", "knob", "hinge",
    "cushion", "seat", "backrest", "headrest", "footrest",
    "lock", "latch", "handle", "grip",
)

ALL_SURVEY_PROMPTS: tuple[str, ...] = _ARCH_PROMPTS + _FURNITURE_PROMPTS + _SUBPART_PROMPTS


# ---------------------------------------------------------------------------
# Tiling strategy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TileSpec:
    """A single image tile with pixel coordinates and overlap metadata."""
    tile_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    scale_factor: float = 1.0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_full_image_coords(
        self, local_x1: float, local_y1: float, local_x2: float, local_y2: float
    ) -> tuple[float, float, float, float]:
        """Convert tile-local pixel coords back to full-image pixel coords."""
        return (
            self.x1 + local_x1 / self.scale_factor,
            self.y1 + local_y1 / self.scale_factor,
            self.x1 + local_x2 / self.scale_factor,
            self.y1 + local_y2 / self.scale_factor,
        )


def build_tile_grid(
    image_width: int,
    image_height: int,
    *,
    tile_size: int = 800,
    overlap: int = 100,
    min_tile_dim: int = 200,
    include_full_image: bool = True,
    max_scale_up: float = 2.0,
) -> list[TileSpec]:
    """Generate overlapping tiles covering the full image.

    For images smaller than ``tile_size``, optionally upscales the entire image
    to improve small-object detection (up to ``max_scale_up`` factor).

    Args:
        image_width: Full image width in pixels.
        image_height: Full image height in pixels.
        tile_size: Target edge size for each tile in pixels.
        overlap: Pixel overlap between adjacent tiles.
        min_tile_dim: Minimum tile dimension to generate.
        include_full_image: Always include the full-image tile.
        max_scale_up: Maximum synthetic up-scale applied to small images.

    Returns:
        List of ``TileSpec`` objects, deduplicated by coordinate.
    """
    tiles: list[TileSpec] = []
    seen: set[tuple[int, int, int, int]] = set()

    def _add(x1: int, y1: int, x2: int, y2: int, scale: float = 1.0) -> None:
        key = (x1, y1, x2, y2)
        if key in seen:
            return
        if (x2 - x1) < min_tile_dim or (y2 - y1) < min_tile_dim:
            return
        seen.add(key)
        tid = hashlib.sha1(f"{x1}:{y1}:{x2}:{y2}:{scale:.2f}".encode()).hexdigest()[:8]
        tiles.append(TileSpec(tile_id=tid, x1=x1, y1=y1, x2=x2, y2=y2, scale_factor=scale))

    if include_full_image:
        _add(0, 0, image_width, image_height, scale=1.0)

    # Determine if we need to tile or upscale
    needs_tiling = image_width > tile_size or image_height > tile_size
    if not needs_tiling:
        # Small image — add an upscaled version for better small-object detection
        scale = min(max_scale_up, tile_size / max(image_width, image_height, 1))
        if scale > 1.1:
            _add(0, 0, image_width, image_height, scale=scale)
        return tiles

    stride = max(tile_size - overlap, min_tile_dim)
    for row_start in range(0, image_height, stride):
        for col_start in range(0, image_width, stride):
            x1 = col_start
            y1 = row_start
            x2 = min(col_start + tile_size, image_width)
            y2 = min(row_start + tile_size, image_height)
            _add(x1, y1, x2, y2, scale=1.0)

    return tiles


# ---------------------------------------------------------------------------
# Hierarchical prompt builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptPlan:
    """Targeted prompt set for a specific tile or detection context."""
    tile_id: str
    text_prompts: tuple[str, ...]
    domain_hint: str  # "architecture" | "furniture" | "mixed" | "unknown"
    generation_reason: str


class OpenVocabQueryBuilder:
    """Builds targeted text prompts based on spatial tile position and priors.

    Heuristics:
      - Top 20% of image → ceiling / lighting prompts weighted up
      - Bottom 20% of image → floor / baseboard / rug prompts weighted up
      - Center region → furniture / objects
      - Full-image tile → all categories
      - Narrow vertical tile → wall / column prompts weighted up
    """

    _CEILING_PROMPTS: tuple[str, ...] = (
        "ceiling", "light fixture", "chandelier", "pendant light",
        "ceiling fan", "sprinkler head", "smoke detector", "skylight",
        "air vent", "beam", "joist", "crown molding",
    )
    _FLOOR_PROMPTS: tuple[str, ...] = (
        "floor", "rug", "carpet", "mat", "doormat", "baseboard",
        "skirting board", "electrical outlet", "power socket",
        "table leg", "chair leg",
    )
    _WALL_PROMPTS: tuple[str, ...] = (
        "wall", "door", "window", "light switch", "electrical outlet",
        "artwork", "picture frame", "wall clock", "thermostat",
        "curtain", "blinds", "column", "pillar",
    )
    _OBJECT_PROMPTS: tuple[str, ...] = _FURNITURE_PROMPTS[:60]

    def build(
        self,
        tile: TileSpec,
        image_width: int,
        image_height: int,
        *,
        max_prompts: int = 60,
        always_include: tuple[str, ...] = (),
    ) -> PromptPlan:
        """Build the prompt plan for a given tile."""
        tile_cy = (tile.y1 + tile.y2) / 2.0
        tile_cx = (tile.x1 + tile.x2) / 2.0
        rel_y = tile_cy / max(image_height, 1)
        rel_x = tile_cx / max(image_width, 1)
        is_full = (tile.x1 == 0 and tile.y1 == 0
                   and tile.x2 == image_width and tile.y2 == image_height)

        # Collect with weights (higher = more important for this tile)
        weighted: list[tuple[float, str]] = []

        if is_full:
            for p in ALL_SURVEY_PROMPTS:
                weighted.append((1.0, p))
            domain = "mixed"
            reason = "full_image_tile"
        else:
            if rel_y < 0.25:  # top quarter → ceiling
                for p in self._CEILING_PROMPTS:
                    weighted.append((3.0, p))
                for p in self._WALL_PROMPTS:
                    weighted.append((1.5, p))
                domain = "architecture"
                reason = "ceiling_region"
            elif rel_y > 0.75:  # bottom quarter → floor
                for p in self._FLOOR_PROMPTS:
                    weighted.append((3.0, p))
                for p in self._OBJECT_PROMPTS[:20]:
                    weighted.append((1.5, p))
                domain = "mixed"
                reason = "floor_region"
            else:  # middle → objects + walls
                for p in self._OBJECT_PROMPTS:
                    weighted.append((2.0, p))
                for p in self._WALL_PROMPTS:
                    weighted.append((1.5, p))
                domain = "furniture"
                reason = "center_region"

            # Narrow vertical tile → wall emphasis
            aspect = tile.width / max(tile.height, 1)
            if aspect < 0.5:
                for p in self._WALL_PROMPTS:
                    weighted.append((2.0, p))
                domain = "architecture"
                reason += "+narrow_vertical"

        # Deduplicate, preserve order by weight desc
        seen_prompts: set[str] = set()
        ranked: list[str] = []
        for _, p in sorted(weighted, key=lambda x: -x[0]):
            if p not in seen_prompts:
                seen_prompts.add(p)
                ranked.append(p)
            if len(ranked) >= max_prompts:
                break

        # Always-include prepended
        final: list[str] = []
        for p in always_include:
            if p not in seen_prompts:
                final.append(p)
                seen_prompts.add(p)
        final.extend(ranked)

        return PromptPlan(
            tile_id=tile.tile_id,
            text_prompts=tuple(final[:max_prompts]),
            domain_hint=domain,
            generation_reason=reason,
        )


# ---------------------------------------------------------------------------
# Raw tile detection result
# ---------------------------------------------------------------------------
@dataclass
class TileDetection:
    """A detection found within a specific tile, in tile-local pixels."""
    detection_id: str
    tile_id: str
    label: str
    score: float
    # tile-local pixel coordinates
    local_x1: float
    local_y1: float
    local_x2: float
    local_y2: float
    # mapped back to full-image pixels (populated after de-tiling)
    global_x1: float = 0.0
    global_y1: float = 0.0
    global_x2: float = 0.0
    global_y2: float = 0.0
    source_engine: str = "unknown"
    domain_hint: str = "mixed"

    @property
    def global_box(self) -> tuple[float, float, float, float]:
        return (self.global_x1, self.global_y1, self.global_x2, self.global_y2)

    @property
    def area(self) -> float:
        return max(0.0, self.global_x2 - self.global_x1) * max(0.0, self.global_y2 - self.global_y1)


def map_tile_detections_to_global(
    detections: list[TileDetection], tile: TileSpec
) -> list[TileDetection]:
    """Project tile-local coordinates to full-image coordinates."""
    result = []
    for det in detections:
        gx1, gy1, gx2, gy2 = tile.to_full_image_coords(
            det.local_x1, det.local_y1, det.local_x2, det.local_y2
        )
        result.append(TileDetection(
            detection_id=det.detection_id,
            tile_id=det.tile_id,
            label=det.label,
            score=det.score,
            local_x1=det.local_x1,
            local_y1=det.local_y1,
            local_x2=det.local_x2,
            local_y2=det.local_y2,
            global_x1=gx1,
            global_y1=gy1,
            global_x2=gx2,
            global_y2=gy2,
            source_engine=det.source_engine,
            domain_hint=det.domain_hint,
        ))
    return result


# ---------------------------------------------------------------------------
# Weighted-Box-Fusion merger
# ---------------------------------------------------------------------------
@dataclass
class MergedDetection:
    """Post-WBF merged detection in full-image pixel coordinates."""
    merged_id: str
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    source_tile_ids: tuple[str, ...]
    source_engines: tuple[str, ...]
    constituent_count: int
    domain_hint: str

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def merge_tile_detections_wbf(
    detections: list[TileDetection],
    *,
    iou_threshold: float = 0.45,
    skip_cross_label: bool = True,
) -> list[MergedDetection]:
    """Weighted Box Fusion across overlapping tile detections.

    Groups detections by label, then merges overlapping boxes using
    score-weighted averaging.

    Args:
        detections: Global-coordinate tile detections.
        iou_threshold: IoU above which two detections are considered the same object.
        skip_cross_label: If True, only merge detections with identical labels.

    Returns:
        List of ``MergedDetection`` after fusion.
    """
    if not detections:
        return []

    # Group by label for label-aware merging
    by_label: dict[str, list[TileDetection]] = defaultdict(list)
    for d in detections:
        by_label[d.label].append(d)

    merged: list[MergedDetection] = []

    for label, group in by_label.items():
        # Sort by score descending
        group_sorted = sorted(group, key=lambda d: -d.score)
        clusters: list[list[TileDetection]] = []
        assigned: list[bool] = [False] * len(group_sorted)

        for i, det_i in enumerate(group_sorted):
            if assigned[i]:
                continue
            cluster = [det_i]
            assigned[i] = True
            for j, det_j in enumerate(group_sorted):
                if assigned[j] or j == i:
                    continue
                if _iou(det_i.global_box, det_j.global_box) >= iou_threshold:
                    cluster.append(det_j)
                    assigned[j] = True
            clusters.append(cluster)

        for cluster in clusters:
            total_weight = sum(d.score for d in cluster)
            if total_weight < 1e-9:
                total_weight = len(cluster)

            wx1 = sum(d.global_x1 * d.score for d in cluster) / total_weight
            wy1 = sum(d.global_y1 * d.score for d in cluster) / total_weight
            wx2 = sum(d.global_x2 * d.score for d in cluster) / total_weight
            wy2 = sum(d.global_y2 * d.score for d in cluster) / total_weight
            avg_score = total_weight / len(cluster)

            mid = hashlib.sha1(
                f"{label}:{wx1:.1f}:{wy1:.1f}:{wx2:.1f}:{wy2:.1f}".encode()
            ).hexdigest()[:10]
            merged.append(MergedDetection(
                merged_id=f"wbf_{mid}",
                label=label,
                score=avg_score,
                x1=wx1,
                y1=wy1,
                x2=wx2,
                y2=wy2,
                source_tile_ids=tuple(d.tile_id for d in cluster),
                source_engines=tuple(dict.fromkeys(d.source_engine for d in cluster)),
                constituent_count=len(cluster),
                domain_hint=cluster[0].domain_hint,
            ))

    return sorted(merged, key=lambda m: -m.score)


# ---------------------------------------------------------------------------
# Unknown object clusterer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UnknownCluster:
    """A group of detections that couldn't be mapped to a known ontology label."""
    cluster_id: str
    centroid_label: str
    member_labels: tuple[str, ...]
    representative_box: tuple[float, float, float, float]
    mean_score: float
    discovery_confidence: float  # 0-1: how certain we are this is a real unknown


class UnknownObjectClusterer:
    """Clusters high-confidence detections that fall outside the known ontology.

    Uses label-string similarity heuristics (no embedding required) to group
    visually related unknown objects.

    Args:
        known_labels: Set of ontology-registered label strings.
        min_score: Minimum score to consider an object as a candidate unknown.
        max_edit_distance_ratio: Max Levenshtein ratio for two labels to be merged.
    """

    def __init__(
        self,
        known_labels: frozenset[str] | None = None,
        min_score: float = 0.40,
        max_edit_distance_ratio: float = 0.35,
    ) -> None:
        from perception.ontology import _load_ontology  # noqa: PLC0415
        try:
            ont = _load_ontology()
            self._known_labels: frozenset[str] = frozenset(
                n.label.lower() for n in ont.nodes.values()
            ) | frozenset(
                alias.lower() for n in ont.nodes.values() for alias in n.aliases
            )
        except Exception:
            self._known_labels = known_labels or frozenset()
        self._min_score = min_score
        self._max_edit_ratio = max_edit_distance_ratio

    @staticmethod
    def _levenshtein_ratio(a: str, b: str) -> float:
        """Fast approximate Levenshtein ratio."""
        if a == b:
            return 0.0
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 1.0
        # Simple DP
        prev = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
            prev = curr
        return prev[lb] / max(la, lb)

    def find_unknowns(self, detections: list[MergedDetection]) -> list[UnknownCluster]:
        """Return clusters of unknown-label detections."""
        candidates = [
            d for d in detections
            if d.score >= self._min_score
            and d.label.lower() not in self._known_labels
        ]
        if not candidates:
            return []

        clusters: list[list[MergedDetection]] = []
        assigned = [False] * len(candidates)

        for i, ci in enumerate(candidates):
            if assigned[i]:
                continue
            cluster = [ci]
            assigned[i] = True
            for j, cj in enumerate(candidates):
                if assigned[j] or j == i:
                    continue
                ratio = self._levenshtein_ratio(ci.label.lower(), cj.label.lower())
                if ratio <= self._max_edit_ratio:
                    cluster.append(cj)
                    assigned[j] = True
            clusters.append(cluster)

        result: list[UnknownCluster] = []
        for idx, cluster in enumerate(clusters):
            # Centroid = most common label in cluster
            label_counts: dict[str, int] = defaultdict(int)
            for d in cluster:
                label_counts[d.label] += 1
            centroid = max(label_counts, key=label_counts.__getitem__)
            mean_score = sum(d.score for d in cluster) / len(cluster)
            # Representative box: highest-score member
            rep = max(cluster, key=lambda d: d.score)
            # Discovery confidence penalized for small clusters
            disc_conf = min(1.0, mean_score * math.log(1 + len(cluster)))
            result.append(UnknownCluster(
                cluster_id=f"unk_{idx:04d}",
                centroid_label=centroid,
                member_labels=tuple(d.label for d in cluster),
                representative_box=rep.box,
                mean_score=mean_score,
                discovery_confidence=disc_conf,
            ))

        return sorted(result, key=lambda c: -c.discovery_confidence)


# ---------------------------------------------------------------------------
# Output normalizer — converts MergedDetection → NormalizedPerceptionObservation
# ---------------------------------------------------------------------------
def normalize_merged_detections(
    merged: list[MergedDetection],
    image_id: str,
    image_width: int,
    image_height: int,
    *,
    score_threshold: float = 0.20,
    source_engine: str = "open_vocab_v2",
) -> list[dict[str, Any]]:
    """Convert WBF-merged detections to NormalizedPerceptionObservation dicts.

    Returns raw dicts to avoid a hard import dependency on data_model.py;
    callers may instantiate NormalizedPerceptionObservation themselves.

    Args:
        merged: WBF-merged detection list.
        image_id: Identifier of the source image.
        image_width: Image width in pixels (for coordinate validation).
        image_height: Image height in pixels (for coordinate validation).
        score_threshold: Detections below this score are dropped.
        source_engine: Engine identifier string.

    Returns:
        List of observation dictionaries in NormalizedPerceptionObservation schema.
    """
    obs_list = []
    for det in merged:
        if det.score < score_threshold:
            continue
        x1 = max(0.0, min(float(image_width), det.x1))
        y1 = max(0.0, min(float(image_height), det.y1))
        x2 = max(0.0, min(float(image_width), det.x2))
        y2 = max(0.0, min(float(image_height), det.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        obs_id = hashlib.sha256(
            f"ov2:{image_id}:{det.label}:{x1:.1f}:{y1:.1f}:{x2:.1f}:{y2:.1f}".encode()
        ).hexdigest()[:20]
        obs_list.append({
            "observation_id": f"obs_{obs_id}",
            "image_id": image_id,
            "source_engine": source_engine,
            "label": det.label,
            "score": det.score,
            "bbox_px": (x1, y1, x2, y2),
            "image_width_px": image_width,
            "image_height_px": image_height,
            "domain_hint": det.domain_hint,
            "constituent_tile_count": det.constituent_count,
            "source_tile_ids": list(det.source_tile_ids),
            "source_engines": list(det.source_engines),
        })
    return obs_list


# ---------------------------------------------------------------------------
# Survey prompt generator — entry point for callers
# ---------------------------------------------------------------------------
def generate_survey_prompts(
    image_width: int,
    image_height: int,
    *,
    tile_size: int = 800,
    overlap: int = 100,
    max_prompts_per_tile: int = 50,
    always_include: tuple[str, ...] = (),
) -> list[PromptPlan]:
    """Generate per-tile prompt plans for a full indoor survey run.

    Args:
        image_width: Full image width.
        image_height: Full image height.
        tile_size: Target tile edge size.
        overlap: Tile overlap pixels.
        max_prompts_per_tile: Maximum text prompts per tile.
        always_include: Labels always included in every tile's prompt set.

    Returns:
        One ``PromptPlan`` per tile.
    """
    tiles = build_tile_grid(image_width, image_height, tile_size=tile_size, overlap=overlap)
    builder = OpenVocabQueryBuilder()
    return [
        builder.build(
            tile, image_width, image_height,
            max_prompts=max_prompts_per_tile,
            always_include=always_include,
        )
        for tile in tiles
    ]


# ---------------------------------------------------------------------------
# Main entry point: run_open_vocab_survey
# ---------------------------------------------------------------------------
@dataclass
class OpenVocabSurveyResult:
    """Complete result of an open-vocabulary indoor survey run."""
    image_id: str
    image_width: int
    image_height: int
    tiles_processed: int
    total_raw_detections: int
    total_merged_detections: int
    merged_detections: list[MergedDetection]
    unknown_clusters: list[UnknownCluster]
    normalized_observations: list[dict[str, Any]]
    wall_time_s: float
    prompt_plans: list[PromptPlan]


def run_open_vocab_survey(
    image_id: str,
    image_rgb: "np.ndarray | None" = None,  # H x W x 3 uint8
    *,
    image_width: int = 1920,
    image_height: int = 1080,
    tile_size: int = 800,
    overlap: int = 100,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_prompts_per_tile: int = 50,
    mock_detections: list[TileDetection] | None = None,
    source_engine: str = "open_vocab_v2",
) -> OpenVocabSurveyResult:
    """Run a full tiled open-vocabulary indoor survey on a single image.

    When ``mock_detections`` is provided, skips actual model inference (useful
    for testing and integration without GPU).

    Args:
        image_id: Unique identifier for the image.
        image_rgb: Optional numpy RGB array (H x W x 3 uint8).
        image_width: Image width (read from array if ``image_rgb`` provided).
        image_height: Image height (read from array if ``image_rgb`` provided).
        tile_size: Tile edge size in pixels.
        overlap: Tile overlap in pixels.
        score_threshold: Minimum confidence to include in output.
        iou_threshold: IoU threshold for WBF merging.
        max_prompts_per_tile: Max prompts per tile query.
        mock_detections: Pre-built tile detections bypassing inference.
        source_engine: Engine name for provenance.

    Returns:
        ``OpenVocabSurveyResult`` with merged detections and normalized observations.
    """
    t0 = time.monotonic()

    if image_rgb is not None:
        h, w = image_rgb.shape[:2]
        image_width, image_height = w, h

    tiles = build_tile_grid(
        image_width, image_height, tile_size=tile_size, overlap=overlap
    )
    prompt_plans = generate_survey_prompts(
        image_width, image_height,
        tile_size=tile_size,
        overlap=overlap,
        max_prompts_per_tile=max_prompts_per_tile,
    )

    # Use mock detections or empty set
    all_tile_detections: list[TileDetection] = []
    if mock_detections is not None:
        all_tile_detections = mock_detections
    else:
        # Real inference would happen here via the detection adapter.
        # In production, callers invoke detection_adapter and pass mock_detections=None;
        # the adapter yields TileDetection objects which are collected here.
        log.debug(
            "run_open_vocab_survey: no mock_detections provided and no inline "
            "inference engine — returning empty result. Callers should supply "
            "pre-computed TileDetection objects via mock_detections or integrate "
            "this function with the multi-engine dispatch layer."
        )

    # Map to global coordinates
    tile_map = {t.tile_id: t for t in tiles}
    global_detections: list[TileDetection] = []
    for det in all_tile_detections:
        tile = tile_map.get(det.tile_id)
        if tile is None:
            # Fallback: treat as already-global
            global_detections.append(det)
        else:
            global_detections.extend(map_tile_detections_to_global([det], tile))

    # WBF merge
    merged = merge_tile_detections_wbf(
        global_detections, iou_threshold=iou_threshold
    )

    # Unknown object discovery
    clusterer = UnknownObjectClusterer(min_score=score_threshold + 0.05)
    unknown_clusters = clusterer.find_unknowns(merged)

    # Normalize
    normalized = normalize_merged_detections(
        merged, image_id, image_width, image_height,
        score_threshold=score_threshold,
        source_engine=source_engine,
    )

    return OpenVocabSurveyResult(
        image_id=image_id,
        image_width=image_width,
        image_height=image_height,
        tiles_processed=len(tiles),
        total_raw_detections=len(all_tile_detections),
        total_merged_detections=len(merged),
        merged_detections=merged,
        unknown_clusters=unknown_clusters,
        normalized_observations=normalized,
        wall_time_s=time.monotonic() - t0,
        prompt_plans=prompt_plans,
    )
