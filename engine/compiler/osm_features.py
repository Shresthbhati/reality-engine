"""OSM evidence -> CityFeature bridge (P14-01 remainder: "OSM ingestion
becomes 'produce CityFeatures' later, without touching this compiler").

Consumes the OSM `way` records produced by evidence/city_import.py
(kind=OTHER EvidenceAssets, sensor_metadata={"osm": {...}}) and turns
each mappable way into a CityFeature for
engine/compiler/city_compiler.py. No new schema: this is a translation,
not a parallel path.

Honesty rules:
  - Tag -> kind mapping is a documented, explicit table. An OSM way
    whose tags don't match any rule is SKIPPED and recorded
    (unmapped_kind), never guessed into an arbitrary bucket.
  - A way needs >= 3 resolved (lon, lat) vertices to form a footprint;
    fewer is recorded (too_few_points) and skipped -- unresolved node
    refs already reduced geometry upstream in city_import.py.
  - Height comes only from the way's own tags (`height`, else
    `building:levels` * a documented per-level height); untagged
    features stay flat (height=None), never invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from engine.compiler.city_compiler import CityFeature

#: OSM tag (key, value) -> CityFeature kind. Checked in order; the
#: first matching rule wins. `None` value means "any value for this
#: key". Documented, not inferred.
_TAG_RULES: List[Tuple[str, Optional[str], str]] = [
    ("building", None, "building"),
    ("highway", None, "road"),
    ("natural", "water", "water"),
    ("waterway", None, "water"),
    ("natural", "wood", "vegetation"),
    ("landuse", "forest", "vegetation"),
    ("landuse", "wood", "vegetation"),
    ("power", None, "infrastructure"),
    ("man_made", None, "infrastructure"),
    ("amenity", None, "infrastructure"),
]

#: Documented per-level height estimate (meters) when only
#: `building:levels` is tagged -- a stated assumption, not a measurement.
_METERS_PER_LEVEL = 3.0


@dataclass
class OsmCompileReport:
    """Measured facts about one OSM-records -> CityFeature compile pass."""

    compiled: int = 0
    unmapped_kind: List[str] = field(default_factory=list)   # osm_ids
    too_few_points: List[str] = field(default_factory=list)  # osm_ids
    not_osm_way: int = 0

    def to_dict(self) -> dict:
        return {
            "compiled": self.compiled,
            "unmapped_kind": list(self.unmapped_kind),
            "too_few_points": list(self.too_few_points),
            "not_osm_way": self.not_osm_way,
        }


def classify_osm_tags(tags: dict) -> Optional[str]:
    """Return the CityFeature kind for a tag set, or None if unmapped."""
    for key, value, kind in _TAG_RULES:
        if key in tags and (value is None or tags[key] == value):
            return kind
    return None


def _osm_height(tags: dict) -> Optional[float]:
    if "height" in tags:
        try:
            return float(tags["height"])
        except (TypeError, ValueError):
            return None
    if "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * _METERS_PER_LEVEL
        except (TypeError, ValueError):
            return None
    return None


def _osm_name(tags: dict, osm_id: str, kind: str) -> str:
    return tags.get("name") or f"osm-{kind}-{osm_id}"


def osm_record_to_city_feature(
    record: dict, report: OsmCompileReport
) -> Optional[CityFeature]:
    """Translate one city_import OSM `way` record into a CityFeature.

    Returns None (and records why on `report`) when the way isn't a
    mappable footprint -- never fabricates a feature to fill the gap.
    """
    osm = record.get("osm")
    if not osm or osm.get("element_type") != "way":
        report.not_osm_way += 1
        return None

    osm_id = osm.get("osm_id", "")
    tags = osm.get("tags", {})
    kind = classify_osm_tags(tags)
    if kind is None:
        report.unmapped_kind.append(osm_id)
        return None

    geometry = osm.get("geometry", [])
    if len(geometry) < 3:
        report.too_few_points.append(osm_id)
        return None

    footprint = [(float(pt[0]), float(pt[1])) for pt in geometry]
    return CityFeature(
        kind=kind,
        name=_osm_name(tags, osm_id, kind),
        footprint=footprint,
        height=_osm_height(tags),
        evidence_note=f"OSM way {osm_id}",
    )


def osm_records_to_city_features(
    records: Iterable[dict],
) -> Tuple[List[CityFeature], OsmCompileReport]:
    """Batch form of osm_record_to_city_feature over city_import records."""
    report = OsmCompileReport()
    features: List[CityFeature] = []
    for record in records:
        feature = osm_record_to_city_feature(record, report)
        if feature is not None:
            features.append(feature)
            report.compiled += 1
    return features, report
