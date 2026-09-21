"""GeoJSON evidence -> CityFeature bridge (P14-01: the `--geojson` half
of `reality city-compile`, previously import-only).

Consumes the `gis` records produced by evidence/city_import.py
(kind=OTHER EvidenceAssets, sensor_metadata={"gis": {...}}) and turns
each Polygon feature into a CityFeature. Reuses
engine.compiler.osm_features.classify_osm_tags for the tag->kind
table -- GeoJSON `properties` and OSM `tags` are both flat string-keyed
dicts checked against the same documented rules, so this is
translation, not a second classification scheme.

Honesty rules (mirroring osm_features.py):
  - Only Polygon geometry is compiled -- a footprint needs a closed
    ring. Point/LineString/MultiPolygon/etc are SKIPPED and recorded
    (unsupported_geometry), never approximated into a fake footprint.
  - Properties that don't match any classify_osm_tags rule are skipped
    and recorded (unmapped_kind), never guessed.
  - Height comes only from the feature's own properties (`height`,
    else `building:levels` / `levels` * the documented per-level
    height, mirroring osm_features.py); untagged features stay flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from engine.compiler.city_compiler import CityFeature
from engine.compiler.osm_features import _METERS_PER_LEVEL, classify_osm_tags


@dataclass
class GeoJsonCompileReport:
    """Measured facts about one GeoJSON-records -> CityFeature compile pass."""

    compiled: int = 0
    unmapped_kind: List[str] = field(default_factory=list)          # feature ids
    unsupported_geometry: List[str] = field(default_factory=list)   # feature ids
    not_gis_feature: int = 0

    def to_dict(self) -> dict:
        return {
            "compiled": self.compiled,
            "unmapped_kind": list(self.unmapped_kind),
            "unsupported_geometry": list(self.unsupported_geometry),
            "not_gis_feature": self.not_gis_feature,
        }


def _height_from_properties(props: dict) -> Optional[float]:
    if "height" in props:
        try:
            return float(props["height"])
        except (TypeError, ValueError):
            return None
    for key in ("building:levels", "levels"):
        if key in props:
            try:
                return float(props[key]) * _METERS_PER_LEVEL
            except (TypeError, ValueError):
                return None
    return None


def _exterior_ring(geometry: dict) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        return None
    coords = geometry.get("coordinates")
    if not coords or not isinstance(coords, list) or not coords[0]:
        return None
    ring = coords[0]
    return [(float(pt[0]), float(pt[1])) for pt in ring]


def geojson_record_to_city_feature(
    record: dict, report: GeoJsonCompileReport
) -> Optional[CityFeature]:
    """Translate one city_import GeoJSON feature record into a CityFeature.

    Returns None (and records why on `report`) when the feature isn't a
    mappable Polygon footprint -- never fabricates a feature.
    """
    gis = record.get("gis")
    if not gis:
        report.not_gis_feature += 1
        return None

    feature_id = gis.get("feature", "")
    props = gis.get("properties") or {}
    kind = classify_osm_tags(props)
    if kind is None:
        report.unmapped_kind.append(feature_id)
        return None

    footprint = _exterior_ring(gis.get("geometry"))
    if footprint is None or len(footprint) < 3:
        report.unsupported_geometry.append(feature_id)
        return None

    return CityFeature(
        kind=kind,
        name=props.get("name") or f"gis-{kind}-{feature_id}",
        footprint=footprint,
        height=_height_from_properties(props),
        evidence_note=f"GeoJSON feature {feature_id}",
    )


def geojson_records_to_city_features(
    records: Iterable[dict],
) -> Tuple[List[CityFeature], GeoJsonCompileReport]:
    """Batch form of geojson_record_to_city_feature over city_import records."""
    report = GeoJsonCompileReport()
    features: List[CityFeature] = []
    for record in records:
        feature = geojson_record_to_city_feature(record, report)
        if feature is not None:
            features.append(feature)
            report.compiled += 1
    return features, report
