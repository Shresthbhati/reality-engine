"""City evidence ingestion boundary (REAL_RECONSTRUCTION_PERCEPTION_
CITY_READY increment B): OSM XML and GeoJSON become canonical
EvidenceAssets in the EXISTING evidence model — no separate city-data
schema, no parallel provenance path.

Design rules (mirroring evidence/importers.py discipline):

  - Everything is an ordinary EvidenceAsset (kind=OTHER): map/vector
    data is none of photo/video/lidar/depth, and pretending otherwise
    would corrupt the kind semantics. The asset's sensor_metadata
    carries the verbatim external facts (element ids, tags, measured
    geometry), so provenance answers "which OSM element / which GIS
    feature produced this".
  - Provenance is OBSERVED with confidence 1.0: externally recorded
    data, not engine inference. We do not upgrade, clean, or guess.
  - Deterministic: document order, canonical-JSON payloads, stable
    content-hash ids. Same input -> same package bytes.
  - Dedup rides the builder's content-hash contract (asset_id_for_
    payload): re-importing the same extract is duplicates_skipped,
    never duplicated assets.
  - Malformed input raises CorruptEvidenceError -- the same honest gate
    photos and LAS get; a broken file is never a silent empty import.
  - Unresolvable node references are recorded per way
    (unresolved_node_refs) and produce NO fabricated geometry: an
    incomplete way imports with geometry=[] plus the recorded refs.
"""

from __future__ import annotations

import json
import struct  # noqa: F401  (kept for parity with importers' header readers)
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from evidence.packages import (
    CorruptEvidenceError,
    DeterministicPackageBuilder,
    EvidenceKind,
    EvidenceSource,
)
from provenance.provenance import Provenance

OSM_SOURCE_NOTE = "OpenStreetMap (ODbL) (c) OpenStreetMap contributors"


@dataclass
class CityImportReport:
    """Measured facts about one city-import call (JSON-serializable)."""

    path: str = ""
    imported_features: int = 0          # ways (OSM) or features (GeoJSON)
    nodes_seen: int = 0                 # OSM only: node elements in the file
    duplicates_skipped: List[Tuple[str, str]] = field(default_factory=list)
    unresolved_node_refs: int = 0       # OSM only: refs with no node element
    imported_asset_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "imported_features": self.imported_features,
            "nodes_seen": self.nodes_seen,
            "duplicates_skipped": [list(p) for p in self.duplicates_skipped],
            "unresolved_node_refs": self.unresolved_node_refs,
            "imported_asset_ids": list(self.imported_asset_ids),
        }


def _canonical_payload_bytes(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _add_derived_asset(
    builder: DeterministicPackageBuilder,
    source: EvidenceSource,
    source_uri: str,
    record: dict,
    report: CityImportReport,
) -> bool:
    """Add one derived city asset; content-hash dedup, deterministic id.
    Returns True when a new asset was added, False when deduped."""
    payload = _canonical_payload_bytes(record)
    existing = builder.asset_id_for_payload(payload)
    if existing is not None:
        report.duplicates_skipped.append((record["osm"]["osm_id"], existing))
        return False
    asset_id = builder.add_payload(
        payload,
        kind=EvidenceKind.OTHER,
        source=source,
        source_uri=source_uri,
        sensor_metadata=record,
        quality={"measured": 1.0},  # verbatim external facts, nothing estimated
        provenance=Provenance.OBSERVED,
    )
    report.imported_asset_ids.append(asset_id)
    return True


def import_osm_xml(
    builder: DeterministicPackageBuilder,
    source: EvidenceSource,
    path: str,
) -> CityImportReport:
    """Import an OSM XML extract: one EvidenceAsset per <way>, plus one
    node-collection asset. Geometry is the way's node refs RESOLVED
    through the file's own <node> elements (measured, in document
    order); refs with no node element are recorded as unresolved and
    contribute no fabricated coordinates."""
    report = CityImportReport(path=str(path))
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise CorruptEvidenceError(f"{path}: malformed OSM XML: {exc}") from exc
    if root.tag != "osm":
        raise CorruptEvidenceError(
            f"{path}: not an OSM document (root element {root.tag!r})"
        )

    nodes: Dict[str, Tuple[float, float]] = {}
    ways: List[dict] = []
    for element in root:
        if element.tag == "node":
            report.nodes_seen += 1
            node_id = element.get("id", "")
            lat, lon = element.get("lat"), element.get("lon")
            if node_id and lat is not None and lon is not None:
                nodes[node_id] = (float(lon), float(lat))  # (x=lon, y=lat)
        elif element.tag == "way":
            refs = [nd.get("ref", "") for nd in element.findall("nd")]
            tags = {
                t.get("k", ""): t.get("v", "") for t in element.findall("tag")
            }
            ways.append({"osm_id": element.get("id", ""), "refs": refs, "tags": tags})

    for way in ways:
        geometry: List[List[float]] = []
        unresolved: List[str] = []
        for ref in way["refs"]:
            if ref in nodes:
                lon, lat = nodes[ref]
                geometry.append([lon, lat])
            elif ref not in unresolved:  # first occurrence order, no double-count
                unresolved.append(ref)
        report.unresolved_node_refs += len(unresolved)
        record = {
            "osm": {
                "element_type": "way",
                "osm_id": way["osm_id"],
                "tags": way["tags"],
                "geometry": geometry,
                "unresolved_node_refs": unresolved,
                "source_note": OSM_SOURCE_NOTE,
            }
        }
        if _add_derived_asset(builder, source, path, record, report):
            report.imported_features += 1

    if nodes:
        orphan_ids = [
            node_id for node_id in nodes
            if not any(node_id in w["refs"] for w in ways)
        ]
        node_record = {
            "osm": {
                "element_type": "node_collection",
                "osm_id": f"nodes-{len(nodes)}",
                "node_count": len(nodes),
                "orphan_node_ids": orphan_ids,
                "source_note": OSM_SOURCE_NOTE,
            }
        }
        _add_derived_asset(builder, source, path, node_record, report)
    return report


def import_geojson(
    builder: DeterministicPackageBuilder,
    source: EvidenceSource,
    path: str,
) -> CityImportReport:
    """Import a GeoJSON FeatureCollection (or single Feature): one
    EvidenceAsset per feature with geometry verbatim and properties
    recorded. Nothing is reprojected or simplified here -- GIS
    geometry enters the evidence layer exactly as supplied."""
    report = CityImportReport(path=str(path))
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorruptEvidenceError(f"{path}: malformed GeoJSON: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("type") not in (
        "FeatureCollection", "Feature"
    ):
        raise CorruptEvidenceError(
            f"{path}: not a GeoJSON Feature/FeatureCollection "
            f"(type={doc.get('type') if isinstance(doc, dict) else type(doc).__name__!r})"
        )

    features = (
        doc.get("features", []) if doc["type"] == "FeatureCollection" else [doc]
    )
    for index, feature in enumerate(features):
        record = {
            "gis": {
                "feature": feature.get("id") or f"feature-{index}",
                "geometry": feature.get("geometry"),
                "properties": feature.get("properties") or {},
                "source_note": "external GIS data (verbatim GeoJSON)",
            }
        }
        if _add_derived_asset(builder, source, path, record, report):
            report.imported_features += 1

    return report
