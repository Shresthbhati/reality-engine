"""Asset extraction slice (T-2.6).

Slices a single furniture (or other architectural) element from a reconstructed room WorldIR
into an independent, reusable asset with its own ID, URL, canonical dimensions,
materials, geometry, and provenance. The original WorldIR is never mutated.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid

from exporters.gltf.exporter import export_to_glb
from world_ir.schema_v1 import ConfidenceVector, Entity, EntityType, Geometry, Material, Surface, Vector3
from world_ir.world_v1 import WorldIR


@dataclass
class ExtractedAsset:
    """Independent representation of an asset extracted from a world entity."""

    id: str
    name: str
    category: str
    source_world_id: str
    source_entity_id: str
    dimensions: dict[str, Any]
    provenance: dict[str, Any]
    geometry_ids: list[str] = field(default_factory=list)
    material_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "source_world_id": self.source_world_id,
            "source_entity_id": self.source_entity_id,
            "dimensions": self.dimensions,
            "provenance": self.provenance,
            "geometry_ids": self.geometry_ids,
            "material_ids": self.material_ids,
            "metadata": self.metadata,
        }


def _extract_dim_scalar(dim_obj: Any) -> Optional[float]:
    if isinstance(dim_obj, (int, float)):
        return float(dim_obj)
    if isinstance(dim_obj, dict):
        val = dim_obj.get("value")
        if val is not None:
            return float(val)
    return None


def extract_entity_asset(
    world: WorldIR,
    entity_id: str,
    name: Optional[str] = None,
    category: Optional[str] = None,
    artifact_store: Any = None,
) -> tuple[ExtractedAsset, WorldIR, bytes]:
    """Extract a single entity from `world` into an independent asset.

    Preserves geometry, materials, dimensions, and provenance back to the source world.
    The source `world` is never mutated.

    Returns:
        tuple[ExtractedAsset, WorldIR, bytes]:
            - ExtractedAsset record metadata
            - Isolated single-entity WorldIR centered at local origin
            - Binary GLB byte container of the asset
    """
    if entity_id not in world.entities:
        raise KeyError(f"Entity '{entity_id}' not found in world '{world.id}'")

    source_entity = world.entities[entity_id]

    asset_id = f"ast-{uuid.uuid4().hex[:12]}"
    asset_name = name or source_entity.name or f"Asset {entity_id}"
    cat = (
        category
        or (source_entity.type.value if hasattr(source_entity.type, "value") else str(source_entity.type))
    )

    # 1. Canonical dimensions
    custom_dims = (source_entity.custom_properties or {}).get("dimensions", {})
    w_dim = _extract_dim_scalar(custom_dims.get("width_m") or custom_dims.get("width"))
    d_dim = _extract_dim_scalar(custom_dims.get("depth_m") or custom_dims.get("depth"))
    h_dim = _extract_dim_scalar(custom_dims.get("height_m") or custom_dims.get("height"))

    # Compute bounds from referenced geometries
    min_x, min_y, min_z = float("inf"), float("inf"), float("inf")
    max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")
    has_bounds = False

    for gid in getattr(source_entity, "geometry_ids", []):
        geom = world.geometries.get(gid)
        if geom and getattr(geom, "bounds_min", None) and getattr(geom, "bounds_max", None):
            bmin, bmax = geom.bounds_min, geom.bounds_max
            min_x = min(min_x, bmin.x if hasattr(bmin, "x") else bmin[0])
            min_y = min(min_y, bmin.y if hasattr(bmin, "y") else bmin[1])
            min_z = min(min_z, bmin.z if hasattr(bmin, "z") else bmin[2])
            max_x = max(max_x, bmax.x if hasattr(bmax, "x") else bmax[0])
            max_y = max(max_y, bmax.y if hasattr(bmax, "y") else bmax[1])
            max_z = max(max_z, bmax.z if hasattr(bmax, "z") else bmax[2])
            has_bounds = True

    if has_bounds:
        geom_w = round(abs(max_x - min_x), 4)
        geom_d = round(abs(max_y - min_y), 4)
        geom_h = round(abs(max_z - min_z), 4)
        bbox = {"min": [round(min_x, 4), round(min_y, 4), round(min_z, 4)], "max": [round(max_x, 4), round(max_y, 4), round(max_z, 4)]}
    else:
        geom_w, geom_d, geom_h = 1.0, 1.0, 1.0
        bbox = None

    dimensions = {
        "width_m": round(w_dim if w_dim is not None else geom_w, 4),
        "depth_m": round(d_dim if d_dim is not None else geom_d, 4),
        "height_m": round(h_dim if h_dim is not None else geom_h, 4),
    }
    if bbox:
        dimensions["bounding_box"] = bbox

    # 2. Provenance link
    prov: dict[str, Any] = {
        "source_world_id": world.id,
        "source_world_version": world.version,
        "source_entity_id": source_entity.id,
        "source_entity_name": source_entity.name,
        "source_entity_type": source_entity.type.value if hasattr(source_entity.type, "value") else str(source_entity.type),
        "entity_provenance": source_entity.provenance.value if hasattr(source_entity.provenance, "value") else str(source_entity.provenance),
        "confidence": getattr(source_entity, "confidence", 1.0),
    }
    if getattr(source_entity, "confidence_vector", None):
        cv = source_entity.confidence_vector
        prov["confidence_vector"] = cv.to_dict() if hasattr(cv, "to_dict") else {}

    # 3. Clone and center entity
    asset_ent = copy.deepcopy(source_entity)
    asset_ent.id = f"asset-ent-{asset_id}"
    asset_ent.name = asset_name

    orig_transform = copy.deepcopy(asset_ent.transform) if getattr(asset_ent, "transform", None) else None
    if not hasattr(asset_ent, "custom_properties") or asset_ent.custom_properties is None:
        asset_ent.custom_properties = {}

    if orig_transform:
        asset_ent.custom_properties["original_world_transform"] = orig_transform
        asset_ent.transform = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": orig_transform.get("rotation", [0.0, 0.0, 0.0, 1.0]),
            "scale": orig_transform.get("scale", [1.0, 1.0, 1.0]),
        }
    else:
        asset_ent.transform = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }

    # 4. Standalone Asset WorldIR
    asset_geometries = {
        gid: copy.deepcopy(world.geometries[gid])
        for gid in getattr(source_entity, "geometry_ids", [])
        if gid in world.geometries
    }
    asset_materials = {
        mid: copy.deepcopy(world.materials[mid])
        for mid in getattr(source_entity, "material_ids", [])
        if mid in world.materials
    }
    asset_surfaces = {
        sid: copy.deepcopy(world.surfaces[sid])
        for sid in getattr(source_entity, "surface_ids", [])
        if sid in world.surfaces
    }

    scale_meta = (
        copy.deepcopy(world.metadata.get("scale"))
        if world.metadata and "scale" in world.metadata
        else {"state": "metric", "meters_per_unit": 1.0}
    )

    asset_world = WorldIR(
        id=f"world-{asset_id}",
        name=asset_name,
        version=1,
        entities={asset_ent.id: asset_ent},
        geometries=asset_geometries,
        materials=asset_materials,
        surfaces=asset_surfaces,
        metadata={
            "asset_id": asset_id,
            "source_world_id": world.id,
            "source_world_version": world.version,
            "scale": scale_meta,
        },
    )

    # 5. Export to binary GLB container
    glb_bytes = export_to_glb(asset_world, artifact_store=artifact_store)

    extracted = ExtractedAsset(
        id=asset_id,
        name=asset_name,
        category=cat,
        source_world_id=world.id,
        source_entity_id=source_entity.id,
        dimensions=dimensions,
        provenance=prov,
        geometry_ids=list(getattr(source_entity, "geometry_ids", [])),
        material_ids=list(getattr(source_entity, "material_ids", [])),
        metadata={"glb_size_bytes": len(glb_bytes)},
    )

    return extracted, asset_world, glb_bytes
