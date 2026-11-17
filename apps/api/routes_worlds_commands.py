"""Commands, mutations, and undo/redo history for world entities (T-4b.1 and T-6.7).

Provides:
- POST /api/worlds/{world_id}/commands: execute move, rotate, transform, delete, replace, surface_finish
- POST /api/worlds/{world_id}/undo: revert most recent active command
- POST /api/worlds/{world_id}/redo: re-apply most recently undone command
- GET /api/worlds/{world_id}/commands: list persistent edit command history
"""

from __future__ import annotations

import math
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service as wss
from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World, WorldEditCommand, new_id, utcnow
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Provenance,
    SurfaceFinish,
    Vector3,
)
from world_ir.world_v1 import WorldIR

commands_router = APIRouter(tags=["commands"])


class WorldCommandIn(BaseModel):
    command_type: str  # move, rotate, delete, replace, surface_finish, transform
    entity_id: str
    params: dict = {}
    description: str | None = None


@commands_router.post("/{world_id}/commands", status_code=201)
async def execute_world_command(
    world_id: str,
    body: WorldCommandIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Execute an edit command on a world entity (T-4b.1 and T-6.7).

    Mutations advance HEAD with provenance GENERATED and persist in WorldEditCommand
    for lossless, reloadable undo/redo with complete world isolation.
    """
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    worldir = wss.load_worldir_from_store(w)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for current version")
    worldir.id = w.id

    cmd_type = body.command_type.lower()
    entity_id = body.entity_id

    if cmd_type != "create" and entity_id not in worldir.entities:
        raise HTTPException(404, f"Entity '{entity_id}' not found in world")

    ent = worldir.entities.get(entity_id)
    inverse_params: dict = {}

    if cmd_type == "move":
        cur_pos = dict((ent.transform or {}).get("position") or {"x": 0.0, "y": 0.0, "z": 0.0})
        inverse_params = {"position": cur_pos}
        p = body.params.get("position")
        if p is not None:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                new_pos = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            elif isinstance(p, dict):
                new_pos = {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
            else:
                raise HTTPException(422, "position must be [x, y, z] or {x, y, z}")
        elif any(k in body.params for k in ("dx", "dy", "dz")):
            new_pos = {
                "x": cur_pos["x"] + float(body.params.get("dx", 0.0)),
                "y": cur_pos["y"] + float(body.params.get("dy", 0.0)),
                "z": cur_pos["z"] + float(body.params.get("dz", 0.0)),
            }
        else:
            raise HTTPException(422, "move requires 'position' or delta 'dx'/'dy'/'dz'")

        if not ent.transform:
            ent.transform = {}
        ent.transform["position"] = new_pos
        ent.provenance = Provenance.GENERATED

    elif cmd_type == "rotate":
        cur_rot = list((ent.transform or {}).get("rotation") or [0.0, 0.0, 0.0])
        cur_yaw = float((ent.custom_properties or {}).get("yaw_rad", 0.0))
        inverse_params = {"rotation": cur_rot, "yaw_rad": cur_yaw}
        if "yaw_deg" in body.params:
            yaw_rad = math.radians(float(body.params["yaw_deg"]))
        elif "yaw_rad" in body.params:
            yaw_rad = float(body.params["yaw_rad"])
        elif "rotation" in body.params:
            r = body.params["rotation"]
            yaw_rad = float(r[1]) if isinstance(r, (list, tuple)) and len(r) > 1 else 0.0
        else:
            raise HTTPException(422, "rotate requires 'yaw_deg', 'yaw_rad', or 'rotation'")

        if not ent.transform:
            ent.transform = {}
        ent.transform["rotation"] = [0.0, yaw_rad, 0.0]
        ent.custom_properties = dict(ent.custom_properties or {})
        ent.custom_properties["yaw_rad"] = yaw_rad
        ent.provenance = Provenance.GENERATED

    elif cmd_type == "transform":
        cur_pos = dict((ent.transform or {}).get("position") or {"x": 0.0, "y": 0.0, "z": 0.0})
        cur_rot = list((ent.transform or {}).get("rotation") or [0.0, 0.0, 0.0])
        cur_yaw = float((ent.custom_properties or {}).get("yaw_rad", 0.0))
        inverse_params = {"position": cur_pos, "rotation": cur_rot, "yaw_rad": cur_yaw}
        if "position" in body.params:
            p = body.params["position"]
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                new_pos = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            elif isinstance(p, dict):
                new_pos = {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
            else:
                raise HTTPException(422, "position must be [x, y, z] or {x, y, z}")
            if not ent.transform:
                ent.transform = {}
            ent.transform["position"] = new_pos
        if "rotation" in body.params or "yaw_rad" in body.params or "yaw_deg" in body.params:
            if "yaw_deg" in body.params:
                yaw_rad = math.radians(float(body.params["yaw_deg"]))
            elif "yaw_rad" in body.params:
                yaw_rad = float(body.params["yaw_rad"])
            else:
                r = body.params["rotation"]
                yaw_rad = float(r[1]) if isinstance(r, (list, tuple)) and len(r) > 1 else 0.0
            if not ent.transform:
                ent.transform = {}
            ent.transform["rotation"] = [0.0, yaw_rad, 0.0]
            ent.custom_properties = dict(ent.custom_properties or {})
            ent.custom_properties["yaw_rad"] = yaw_rad
        ent.provenance = Provenance.GENERATED

    elif cmd_type == "delete":
        inverse_params = {"entity": ent.to_dict()}
        del worldir.entities[entity_id]

    elif cmd_type == "replace":
        from engine.furniture.catalog import get_catalog_item

        catalog_id = body.params.get("catalog_id")
        if not catalog_id:
            raise HTTPException(422, "replace command requires 'catalog_id'")
        cat_item = get_catalog_item(catalog_id)
        if not cat_item:
            raise HTTPException(404, f"Catalog item '{catalog_id}' not found")

        inverse_params = {"entity": ent.to_dict()}
        w_dims = cat_item.dimensions
        w_val = float(w_dims.get("width", 1.0))
        d_val = float(w_dims.get("depth", 1.0))
        h_val = float(w_dims.get("height", 1.0))

        geom_id = f"geom-cat-{cat_item.id}-{uuid4().hex[:6]}"
        mat_id = f"mat-cat-{cat_item.id}-{uuid4().hex[:6]}"

        new_geom = Geometry(
            id=geom_id,
            type=GeometryType.BOX,
            bounds_min=Vector3(x=-w_val / 2, y=0.0, z=-d_val / 2),
            bounds_max=Vector3(x=w_val / 2, y=h_val, z=d_val / 2),
        )
        new_mat = Material(
            id=mat_id,
            name=f"{cat_item.name} Material",
            class_name=cat_item.category,
            surface_finish=SurfaceFinish.SMOOTH,
            color_rgb=cat_item.color_rgb,
            provenance=Provenance.GENERATED,
            confidence=1.0,
        )
        worldir.geometries[geom_id] = new_geom
        worldir.materials[mat_id] = new_mat

        ent.name = cat_item.name
        ent.type = EntityType.FURNITURE
        ent.geometry_ids = [geom_id]
        ent.material_ids = [mat_id]
        ent.semantic_labels = ["furniture", cat_item.category, "cc0_catalog"]
        ent.provenance = Provenance.GENERATED
        ent.confidence = 1.0
        ent.custom_properties = dict(ent.custom_properties or {})
        ent.custom_properties.update({
            "catalog_id": cat_item.id,
            "catalog_name": cat_item.name,
            "category": cat_item.category,
            "license": cat_item.license,
            "license_url": cat_item.license_url,
            "author": cat_item.author,
            "dimensions": cat_item.dimensions,
        })

    elif cmd_type == "surface_finish":
        inverse_params = {
            "material_ids": list(ent.material_ids or []),
            "custom_properties": dict(ent.custom_properties or {}),
        }
        finish_val = SurfaceFinish.SMOOTH
        color_rgb = (0.8, 0.8, 0.8)
        mat_name = "Custom Finish"
        mat_class = "finish"

        val = body.params
        if "surface_finish" in val:
            try:
                finish_val = SurfaceFinish(val["surface_finish"])
            except ValueError:
                finish_val = SurfaceFinish.SMOOTH
        if "color_rgb" in val and isinstance(val["color_rgb"], (list, tuple)) and len(val["color_rgb"]) == 3:
            color_rgb = tuple(float(c) for c in val["color_rgb"])
        if "name" in val:
            mat_name = str(val["name"])
        if "class_name" in val:
            mat_class = str(val["class_name"])

        mat_id = f"mat-{ent.id}-{uuid4().hex[:6]}"
        new_mat = Material(
            id=mat_id,
            name=mat_name,
            class_name=mat_class,
            surface_finish=finish_val,
            color_rgb=color_rgb,
            provenance=Provenance.GENERATED,
            confidence=1.0,
        )
        worldir.materials[mat_id] = new_mat
        ent.material_ids = [mat_id]
        ent.custom_properties = dict(ent.custom_properties or {})
        ent.custom_properties["surface_finish"] = finish_val.value
        ent.custom_properties["material"] = mat_class
        ent.custom_properties["material_label"] = mat_name
        ent.custom_properties["design_layer"] = val.get("design_layer", "finish")
        ent.provenance = Provenance.GENERATED

    else:
        raise HTTPException(422, f"Unknown command type '{cmd_type}'")

    desc = body.description or f"{cmd_type.capitalize()} {entity_id}"

    vrow = await wss.commit_version(
        db,
        world_id=w.id,
        world=worldir,
        parent=w.current_version_id,
        label=desc,
    )

    res_seq = await db.execute(
        select(func.max(WorldEditCommand.sequence)).where(WorldEditCommand.world_id == world_id)
    )
    next_seq = (res_seq.scalar() or 0) + 1

    cmd_record = WorldEditCommand(
        id=new_id("cmd"),
        world_id=world_id,
        team_id=w.team_id,
        command_type=cmd_type,
        entity_id=entity_id,
        description=desc,
        params=body.params,
        inverse_params=inverse_params,
        version_id_before=w.current_version_id,
        version_id_after=vrow.id,
        undone=False,
        sequence=next_seq,
        created_at=utcnow(),
    )
    db.add(cmd_record)
    await db.commit()

    return {
        "command_id": cmd_record.id,
        "world_id": world_id,
        "version_id": vrow.id,
        "command_type": cmd_type,
        "entity_id": entity_id,
        "sequence": next_seq,
        "provenance": "generated",
        "description": desc,
    }


@commands_router.post("/{world_id}/undo")
async def undo_world_command(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Revert the most recent active edit command for this world (T-6.7)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    res = await db.execute(
        select(WorldEditCommand)
        .where(WorldEditCommand.world_id == world_id, WorldEditCommand.undone.is_(False))
        .order_by(WorldEditCommand.sequence.desc())
        .limit(1)
    )
    cmd = res.scalars().first()
    if cmd is None:
        raise HTTPException(400, "Nothing to undo")

    worldir = wss.load_worldir_from_store(w)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for current version")
    worldir.id = w.id

    inv = cmd.inverse_params or {}
    cmd_type = cmd.command_type.lower()

    if cmd_type in ("move", "rotate", "transform"):
        ent = worldir.entities.get(cmd.entity_id)
        if ent:
            if "position" in inv:
                if not ent.transform:
                    ent.transform = {}
                ent.transform["position"] = inv["position"]
            if "rotation" in inv:
                if not ent.transform:
                    ent.transform = {}
                ent.transform["rotation"] = inv["rotation"]
            if "yaw_rad" in inv:
                ent.custom_properties = dict(ent.custom_properties or {})
                ent.custom_properties["yaw_rad"] = inv["yaw_rad"]
    elif cmd_type in ("delete", "replace"):
        ent_dict = inv.get("entity")
        if ent_dict:
            worldir.entities[cmd.entity_id] = Entity.from_dict(ent_dict)
    elif cmd_type == "surface_finish":
        ent = worldir.entities.get(cmd.entity_id)
        if ent:
            if "material_ids" in inv:
                ent.material_ids = inv["material_ids"]
            if "custom_properties" in inv:
                ent.custom_properties = inv["custom_properties"]

    vrow = await wss.commit_version(
        db,
        world_id=w.id,
        world=worldir,
        parent=w.current_version_id,
        label=f"Undo: {cmd.description}",
    )

    cmd.undone = True
    cmd.undone_at = utcnow()
    await db.commit()

    return {
        "undone": True,
        "command_id": cmd.id,
        "command_type": cmd.command_type,
        "entity_id": cmd.entity_id,
        "sequence": cmd.sequence,
        "current_version_id": vrow.id,
        "description": f"Undo: {cmd.description}",
    }


@commands_router.post("/{world_id}/redo")
async def redo_world_command(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Re-apply the most recently undone command for this world (T-6.7)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    res_active = await db.execute(
        select(func.max(WorldEditCommand.sequence))
        .where(WorldEditCommand.world_id == world_id, WorldEditCommand.undone.is_(False))
    )
    active_seq = res_active.scalar() or 0

    res = await db.execute(
        select(WorldEditCommand)
        .where(
            WorldEditCommand.world_id == world_id,
            WorldEditCommand.undone.is_(True),
            WorldEditCommand.sequence == active_seq + 1,
        )
        .limit(1)
    )
    cmd = res.scalars().first()
    if cmd is None:
        res = await db.execute(
            select(WorldEditCommand)
            .where(WorldEditCommand.world_id == world_id, WorldEditCommand.undone.is_(True))
            .order_by(WorldEditCommand.sequence.asc())
            .limit(1)
        )
        cmd = res.scalars().first()

    if cmd is None:
        raise HTTPException(400, "Nothing to redo")

    worldir = wss.load_worldir_from_store(w)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for current version")
    worldir.id = w.id

    cmd_type = cmd.command_type.lower()
    params = cmd.params or {}

    if cmd_type == "move":
        ent = worldir.entities.get(cmd.entity_id)
        if ent:
            p = params.get("position")
            if p is not None:
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    new_pos = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                elif isinstance(p, dict):
                    new_pos = {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                else:
                    new_pos = ent.transform.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
                if not ent.transform:
                    ent.transform = {}
                ent.transform["position"] = new_pos
                ent.provenance = Provenance.GENERATED

    elif cmd_type == "rotate":
        ent = worldir.entities.get(cmd.entity_id)
        if ent:
            if "yaw_deg" in params:
                yaw_rad = math.radians(float(params["yaw_deg"]))
            elif "yaw_rad" in params:
                yaw_rad = float(params["yaw_rad"])
            elif "rotation" in params:
                r = params["rotation"]
                yaw_rad = float(r[1]) if isinstance(r, (list, tuple)) and len(r) > 1 else 0.0
            else:
                yaw_rad = 0.0
            if not ent.transform:
                ent.transform = {}
            ent.transform["rotation"] = [0.0, yaw_rad, 0.0]
            ent.custom_properties = dict(ent.custom_properties or {})
            ent.custom_properties["yaw_rad"] = yaw_rad
            ent.provenance = Provenance.GENERATED

    elif cmd_type == "transform":
        ent = worldir.entities.get(cmd.entity_id)
        if ent:
            if "position" in params:
                p = params["position"]
                if isinstance(p, (list, tuple)) and len(p) >= 3:
                    new_pos = {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                elif isinstance(p, dict):
                    new_pos = {"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0)), "z": float(p.get("z", 0.0))}
                else:
                    new_pos = ent.transform.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
                if not ent.transform:
                    ent.transform = {}
                ent.transform["position"] = new_pos
            if "rotation" in params or "yaw_rad" in params or "yaw_deg" in params:
                if "yaw_deg" in params:
                    yaw_rad = math.radians(float(params["yaw_deg"]))
                elif "yaw_rad" in params:
                    yaw_rad = float(params["yaw_rad"])
                else:
                    r = params["rotation"]
                    yaw_rad = float(r[1]) if isinstance(r, (list, tuple)) and len(r) > 1 else 0.0
                if not ent.transform:
                    ent.transform = {}
                ent.transform["rotation"] = [0.0, yaw_rad, 0.0]
                ent.custom_properties = dict(ent.custom_properties or {})
                ent.custom_properties["yaw_rad"] = yaw_rad
            ent.provenance = Provenance.GENERATED

    elif cmd_type == "delete":
        if cmd.entity_id in worldir.entities:
            del worldir.entities[cmd.entity_id]

    elif cmd_type == "replace":
        from engine.furniture.catalog import get_catalog_item

        catalog_id = params.get("catalog_id")
        cat_item = get_catalog_item(catalog_id) if catalog_id else None
        ent = worldir.entities.get(cmd.entity_id)
        if cat_item and ent:
            w_dims = cat_item.dimensions
            w_val = float(w_dims.get("width", 1.0))
            d_val = float(w_dims.get("depth", 1.0))
            h_val = float(w_dims.get("height", 1.0))

            geom_id = f"geom-cat-{cat_item.id}-{uuid4().hex[:6]}"
            mat_id = f"mat-cat-{cat_item.id}-{uuid4().hex[:6]}"

            new_geom = Geometry(
                id=geom_id,
                type=GeometryType.BOX,
                bounds_min=Vector3(x=-w_val / 2, y=0.0, z=-d_val / 2),
                bounds_max=Vector3(x=w_val / 2, y=h_val, z=d_val / 2),
            )
            new_mat = Material(
                id=mat_id,
                name=f"{cat_item.name} Material",
                class_name=cat_item.category,
                surface_finish=SurfaceFinish.SMOOTH,
                color_rgb=cat_item.color_rgb,
                provenance=Provenance.GENERATED,
                confidence=1.0,
            )
            worldir.geometries[geom_id] = new_geom
            worldir.materials[mat_id] = new_mat

            ent.name = cat_item.name
            ent.type = EntityType.FURNITURE
            ent.geometry_ids = [geom_id]
            ent.material_ids = [mat_id]
            ent.semantic_labels = ["furniture", cat_item.category, "cc0_catalog"]
            ent.provenance = Provenance.GENERATED
            ent.confidence = 1.0
            ent.custom_properties = dict(ent.custom_properties or {})
            ent.custom_properties.update({
                "catalog_id": cat_item.id,
                "catalog_name": cat_item.name,
                "category": cat_item.category,
                "license": cat_item.license,
                "license_url": cat_item.license_url,
                "author": cat_item.author,
                "dimensions": cat_item.dimensions,
            })

    elif cmd_type == "surface_finish":
        ent = worldir.entities.get(cmd.entity_id)
        if ent:
            finish_val = SurfaceFinish.SMOOTH
            color_rgb = (0.8, 0.8, 0.8)
            mat_name = "Custom Finish"
            mat_class = "finish"

            val = params
            if "surface_finish" in val:
                try:
                    finish_val = SurfaceFinish(val["surface_finish"])
                except ValueError:
                    finish_val = SurfaceFinish.SMOOTH
            if "color_rgb" in val and isinstance(val["color_rgb"], (list, tuple)) and len(val["color_rgb"]) == 3:
                color_rgb = tuple(float(c) for c in val["color_rgb"])
            if "name" in val:
                mat_name = str(val["name"])
            if "class_name" in val:
                mat_class = str(val["class_name"])

            mat_id = f"mat-{ent.id}-{uuid4().hex[:6]}"
            new_mat = Material(
                id=mat_id,
                name=mat_name,
                class_name=mat_class,
                surface_finish=finish_val,
                color_rgb=color_rgb,
                provenance=Provenance.GENERATED,
                confidence=1.0,
            )
            worldir.materials[mat_id] = new_mat
            ent.material_ids = [mat_id]
            ent.custom_properties = dict(ent.custom_properties or {})
            ent.custom_properties["surface_finish"] = finish_val.value
            ent.custom_properties["material"] = mat_class
            ent.custom_properties["material_label"] = mat_name
            ent.custom_properties["design_layer"] = val.get("design_layer", "finish")
            ent.provenance = Provenance.GENERATED

    vrow = await wss.commit_version(
        db,
        world_id=w.id,
        world=worldir,
        parent=w.current_version_id,
        label=f"Redo: {cmd.description}",
    )

    cmd.undone = False
    cmd.undone_at = None
    await db.commit()

    return {
        "redone": True,
        "command_id": cmd.id,
        "command_type": cmd.command_type,
        "entity_id": cmd.entity_id,
        "sequence": cmd.sequence,
        "current_version_id": vrow.id,
        "description": f"Redo: {cmd.description}",
    }


@commands_router.get("/{world_id}/commands")
async def get_world_commands(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve persistent edit command history for this world (T-6.7).

    Guarantees strict per-world isolation: only commands recorded for world_id
    are returned, and reload preserves undo/redo status.
    """
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    res = await db.execute(
        select(WorldEditCommand)
        .where(WorldEditCommand.world_id == world_id)
        .order_by(WorldEditCommand.sequence.asc())
    )
    cmds = res.scalars().all()

    can_undo = any(not c.undone for c in cmds)
    can_redo = any(c.undone for c in cmds)

    return {
        "world_id": world_id,
        "can_undo": can_undo,
        "can_redo": can_redo,
        "commands": [
            {
                "id": c.id,
                "command_type": c.command_type,
                "entity_id": c.entity_id,
                "description": c.description,
                "undone": c.undone,
                "sequence": c.sequence,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "undone_at": c.undone_at.isoformat() if c.undone_at else None,
            }
            for c in cmds
        ],
    }
