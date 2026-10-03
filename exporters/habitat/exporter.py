"""Habitat-Sim export: a STAGE bundle -- the scene as glTF plus the stage config and scene-dataset config that load it.

Files (``content`` is a JSON object {filename: text}; ``write_habitat_files`` writes them):

    scene.gltf                       the world, from the repo's own glTF exporter (real geometry where artifacts resolve)
    scene.stage_config.json          render_asset + the asset's up/front axes
    scene.scene_dataset_config.json  a dataset config that registers the stage

Axes: the glTF exporter writes WorldIR coordinates unchanged, i.e. the asset is Z-up. The stage config therefore
declares up = +Z and front = +Y so Habitat re-orients it to its own Y-up/-Z-front world; "front" is arbitrary for a
scene with no intrinsic facing. No semantic annotation, navmesh, objects or physical properties are invented
(a navmesh is computed by Habitat from the loaded stage). Not loaded in habitat-sim here (it is not installed): the
bundle structure is validated by tests, loading is UNVERIFIED.
"""

from __future__ import annotations

import json
from pathlib import Path

from exporters.gltf.exporter import export_to_gltf_with_report
from exporters.report import ExportReport, content_hash


def export_to_habitat_with_report(world, artifact_store=None):
    gltf, gltf_report = export_to_gltf_with_report(world, artifact_store)
    stage = {"render_asset": "scene.gltf", "up": [0, 0, 1], "front": [0, 1, 0], "origin": [0, 0, 0],
             "requires_lighting": True}
    dataset = {"stages": {"configs": ["scene.stage_config.json"]}}
    content = json.dumps({
        "scene.gltf": json.dumps(gltf, sort_keys=True),
        "scene.stage_config.json": json.dumps(stage, sort_keys=True, indent=1),
        "scene.scene_dataset_config.json": json.dumps(dataset, sort_keys=True, indent=1),
    }, sort_keys=True, indent=1)
    report = ExportReport(
        format="habitat", world_id=world.id, world_version=gltf_report.world_version,
        entities_exported=gltf_report.entities_exported, entities_skipped=gltf_report.entities_skipped,
        skip_reasons=gltf_report.skip_reasons, content_hash=content_hash(content))
    return content, report


def write_habitat_files(content: str, directory) -> list:
    """Write the files of an export_to_habitat_with_report() result into ``directory``; returns their paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in json.loads(content).items():
        p = out / name
        p.write_text(text, encoding="utf-8")
        paths.append(p)
    return paths
