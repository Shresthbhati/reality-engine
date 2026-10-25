# exporters

`gltf/` -- real glTF 2.0 export (box-geometry entities only).
`usd/` -- real USD ASCII (`.usda`) export (box-geometry entities only).
See `docs/CAPABILITY_MATRIX.md` row R for exact scope and honest limits.
Blender/Unreal targets and materials/textures/hierarchy are not
implemented.

Box-based world compilers (P16-01, 2026-10-03; shared input in `boxes.py`): `gis/` GeoJSON footprints, `ros/` Gazebo
SDF world, `godot/` .tscn, `unreal/` editor Python script, `sumo/` netconvert node+edge files, `habitat/` stage
bundle. Structure-validated by tests/test_box_exporters.py; NONE has been loaded into its target runtime (not installed
on the build machine). `sumo` and `habitat` produce several files: the CLI writes them into the `-o` directory.
