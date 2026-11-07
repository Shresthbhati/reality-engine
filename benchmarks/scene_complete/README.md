# scene_complete eval set (task 8.6)

Scorer, runner and label checker for the scene-complete reconstruction spec
(`docs/future/perception/SCENE_COMPLETE_RECONSTRUCTION.md`).

    python -m benchmarks.scene_complete.check                 # validate labels/ against images/ before scoring
    python -m benchmarks.scene_complete.run images/ --source real   # photos -> predictions/ (real Mask R-CNN, CPU)
    python -m benchmarks.scene_complete.score predictions/    # recall / precision per tier and overall

**Status: tooling done, data missing.** The acceptance target (>= 20 AI-generated + >= 20 real room photos,
hand-labelled) is NOT met: `labels/` holds one synthetic self-check. Labels must be written by a person looking at the
photo. Do not generate them from a model's output: that would score the detector against itself.

## How to add a photo (about 2 minutes each)

1. Copy the photo to `images/<stem>.jpg` (keep it out of git if it is large or not yours to publish).
2. Write `labels/<stem>.json`, same stem:

       {"image": "<stem>.jpg", "source": "real",
        "objects": [{"class": "sofa", "tier": "furniture", "box": [x0, y0, x1, y1], "support": "floor"}]}

   - `source`: `real` (a photograph of a real place) or `ai` (an AI-generated image).
   - `box`: pixels in the ORIGINAL image, `[left, top, right, bottom]`.
   - `class`: lowercase with underscores. Furniture the engine can place: `sofa`, `armchair`, `chair`, `dining_table`,
     `coffee_table`, `desk`, `bed`, `wardrobe`, `bookshelf`, `dresser`, `nightstand`, `tv_stand`, plus `tv` and `lamp`.
     Other classes (`switch`, `socket`, `radiator`, `rug`, ...) are allowed; they only matter for the open-vocabulary stage (8.8).
   - `tier`: `furniture`, `appliance`, `infrastructure`, `decor`, `small`, `architecture`.
   - `support`: `floor`, `wall`, `ceiling`, `table`, `shelf`, `other`, `unknown`.
   - Label every visible object of those classes, including partly hidden ones (box the visible part).
3. Run `check`; fix every ERROR it prints (a misspelt class can never match, so it silently lowers the score).

## What to collect

- 20 AI + 20 real photos of rooms; mix of living rooms, bedrooms, kitchens, offices; some cluttered, some dark, a few with
  mirrors or glass (the spec's failure modes).
- The M1 acceptance target is COCO-class furniture recall >= 70 % at IoU 0.5, with floor contact < 5 % of room height
  (the contact check is a separate 3D measurement; this scorer only measures 2D detection).
