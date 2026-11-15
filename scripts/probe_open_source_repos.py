"""Read-only: query GitHub + Hugging Face metadata (licence, activity, HEAD sha) for the candidate repos/weights.
Usage: python scripts/probe_open_source_repos.py   (needs an authenticated `gh`); writes repos_meta.json in the cwd.
The committed result is engine/technologies/open_source_probe.json (probed 2026-11-15)."""
import json, subprocess, concurrent.futures as cf, httpx
REPOS = """colmap/colmap cvg/Hierarchical-Localization openMVG/openMVG alicevision/AliceVision alicevision/Meshroom cdcseacave/openMVS
IDEA-Research/GroundingDINO facebookresearch/sam2 IDEA-Research/Grounded-SAM-2 facebookresearch/dinov2 microsoft/Florence-2 facebookresearch/detectron2 IDEA-Research/Grounded-Segment-Anything
DepthAnything/Depth-Anything-V2 YvanYin/Metric3D prs-eth/Marigold
isl-org/Open3D PointCloudLibrary/pcl Pointcept/Pointcept mikedh/trimesh CGAL/cgal libigl/libigl elalish/manifold
facebookresearch/co-tracker open-mmlab/mmtracking opencv/opencv ifzhang/ByteTrack
nerfstudio-project/gsplat nerfstudio-project/nerfstudio pierotofy/OpenSplat PixarAnimationStudios/OpenUSD
IfcOpenShell/IfcOpenShell ThatOpen/engine_web-ifc xeokit/xeokit-sdk qgis/QGIS compas-dev/compas blender/blender
cvat-ai/cvat HumanSignal/label-studio voxel51/fiftyone ScanNet/ScanNet apple/ARKitScenes scannetpp/scannetpp apple/ml-hypersim bertjiazheng/Structured3D
cvg/LightGlue facebookresearch/vggt naver/mast3r naver/dust3r ultralytics/ultralytics open-mmlab/mmdetection facebookresearch/Mask2Former
cvg/glue-factory cvg/pixel-perfect-sfm colmap/pycolmap cvg/limap luigifreda/pyslam apple/ml-depth-pro DepthAnything/Depth-Anything-V3
SysCV/sam-hq IDEA-Research/OpenSeeD OpenGVLab/InternImage microsoft/GLIP isl-org/MiDaS LiheYoung/Depth-Anything
graphdeco-inria/gaussian-splatting nianticlabs/mickey OpenDroneMap/ODM OpenDroneMap/WebODM""".split()
HF = ["facebook/sam2.1-hiera-small","facebook/sam2.1-hiera-large","IDEA-Research/grounding-dino-tiny","IDEA-Research/grounding-dino-base","facebook/dinov2-small","facebook/dinov2-base",
      "depth-anything/Depth-Anything-V2-Small-hf","depth-anything/Depth-Anything-V2-Base-hf","depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf","prs-eth/marigold-depth-v1-1",
      "microsoft/Florence-2-base","facebook/cotracker3","facebook/VGGT-1B","apple/DepthPro","LiheYoung/depth_anything_vitl14","ETH-CVG/lightglue_superpoint","facebook/mask2former-swin-tiny-coco-instance",
      "Intel/dpt-hybrid-midas","zhoutiancheng/metric3d_vit_small" ]
def gh(path):
    r = subprocess.run(["gh", "api", path], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    return json.loads(r.stdout) if r.returncode == 0 and (r.stdout or "").strip() else {"_error": (r.stderr or r.stdout)[:120]}
def one(repo):
    m = gh(f"repos/{repo}")
    if "_error" in m: return repo, {"error": m["_error"]}
    c = gh(f"repos/{repo}/commits?per_page=1")
    rel = gh(f"repos/{repo}/releases/latest")
    return repo, {"full_name": m.get("full_name"), "language": m.get("language"), "license": (m.get("license") or {}).get("spdx_id"),
                  "license_name": (m.get("license") or {}).get("name"), "pushed_at": m.get("pushed_at"), "archived": m.get("archived"),
                  "open_issues": m.get("open_issues_count"), "stars": m.get("stargazers_count"), "default_branch": m.get("default_branch"),
                  "head_sha": (c[0]["sha"] if isinstance(c, list) and c else None), "latest_release": (rel.get("tag_name") if "_error" not in rel else None),
                  "moved_to": m.get("full_name") if m.get("full_name", "").lower() != repo.lower() else None}
with cf.ThreadPoolExecutor(8) as ex: repos = dict(ex.map(one, REPOS))
hf = {}
for mid in HF:
    try:
        r = httpx.get(f"https://huggingface.co/api/models/{mid}", timeout=30)
        d = r.json() if r.status_code == 200 else {}
        hf[mid] = {"status": r.status_code, "license": (d.get("cardData") or {}).get("license") or next((t.split(":",1)[1] for t in d.get("tags", []) if t.startswith("license:")), None),
                   "gated": d.get("gated"), "last_modified": d.get("lastModified"), "sha": d.get("sha")}
    except Exception as e: hf[mid] = {"error": str(e)[:80]}
json.dump({"repos": repos, "hf": hf}, open("repos_meta.json", "w"), indent=1)
print(len(repos), "repos", sum("error" in v for v in repos.values()), "errors;", len(hf), "hf models")
