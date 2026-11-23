"""Licence / access records for the six candidate datasets.

NOTHING HERE ACCEPTS TERMS, CREATES AN ACCOUNT OR DOWNLOADS ANYTHING. Each record states what the USER must do. ``verified`` is True only
where the cited official page said it (retrieved 2026-11-23); ``commercial_use=None`` means the terms text was NOT read by this tooling:
treat every dataset here as research / internal-benchmark only until the user has read the actual terms (several are known to restrict
commercial use and redistribution, which matters for a commercial product and for training on the data).
"""

from __future__ import annotations

from .manifest import Licence

LICENCES: dict[str, Licence] = {
    "scannet": Licence(
        name="ScanNet Terms of Use", url="http://kaldir.vc.cit.tum.de/scannet/ScanNet_TOS.pdf",
        registration_required=True, verified=True,
        user_actions=[
            "Read the ScanNet Terms of Use (URL above).",
            "Fill in the agreement and send it from an institutional e-mail address to scannet@googlegroups.com (per the official README).",
            "Wait for the reply with the download script (the README says to allow a week); download it yourself into a local folder.",
            "Export frames from each <scanId>.sens with the official SensReader (python) into <scene>/exported/{color,depth,pose,intrinsic}.",
        ],
        notes="Commercial-use terms NOT checked here; assume research-only until you read the PDF.",
    ),
    "scannetpp": Licence(
        name="ScanNet++ Terms of Use", url="https://kaldir.vc.in.tum.de/scannetpp/",
        registration_required=True, verified=False,
        user_actions=[
            "Open the ScanNet++ site, read the terms, and submit the data-access request yourself (account/terms are yours to accept).",
            "Download the default group (DSLR 2MP + meshes/semantics ~ 1.5 TB with iPhone) or just 'DSLR (2 MP)' + 'Meshes and semantics' "
            "(documentation: 371 GB + 132 GB) into <root>/data/<scene_id>/ plus <root>/split/.",
            "The nvs_test split deliberately has NO scan data: such scenes cannot be scored for geometry.",
        ],
        notes="Access procedure and commercial terms are UNVERIFIED (only the on-disk documentation page was read).",
    ),
    "arkitscenes": Licence(
        name="ARKitScenes licence (Apple)", url="https://github.com/apple/ARKitScenes",
        registration_required=False, verified=False,
        user_actions=[
            "Read the LICENSE in the ARKitScenes repository before downloading; this tooling did not verify its terms (believed CC BY-NC-SA).",
            "Download yourself with the official download_data.py (documented in DATA.md), e.g. `python3 download_data.py 3dod --video_id <id> "
            "--download_dir <dir>`; 3dod is 623.4 GB for all 5047 scans, so fetch only the selected video ids.",
            "Raw laser-scanner point clouds (`--download_laser_scanner_point_cloud`, raw dataset only) are the only independent physical GT; "
            "this adapter does not wire them yet.",
        ],
        notes="Licence terms UNVERIFIED. Registration is not documented in DATA.md; terms still apply.",
    ),
    "structured3d": Licence(
        name="Structured3D Terms of Use", url="https://drive.google.com/open?id=13ZwWpU_557ZQccwOUJ8H5lvXD7MeZFMa",
        registration_required=True, verified=True,
        user_actions=[
            "Read the Structured3D Terms of Use (URL above).",
            "Fill in the agreement form linked from the official README (https://forms.gle/LXg4bcjC2aEjrL9o8); the authors then grant download access.",
            "Download the selected scene_XXXXX folders yourself. Official split: scene_00000-02999 train, 03000-03249 val, 03250-03499 test.",
        ],
        notes="SYNTHETIC (rendered from designer layouts): never mix its numbers with real-capture results. Commercial terms NOT checked.",
    ),
    "3rscan": Licence(
        name="3RScan Terms of Use", url="https://github.com/WaldJohannaU/3RScan",
        registration_required=True, verified=False,
        user_actions=[
            "Read the 3RScan terms on the project page and follow its access procedure yourself (the README does not describe it; UNVERIFIED).",
            "Download the selected <scanId> folders, then UNZIP each sequence.zip into <scanId>/sequence/ yourself (this tooling never extracts archives).",
            "Keep 3RScan.json next to the scan folders: it defines reference/rescan groups, which must stay in one split.",
        ],
        notes="Access procedure and commercial terms UNVERIFIED.",
    ),
    "multiscan": Licence(
        name="MultiScan dataset terms (Hugging Face)", url="https://huggingface.co/datasets/3dlg-hcvc/MultiScan",
        registration_required=True, verified=False,
        user_actions=[
            "Open the Hugging Face dataset page, read its terms/gating conditions and accept them yourself if required (UNVERIFIED which apply).",
            "Download and `unzip \"*.zip\"` yourself (per the official README).",
            "RGB exists only as <scene>.mp4 (and depth as .depth.zlib): decode frames yourself to <scene>/frames/<NNNNNN>.png|jpg where NNNNNN is "
            "the 0-based line index in <scene>.jsonl. This frames convention is OURS, not the dataset's.",
        ],
        notes="Terms UNVERIFIED. The official docs list the mp4 -> png decode instructions as a TODO.",
    ),
}
