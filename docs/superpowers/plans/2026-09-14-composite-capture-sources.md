# Composite Capture Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `evidence/multi_source.py` so a single folder representing one physical acquisition (a phone or drone capture with synchronized RGB/video + IMU + GPS + calibration + telemetry subfolders) is represented and ingested as ONE composite `SourceRecord` with per-component provenance, instead of being silently flattened into an undifferentiated `DATASET`.

**Architecture:** No new evidence model. Visual components (`rgb/`, `video/`) of a detected composite folder are ingested through the existing `evidence.importers.import_folder`/`EvidencePackage` path exactly as today, then each resulting asset is tagged with a `ProcessingRecord` naming its source component (`EvidencePackage.record_processing`, already exists, unmodified). Non-visual sidecar components (`depth/`, `imu/`, `gps/`, `calibration/`, `telemetry/`) have no real parser in this codebase yet, so they are honestly recorded as a component file manifest (relative path + presence) on the `SourceRecord` — never fabricated as parsed sensor values. Cross-source registration reuses the existing `world_ir.coordinates.Transform`/`Frame` types; a source's relationship to the session frame defaults to an explicit `"unknown"` status and is only ever set by an explicit caller call, never inferred.

**Tech Stack:** Python 3, stdlib only (`os`, `hashlib`, `dataclasses`, `enum`), pytest. Reuses `evidence.importers`, `evidence.packages`, `evidence.session`, `world_ir.coordinates`, `provenance`.

**Spec:** The milestone brief this plan implements is the multi-part "Reality Engine — Universal Source Ingestion" instruction the user pasted into this session (no separate spec file exists in-repo); the two clauses this plan targets are:
- §10-11 (composite/multi-component sources; portable capture package: preserve original source, source components, timestamps, sensor identity, calibration, derived frames, telemetry, provenance)
- §65 (future cross-source registration: represent `source_frame_A`, `source_frame_B`, `session_frame`, `transform_A_to_session`; unknown state is `UNKNOWN`, never a fabricated identity transform)

## Global Constraints

- Do not create a parallel evidence system. Visual files inside a composite folder go through the SAME `evidence.importers.import_folder`/`DeterministicPackageBuilder`/`EvidencePackage` path every other folder source already uses.
- Do not fabricate metadata. IMU/GPS/calibration/telemetry files are recorded by relative path only — their internal fields are NEVER parsed or invented in this plan (no CSV/JSON schema for those formats exists in this repo).
- Do not assume coordinate-frame alignment across sources. A source's registration status defaults to `{"status": "unknown", "transform": None}` and changes ONLY via an explicit `register_source_frame()` call.
- Backward compatibility: a `session.json` written by the current `MultiSourceSession`/`SourceRecord` (without `components`/`registration` keys) MUST still load via `from_dict()` — every new field needs a safe default when absent.
- A plain photo/video folder with no sidecar subdirectory (the existing, already-tested behavior) MUST classify as `SourceType.DATASET` exactly as before — composite detection must never change behavior for non-composite folders.
- Every new public method/class gets at least one test exercising real files in a `tmp_path` fixture (no mocks of the ingestion path), matching the existing test style in `tests/test_multi_source_session.py`.

---

### Task 1: Composite-capture taxonomy, detection, and SourceRecord fields

**Files:**
- Modify: `evidence/multi_source.py` (enums, detection helpers, `SourceRecord` fields/serialization, `_source_type_of` signature)
- Test: `tests/test_multi_source_session.py` (new test classes `TestCompositeDetection`, `TestSourceRecordSerializationDefaults`)

**Interfaces:**
- Consumes: nothing new (stdlib `os` only).
- Produces (for Task 2 and Task 3 to consume):
  - `CaptureComponent(str, Enum)` with members `RGB`, `VIDEO`, `DEPTH`, `IMU`, `GPS`, `CALIBRATION`, `TELEMETRY` (values are the lowercase strings `"rgb"`, `"video"`, `"depth"`, `"imu"`, `"gps"`, `"calibration"`, `"telemetry"`).
  - `SourceType` gains members `PHONE_CAPTURE = "phone_capture"`, `DRONE_CAPTURE = "drone_capture"`, `COMPOSITE_CAPTURE = "composite_capture"` (existing members `IMAGE`, `VIDEO`, `POINT_CLOUD`, `DATASET`, `UNKNOWN` are unchanged).
  - `_detect_composite_components(path: str) -> Dict[CaptureComponent, List[str]]` — direct-child subdirectory scan; keys only for subdirectories that exist AND contain >=1 file (recursively); values are sorted relative paths (POSIX-style, `/` separator) from `path`.
  - `_is_composite_capture(components: Dict[CaptureComponent, List[str]]) -> bool`.
  - `_source_type_of(path: str, capture_type: Optional[str] = None) -> SourceType` — signature CHANGES to take an optional second parameter; existing single-arg callers (none exist outside this file yet) are unaffected since it's a new optional parameter.
  - `SourceRecord` gains two fields with safe defaults: `components: Dict[str, List[str]] = field(default_factory=dict)` (keys are `CaptureComponent.value` strings, e.g. `"gps"`) and `registration: dict = field(default_factory=lambda: {"status": "unknown", "transform": None})`.
  - `SourceRecord.to_dict()`/`from_dict()` include both new fields; `from_dict()` defaults `components` to `{}` and `registration` to `{"status": "unknown", "transform": None}` when the input dict lacks those keys (old-format compatibility).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_source_session.py` (after the existing `TestIncrementalMultiSource` class, before `TestEvidenceSummary`):

```python
class TestCompositeDetection:
    def test_plain_photo_folder_has_no_composite_components(self, tmp_path):
        from evidence.multi_source import _detect_composite_components, _is_composite_capture

        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(_jpeg())

        components = _detect_composite_components(str(folder))

        assert components == {}
        assert _is_composite_capture(components) is False

    def test_rgb_plus_gps_subfolders_detected_as_composite(self, tmp_path):
        from evidence.multi_source import CaptureComponent, _detect_composite_components, _is_composite_capture

        folder = tmp_path / "phone_capture_001"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "0001.jpg").write_bytes(_jpeg(0x01))
        (folder / "gps").mkdir()
        (folder / "gps" / "track.csv").write_text("lat,lon\n1.0,2.0\n")

        components = _detect_composite_components(str(folder))

        assert components[CaptureComponent.RGB] == ["rgb/0001.jpg"]
        assert components[CaptureComponent.GPS] == ["gps/track.csv"]
        assert _is_composite_capture(components) is True

    def test_rgb_only_subfolder_is_not_composite(self, tmp_path):
        # A folder with only a visual component (no sidecar) is not a
        # synchronized composite acquisition -- just an organized photo
        # folder. Must stay DATASET, not falsely promoted.
        from evidence.multi_source import _detect_composite_components, _is_composite_capture

        folder = tmp_path / "rgb_only"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())

        components = _detect_composite_components(str(folder))

        assert _is_composite_capture(components) is False

    def test_empty_component_subfolder_is_ignored(self, tmp_path):
        from evidence.multi_source import CaptureComponent, _detect_composite_components

        folder = tmp_path / "capture"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())
        (folder / "imu").mkdir()  # empty -- no files

        components = _detect_composite_components(str(folder))

        assert CaptureComponent.IMU not in components
        assert CaptureComponent.RGB in components

    def test_source_type_of_plain_folder_is_dataset(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(_jpeg())

        assert _source_type_of(str(folder)) is SourceType.DATASET

    def test_source_type_of_composite_without_capture_type_is_generic(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "capture"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())
        (folder / "imu").mkdir()
        (folder / "imu" / "log.csv").write_text("t,ax,ay,az\n")

        assert _source_type_of(str(folder)) is SourceType.COMPOSITE_CAPTURE

    def test_source_type_of_composite_with_explicit_phone_capture_type(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "capture"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "a.jpg").write_bytes(_jpeg())
        (folder / "gps").mkdir()
        (folder / "gps" / "track.csv").write_text("lat,lon\n")

        assert _source_type_of(str(folder), capture_type="phone") is SourceType.PHONE_CAPTURE

    def test_source_type_of_composite_with_explicit_drone_capture_type(self, tmp_path):
        from evidence.multi_source import _source_type_of, SourceType

        folder = tmp_path / "capture"
        (folder / "video").mkdir(parents=True)
        (folder / "video" / "flight.mp4").write_bytes(b"not-a-real-video")
        (folder / "telemetry").mkdir()
        (folder / "telemetry" / "log.json").write_text("{}")

        assert _source_type_of(str(folder), capture_type="drone") is SourceType.DRONE_CAPTURE


class TestSourceRecordSerializationDefaults:
    def test_round_trips_components_and_registration(self):
        from evidence.multi_source import CaptureComponent, SourceRecord, SourceStatus, SourceType

        record = SourceRecord(
            source_id="src-0000-abc",
            original_path="/tmp/capture",
            source_type=SourceType.COMPOSITE_CAPTURE,
            content_hash="abc123",
            status=SourceStatus.INGESTED,
            components={CaptureComponent.GPS.value: ["gps/track.csv"]},
        )

        restored = SourceRecord.from_dict(record.to_dict())

        assert restored.components == {"gps": ["gps/track.csv"]}
        assert restored.registration == {"status": "unknown", "transform": None}

    def test_from_dict_defaults_missing_new_fields_for_old_format(self):
        # Simulates a session.json written before this plan existed --
        # no "components"/"registration" keys at all.
        from evidence.multi_source import SourceRecord

        old_format = {
            "source_id": "src-0000-abc",
            "original_path": "/tmp/a.jpg",
            "source_type": "image",
            "content_hash": "abc123",
            "status": "ingested",
            "asset_ids": ["ev-x-0000-abc"],
            "unhandled_paths": [],
            "error": None,
        }

        restored = SourceRecord.from_dict(old_format)

        assert restored.components == {}
        assert restored.registration == {"status": "unknown", "transform": None}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_multi_source_session.py -k "TestCompositeDetection or TestSourceRecordSerializationDefaults" -v`
Expected: FAIL — `ImportError: cannot import name 'CaptureComponent'` (or `AttributeError`) since none of these names exist yet.

- [ ] **Step 3: Implement the taxonomy and detection helpers**

In `evidence/multi_source.py`, add this block immediately after the existing `class SourceType(str, Enum): ...` block (i.e. right before `def _source_type_of(path: str) -> SourceType:`):

```python
class CaptureComponent(str, Enum):
    """One synchronized component of a composite acquisition (spec §10:
    a phone/drone capture may bundle RGB, video, depth, IMU, GPS,
    calibration, and telemetry as related, not merged, evidence)."""

    RGB = "rgb"
    VIDEO = "video"
    DEPTH = "depth"
    IMU = "imu"
    GPS = "gps"
    CALIBRATION = "calibration"
    TELEMETRY = "telemetry"


#: Direct-child subdirectory names (case-insensitive) recognized as
#: composite-capture components. "images"/"photos" are accepted aliases
#: for RGB so a capture package can use whichever the source device
#: convention prefers.
_COMPONENT_DIR_NAMES: Dict[str, CaptureComponent] = {
    "rgb": CaptureComponent.RGB,
    "images": CaptureComponent.RGB,
    "photos": CaptureComponent.RGB,
    "video": CaptureComponent.VIDEO,
    "depth": CaptureComponent.DEPTH,
    "imu": CaptureComponent.IMU,
    "gps": CaptureComponent.GPS,
    "calibration": CaptureComponent.CALIBRATION,
    "telemetry": CaptureComponent.TELEMETRY,
}

#: Components this repo can decode into real evidence today (via the
#: existing photo/video importers). Everything else (DEPTH/IMU/GPS/
#: CALIBRATION/TELEMETRY) has no parser here yet, so it is recorded as a
#: file manifest only -- never fabricated as parsed sensor values.
_VISUAL_COMPONENTS = frozenset({CaptureComponent.RGB, CaptureComponent.VIDEO})
_SIDECAR_COMPONENTS = frozenset(
    {CaptureComponent.DEPTH, CaptureComponent.IMU, CaptureComponent.GPS,
     CaptureComponent.CALIBRATION, CaptureComponent.TELEMETRY}
)


def _detect_composite_components(path: str) -> Dict[CaptureComponent, List[str]]:
    """Scan `path`'s DIRECT child subdirectories for known composite-
    capture component names. Returns {component: [sorted relative
    paths]} for every matching, non-empty subdirectory; a subdirectory
    that exists but holds zero files is not reported (nothing to
    preserve). Paths are relative to `path`, POSIX-separated."""
    found: Dict[CaptureComponent, List[str]] = {}
    if not os.path.isdir(path):
        return found
    for entry in sorted(os.listdir(path)):
        component = _COMPONENT_DIR_NAMES.get(entry.lower())
        if component is None:
            continue
        subdir = os.path.join(path, entry)
        if not os.path.isdir(subdir):
            continue
        files: List[str] = []
        for root, _dirs, names in os.walk(subdir):
            for name in names:
                full = os.path.join(root, name)
                files.append(os.path.relpath(full, path).replace(os.sep, "/"))
        if files:
            files.sort()
            found[component] = files
    return found


def _is_composite_capture(components: Dict[CaptureComponent, List[str]]) -> bool:
    """A composite capture needs at least one visual component (rgb/
    video) PLUS at least one sidecar component (depth/imu/gps/
    calibration/telemetry). A folder with only rgb/video is an ordinary
    photo/video collection and must keep classifying as DATASET --
    composite detection must never change existing behavior for it."""
    has_visual = any(c in components for c in _VISUAL_COMPONENTS)
    has_sidecar = any(c in components for c in _SIDECAR_COMPONENTS)
    return has_visual and has_sidecar
```

Then REPLACE the existing `_source_type_of` function:

```python
def _source_type_of(path: str) -> SourceType:
    if os.path.isdir(path):
        return SourceType.DATASET
    extension = os.path.splitext(path)[1].lower()
    if extension in _VIDEO_EXTS:
        return SourceType.VIDEO
    if extension in _PHOTO_EXTS:
        return SourceType.IMAGE
    if extension in _LAS_EXTS:
        return SourceType.POINT_CLOUD
    return SourceType.UNKNOWN
```

with:

```python
def _source_type_of(path: str, capture_type: Optional[str] = None) -> SourceType:
    if os.path.isdir(path):
        components = _detect_composite_components(path)
        if _is_composite_capture(components):
            if capture_type == "phone":
                return SourceType.PHONE_CAPTURE
            if capture_type == "drone":
                return SourceType.DRONE_CAPTURE
            return SourceType.COMPOSITE_CAPTURE
        return SourceType.DATASET
    extension = os.path.splitext(path)[1].lower()
    if extension in _VIDEO_EXTS:
        return SourceType.VIDEO
    if extension in _PHOTO_EXTS:
        return SourceType.IMAGE
    if extension in _LAS_EXTS:
        return SourceType.POINT_CLOUD
    return SourceType.UNKNOWN
```

In the existing `class SourceType(str, Enum):` block, REPLACE:

```python
class SourceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"
    DATASET = "dataset"    # a directory of mixed evidence
    UNKNOWN = "unknown"    # unrecognized single-file extension
```

with:

```python
class SourceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"
    DATASET = "dataset"                        # a directory of mixed, non-composite evidence
    PHONE_CAPTURE = "phone_capture"             # composite: caller declared capture_type="phone"
    DRONE_CAPTURE = "drone_capture"             # composite: caller declared capture_type="drone"
    COMPOSITE_CAPTURE = "composite_capture"     # composite: capture_type not declared
    UNKNOWN = "unknown"    # unrecognized single-file extension
```

- [ ] **Step 4: Add `components`/`registration` to `SourceRecord`**

In the `SourceRecord` dataclass, REPLACE:

```python
    source_id: str
    original_path: str
    source_type: SourceType
    content_hash: str
    status: SourceStatus
    asset_ids: List[str] = field(default_factory=list)
    unhandled_paths: List[str] = field(default_factory=list)
    error: Optional[str] = None
```

with:

```python
    source_id: str
    original_path: str
    source_type: SourceType
    content_hash: str
    status: SourceStatus
    asset_ids: List[str] = field(default_factory=list)
    unhandled_paths: List[str] = field(default_factory=list)
    error: Optional[str] = None
    components: Dict[str, List[str]] = field(default_factory=dict)
    registration: dict = field(default_factory=lambda: {"status": "unknown", "transform": None})
```

REPLACE `SourceRecord.to_dict`:

```python
    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "original_path": self.original_path,
            "source_type": self.source_type.value,
            "content_hash": self.content_hash,
            "status": self.status.value,
            "asset_ids": list(self.asset_ids),
            "unhandled_paths": list(self.unhandled_paths),
            "error": self.error,
        }
```

with:

```python
    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "original_path": self.original_path,
            "source_type": self.source_type.value,
            "content_hash": self.content_hash,
            "status": self.status.value,
            "asset_ids": list(self.asset_ids),
            "unhandled_paths": list(self.unhandled_paths),
            "error": self.error,
            "components": {k: list(v) for k, v in self.components.items()},
            "registration": dict(self.registration),
        }
```

REPLACE `SourceRecord.from_dict`:

```python
    @staticmethod
    def from_dict(data: dict) -> "SourceRecord":
        return SourceRecord(
            source_id=data["source_id"],
            original_path=data["original_path"],
            source_type=SourceType(data["source_type"]),
            content_hash=data["content_hash"],
            status=SourceStatus(data["status"]),
            asset_ids=list(data.get("asset_ids", [])),
            unhandled_paths=list(data.get("unhandled_paths", [])),
            error=data.get("error"),
        )
```

with:

```python
    @staticmethod
    def from_dict(data: dict) -> "SourceRecord":
        return SourceRecord(
            source_id=data["source_id"],
            original_path=data["original_path"],
            source_type=SourceType(data["source_type"]),
            content_hash=data["content_hash"],
            status=SourceStatus(data["status"]),
            asset_ids=list(data.get("asset_ids", [])),
            unhandled_paths=list(data.get("unhandled_paths", [])),
            error=data.get("error"),
            components={k: list(v) for k, v in data.get("components", {}).items()},
            registration=dict(data.get("registration", {"status": "unknown", "transform": None})),
        )
```

Add `"CaptureComponent"` to the module's `__all__` list (currently `["SourceStatus", "SourceType", "SourceRecord", "MultiSourceSession"]`) — insert it after `"SourceType"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_source_session.py -v`
Expected: ALL PASS (the full file, not just the new classes — this step must not regress any pre-existing test).

- [ ] **Step 6: Commit**

```bash
git add evidence/multi_source.py tests/test_multi_source_session.py
git commit -m "feat: add composite-capture taxonomy and detection to multi_source"
```

---

### Task 2: Ingest composite sources with per-component provenance

**Files:**
- Modify: `evidence/multi_source.py` (`add_source` signature + body; new private helper `_ingest_composite`)
- Test: `tests/test_multi_source_session.py` (new test class `TestCompositeIngestion`)

**Interfaces:**
- Consumes from Task 1: `CaptureComponent`, `_detect_composite_components`, `_is_composite_capture`, `_VISUAL_COMPONENTS`, `_SIDECAR_COMPONENTS`, `_source_type_of(path, capture_type=None)`, `SourceRecord.components`/`.registration`.
- Consumes from existing code (unchanged): `evidence.importers.import_folder`, `evidence.packages.EvidencePackage.record_processing`, `evidence.session.ProcessingRecord` (import `ProcessingRecord` from `evidence.session`, NOT `evidence.packages` — it is re-exported there but defined in `evidence.session`).
- Produces for Task 3/4: `MultiSourceSession.add_source(path, *, source=None, frame_strategy=None, capture_type=None)` — ONE new keyword-only parameter `capture_type: Optional[str] = None`, accepted values `"phone"` / `"drone"` / `None`; passed straight through to `_source_type_of`. Non-composite call sites (all existing tests) are unaffected since the parameter defaults to `None` and plain folders/files behave exactly as before. New private helper: `MultiSourceSession._ingest_composite(builder, path, use_source, frame_strategy) -> Tuple[FolderImportReport, Dict[str, List[str]], Dict[str, List[str]]]` returning `(merged_report, components, asset_ids_by_component)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_source_session.py` (after `TestSourceRecordSerializationDefaults`, before `TestEvidenceSummary`):

```python
class TestCompositeIngestion:
    def _phone_capture_folder(self, tmp_path):
        folder = tmp_path / "phone_capture_001"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "0001.jpg").write_bytes(_jpeg(0x01))
        (folder / "rgb" / "0002.jpg").write_bytes(_jpeg(0x02))
        (folder / "gps").mkdir()
        (folder / "gps" / "track.csv").write_text("lat,lon\n1.0,2.0\n")
        (folder / "imu").mkdir()
        (folder / "imu" / "log.csv").write_text("t,ax,ay,az\n0,0,0,9.8\n")
        return folder

    def test_composite_source_type_and_status(self, tmp_path):
        from evidence.multi_source import SourceStatus, SourceType

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(folder), capture_type="phone")

        assert record.source_type is SourceType.PHONE_CAPTURE
        assert record.status is SourceStatus.INGESTED

    def test_visual_component_files_become_real_evidence(self, tmp_path):
        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(folder), capture_type="phone")

        # 2 real JPEGs in rgb/ -> 2 real PHOTO evidence assets, same as
        # any other folder ingest -- composite detection does not change
        # HOW visual files are ingested, only how they're tagged after.
        assert len(record.asset_ids) == 2
        assert len(session.package.all_assets()) == 2

    def test_visual_assets_are_tagged_with_their_component(self, tmp_path):
        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(folder), capture_type="phone")

        for asset_id in record.asset_ids:
            asset = session.package.assets[asset_id]
            component_tags = [
                p for p in asset.processing_history
                if p.operation == "composite_component_tag"
            ]
            assert len(component_tags) == 1
            assert component_tags[0].detail["component"] == "rgb"
            assert component_tags[0].detail["composite_source_id"] == record.source_id

    def test_sidecar_components_are_recorded_not_fabricated_as_evidence(self, tmp_path):
        from evidence.multi_source import CaptureComponent

        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(folder), capture_type="phone")

        # gps/ and imu/ are real files on disk but NOT parseable by any
        # importer in this repo -- they must be recorded as a manifest
        # (path present), never turned into fake GPS/IMU evidence assets.
        assert record.components[CaptureComponent.GPS.value] == ["gps/track.csv"]
        assert record.components[CaptureComponent.IMU.value] == ["imu/log.csv"]
        assert len(session.package.all_assets()) == 2  # only the 2 rgb photos

    def test_registration_defaults_to_unknown_on_ingest(self, tmp_path):
        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(folder), capture_type="phone")

        assert record.registration == {"status": "unknown", "transform": None}

    def test_plain_photo_folder_still_ingests_exactly_as_before(self, tmp_path):
        # Regression guard: a non-composite folder (existing behavior)
        # must be completely unaffected by this task's changes.
        from evidence.multi_source import SourceType

        folder = tmp_path / "photos"
        folder.mkdir()
        (folder / "a.jpg").write_bytes(_jpeg())
        (folder / "b.jpg").write_bytes(_jpeg(0x02))
        session = MultiSourceSession(session_id="sess1")

        record = session.add_source(str(folder))

        assert record.source_type is SourceType.DATASET
        assert len(record.asset_ids) == 2
        assert record.components == {}

    def test_composite_folder_dedupes_on_second_add(self, tmp_path):
        folder = self._phone_capture_folder(tmp_path)
        session = MultiSourceSession(session_id="sess1")

        first = session.add_source(str(folder), capture_type="phone")
        second = session.add_source(str(folder), capture_type="phone")

        assert second is first
        assert len(session.sources()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_multi_source_session.py -k TestCompositeIngestion -v`
Expected: FAIL — visual assets will not carry `composite_component_tag` processing records, `record.components` will always be `{}` (the field exists from Task 1 but `add_source` never populates it).

- [ ] **Step 3: Implement composite ingestion in `add_source`**

At the top of `evidence/multi_source.py`, add one import line right after the existing `from evidence.packages import (...)` block:

```python
from evidence.session import ProcessingRecord
```

Add this new private method to `MultiSourceSession`, immediately after `_builder_seeded_from_package`:

```python
    def _ingest_composite(
        self,
        builder: DeterministicPackageBuilder,
        path: str,
        use_source: EvidenceSource,
        frame_strategy: Optional[IFrameSelectionStrategy],
    ):
        """Ingest a detected composite capture folder.

        Visual components (rgb/video subfolders) go through the SAME
        import_folder() every other folder source uses. Sidecar
        components (depth/imu/gps/calibration/telemetry) have no real
        parser in this repo, so their files are recorded as a manifest
        (relative path only, real files on disk) -- never turned into
        fabricated evidence assets.

        Returns (merged_report, components, asset_ids_by_component) --
        the third value lets the caller tag each visual asset with its
        component AFTER the package is built (record_processing needs a
        built EvidencePackage, which does not exist yet at this point).
        """
        detected = _detect_composite_components(path)
        merged_report = FolderImportReport()
        components: Dict[str, List[str]] = {}
        asset_ids_by_component: Dict[str, List[str]] = {}

        for component, relative_paths in sorted(detected.items(), key=lambda kv: kv[0].value):
            components[component.value] = list(relative_paths)
            if component not in _VISUAL_COMPONENTS:
                continue
            component_dir = os.path.join(path, relative_paths[0].split("/")[0])
            before_ids = {a.asset_id for a in merged_report.imported}
            sub_report = import_folder(
                builder, component_dir, source=use_source, frame_strategy=frame_strategy,
            )
            new_assets = [a for a in sub_report.imported if a.asset_id not in before_ids]
            merged_report.imported.extend(new_assets)
            merged_report.duplicates_skipped.extend(sub_report.duplicates_skipped)
            merged_report.near_duplicates_marked.extend(sub_report.near_duplicates_marked)
            merged_report.unhandled_paths.extend(sub_report.unhandled_paths)
            asset_ids_by_component[component.value] = [a.asset_id for a in new_assets]

        return merged_report, components, asset_ids_by_component
```

Now REPLACE the entire `add_source` method:

```python
    def add_source(
        self,
        path: str,
        *,
        source: Optional[EvidenceSource] = None,
        frame_strategy: Optional[IFrameSelectionStrategy] = None,
    ) -> SourceRecord:
        """Ingest one file or directory into the session.

        Returns the SourceRecord for this path -- always, even on
        failure/unsupported. A source whose content hash already exists
        in this session is a no-op (ALREADY_INGESTED, existing record
        returned unchanged); the package is only touched on genuinely
        new content.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        content_hash = _content_hash_of_path(path)
        existing = self.source_for_content(content_hash)
        if existing is not None:
            return existing

        source_id = f"src-{len(self._source_order):04d}-{content_hash[:12]}"
        source_type = _source_type_of(path)
        use_source = source or EvidenceSource(
            source_id=f"disk:{os.path.basename(os.path.normpath(path))}",
            platform="filesystem",
            device="local disk",
        )

        record: SourceRecord
        builder = self._builder_seeded_from_package()
        builder.register_source(use_source)
        try:
            if os.path.isdir(path):
                report = import_folder(builder, path, source=use_source, frame_strategy=frame_strategy)
            else:
                report = FolderImportReport()
                if os.path.splitext(path)[1].lower() not in _KNOWN_EXTS:
                    raise ValueError(f"unknown evidence extension for {path!r}")
                import_file(builder, path, source=use_source, report=report, frame_strategy=frame_strategy)
        except ValueError as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.UNSUPPORTED, error=str(exc),
            )
        except (CorruptEvidenceError, ImporterCapabilityError) as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.FAILED, error=str(exc),
            )
        else:
            self._package = builder.build(package_id="")
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.INGESTED,
                asset_ids=[a.asset_id for a in report.imported],
                unhandled_paths=list(report.unhandled_paths),
            )

        self._sources[source_id] = record
        self._source_order.append(source_id)
        return record
```

with:

```python
    def add_source(
        self,
        path: str,
        *,
        source: Optional[EvidenceSource] = None,
        frame_strategy: Optional[IFrameSelectionStrategy] = None,
        capture_type: Optional[str] = None,
    ) -> SourceRecord:
        """Ingest one file or directory into the session.

        Returns the SourceRecord for this path -- always, even on
        failure/unsupported. A source whose content hash already exists
        in this session is a no-op (ALREADY_INGESTED, existing record
        returned unchanged); the package is only touched on genuinely
        new content.

        `capture_type` ("phone" | "drone" | None) only affects a
        directory that `_detect_composite_components` recognizes as a
        composite acquisition (>=1 visual + >=1 sidecar component
        subfolder); it is ignored for plain files and non-composite
        folders, which ingest exactly as before this parameter existed.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        content_hash = _content_hash_of_path(path)
        existing = self.source_for_content(content_hash)
        if existing is not None:
            return existing

        source_id = f"src-{len(self._source_order):04d}-{content_hash[:12]}"
        source_type = _source_type_of(path, capture_type=capture_type)
        use_source = source or EvidenceSource(
            source_id=f"disk:{os.path.basename(os.path.normpath(path))}",
            platform="filesystem",
            device="local disk",
        )

        is_composite = source_type in (
            SourceType.PHONE_CAPTURE, SourceType.DRONE_CAPTURE, SourceType.COMPOSITE_CAPTURE,
        )
        components: Dict[str, List[str]] = {}

        record: SourceRecord
        builder = self._builder_seeded_from_package()
        builder.register_source(use_source)
        asset_ids_by_component: Dict[str, List[str]] = {}
        try:
            if is_composite:
                report, components, asset_ids_by_component = self._ingest_composite(
                    builder, path, use_source, frame_strategy,
                )
            elif os.path.isdir(path):
                report = import_folder(builder, path, source=use_source, frame_strategy=frame_strategy)
            else:
                report = FolderImportReport()
                if os.path.splitext(path)[1].lower() not in _KNOWN_EXTS:
                    raise ValueError(f"unknown evidence extension for {path!r}")
                import_file(builder, path, source=use_source, report=report, frame_strategy=frame_strategy)
        except ValueError as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.UNSUPPORTED, error=str(exc),
            )
        except (CorruptEvidenceError, ImporterCapabilityError) as exc:
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.FAILED, error=str(exc),
            )
        else:
            self._package = builder.build(package_id="")
            for component_value, asset_ids in asset_ids_by_component.items():
                for asset_id in asset_ids:
                    self._package.record_processing(
                        asset_id,
                        ProcessingRecord(
                            operation="composite_component_tag",
                            detail={"component": component_value, "composite_source_id": source_id},
                        ),
                    )
            record = SourceRecord(
                source_id=source_id, original_path=path, source_type=source_type,
                content_hash=content_hash, status=SourceStatus.INGESTED,
                asset_ids=[a.asset_id for a in report.imported],
                unhandled_paths=list(report.unhandled_paths),
                components=components,
            )

        self._sources[source_id] = record
        self._source_order.append(source_id)
        return record
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_source_session.py -v`
Expected: ALL PASS, including every pre-existing test (this step is also the regression check for Task 1's and the original module's tests).

- [ ] **Step 5: Commit**

```bash
git add evidence/multi_source.py tests/test_multi_source_session.py
git commit -m "feat: ingest composite capture sources with per-component provenance"
```

---

### Task 3: Explicit cross-source frame registration

**Files:**
- Modify: `evidence/multi_source.py` (imports, new exception, new method)
- Test: `tests/test_multi_source_session.py` (new test class `TestSourceFrameRegistration`)

**Interfaces:**
- Consumes: `world_ir.coordinates.Transform`, `world_ir.coordinates.Frame` (existing, unmodified — `Transform.to_dict()`/`Transform.from_dict()` already exist per `world_ir/coordinates.py:118-136`).
- Produces: `UnknownSourceError(ValueError)` (new exception class); `MultiSourceSession.register_source_frame(source_id: str, transform: Transform) -> SourceRecord` — sets `record.registration = {"status": "registered", "transform": transform.to_dict()}` and returns the (same, mutated) `SourceRecord`; raises `UnknownSourceError` for an unrecognized `source_id`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multi_source_session.py` (after `TestCompositeIngestion`, before `TestEvidenceSummary`):

```python
class TestSourceFrameRegistration:
    def test_register_source_frame_sets_registered_status(self, tmp_path):
        from world_ir.coordinates import Frame, IDENTITY_MATRIX, Transform

        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(photo))
        assert record.registration == {"status": "unknown", "transform": None}

        transform = Transform(
            source_frame=Frame.SENSOR, target_frame=Frame.SESSION_LOCAL, matrix=IDENTITY_MATRIX,
        )
        updated = session.register_source_frame(record.source_id, transform)

        assert updated.registration["status"] == "registered"
        assert updated.registration["transform"]["source_frame"] == "sensor"
        assert updated.registration["transform"]["target_frame"] == "session-local"
        # The same object living in session.sources() reflects the change
        # (SourceRecord is a plain mutable dataclass; registration is the
        # one field that legitimately changes after creation).
        assert session.sources()[0].registration["status"] == "registered"

    def test_register_source_frame_unknown_source_raises(self):
        from evidence.multi_source import UnknownSourceError
        from world_ir.coordinates import Frame, Transform

        session = MultiSourceSession(session_id="sess1")
        transform = Transform(source_frame=Frame.SENSOR, target_frame=Frame.SESSION_LOCAL)

        import pytest
        with pytest.raises(UnknownSourceError):
            session.register_source_frame("does-not-exist", transform)

    def test_registration_round_trips_through_serialization(self, tmp_path):
        from world_ir.coordinates import Frame, IDENTITY_MATRIX, Transform

        photo = tmp_path / "a.jpg"
        photo.write_bytes(_jpeg())
        session = MultiSourceSession(session_id="sess1")
        record = session.add_source(str(photo))
        transform = Transform(source_frame=Frame.SENSOR, target_frame=Frame.SESSION_LOCAL, matrix=IDENTITY_MATRIX)
        session.register_source_frame(record.source_id, transform)

        restored = MultiSourceSession.from_dict(session.to_dict())

        assert restored.sources()[0].registration["status"] == "registered"
        assert restored.sources()[0].registration["transform"]["source_frame"] == "sensor"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_multi_source_session.py -k TestSourceFrameRegistration -v`
Expected: FAIL — `ImportError: cannot import name 'UnknownSourceError'` / `AttributeError: 'MultiSourceSession' object has no attribute 'register_source_frame'`.

- [ ] **Step 3: Implement `register_source_frame`**

In `evidence/multi_source.py`, add this new exception class right after the module's `__all__` list (before `class SourceStatus(str, Enum):`):

```python
class UnknownSourceError(ValueError):
    """Raised when an operation names a source_id that isn't in this session."""
```

Add `"UnknownSourceError"` to `__all__` (after `"SourceRecord"`).

Add this method to `MultiSourceSession`, immediately after `source_for_content`:

```python
    def register_source_frame(self, source_id: str, transform) -> SourceRecord:
        """Explicitly record how one source's own coordinate frame
        relates to the session frame. NEVER called automatically by
        add_source -- alignment across independent sources must not be
        assumed (spec §35/§65: a phone source and a drone source start
        with independent coordinate frames; only an explicit caller
        establishes a real transform between them).

        `transform` is a world_ir.coordinates.Transform; its
        source_frame/target_frame/matrix/uncertainty are stored verbatim
        via its existing to_dict().
        """
        if source_id not in self._sources:
            raise UnknownSourceError(f"no source {source_id!r} in session {self.session_id!r}")
        record = self._sources[source_id]
        record.registration = {"status": "registered", "transform": transform.to_dict()}
        return record
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_source_session.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add evidence/multi_source.py tests/test_multi_source_session.py
git commit -m "feat: add explicit cross-source frame registration"
```

---

### Task 4: CLI exposure + end-to-end test

**Files:**
- Modify: `apps/cli/main.py` (`cmd_source_add` gains `--capture-type` flag; `cmd_source_inspect` prints components/registration; `build_parser`'s `p_source_add` registration)
- Test: `tests/test_cli.py` (new test class `TestCompositeCaptureCLI`)
- Test: `tests/test_multi_source_session.py` (new end-to-end test class `TestCompositeEndToEnd`)

**Interfaces:**
- Consumes from Tasks 1-3: `MultiSourceSession.add_source(..., capture_type=...)`, `SourceRecord.components`, `SourceRecord.registration`, `MultiSourceSession.register_source_frame`.
- Produces: no new public API — this task only wires existing CLI commands to the new session-layer capability and adds the closing end-to-end test the plan's Goal requires.

- [ ] **Step 1: Write the failing CLI test**

Read `tests/test_cli.py` lines 1-70 first to match its exact fixture/import style (it uses `from apps.cli.main import main, _artifacts_dir_for` and calls `main([...])` with `capsys` to capture stdout). Append this class to the end of `tests/test_cli.py`:

```python
class TestCompositeCaptureCLI:
    def _jpeg(self, seed_byte: int = 0x01) -> bytes:
        return b"\xff\xd8\xff\xe0" + bytes([seed_byte]) * 64

    def _phone_capture_folder(self, tmp_path):
        folder = tmp_path / "phone_capture_001"
        (folder / "rgb").mkdir(parents=True)
        (folder / "rgb" / "0001.jpg").write_bytes(self._jpeg(0x01))
        (folder / "rgb" / "0002.jpg").write_bytes(self._jpeg(0x02))
        (folder / "gps").mkdir()
        (folder / "gps" / "track.csv").write_text("lat,lon\n1.0,2.0\n")
        return folder

    def test_add_source_with_capture_type_flag(self, tmp_path, capsys):
        session_dir = tmp_path / "session"
        folder = self._phone_capture_folder(tmp_path)

        assert main(["session", "create", "Building_A", "-o", str(session_dir)]) == 0
        capsys.readouterr()
        exit_code = main([
            "session", "add-source", str(session_dir), str(folder), "--capture-type", "phone",
        ])
        out = capsys.readouterr().out

        assert exit_code == 0
        assert "ingested" in out
        assert "phone_capture" in out

    def test_inspect_source_shows_components_and_registration(self, tmp_path, capsys):
        session_dir = tmp_path / "session"
        folder = self._phone_capture_folder(tmp_path)
        main(["session", "create", "Building_A", "-o", str(session_dir)])
        capsys.readouterr()
        main(["session", "add-source", str(session_dir), str(folder), "--capture-type", "phone"])
        capsys.readouterr()

        # Discover the source id the same way a user would: via `session list`.
        main(["session", "list", str(session_dir)])
        list_out = capsys.readouterr().out
        source_id = list_out.splitlines()[0].split("\t")[0]

        exit_code = main(["session", "inspect-source", str(session_dir), source_id])
        out = capsys.readouterr().out

        assert exit_code == 0
        assert "gps" in out
        assert "registration: unknown" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -k TestCompositeCaptureCLI -v`
Expected: FAIL — `argparse` rejects the unrecognized `--capture-type` argument (`error: unrecognized arguments: --capture-type phone`), and `cmd_source_inspect`'s current output has no "registration:" line.

- [ ] **Step 3: Wire `--capture-type` and component/registration display into the CLI**

Read `apps/cli/main.py` in full before editing (it currently has `cmd_source_add`, `cmd_source_inspect`, and their `add_parser` registrations — locate by function name, not assumed line numbers, since earlier session work may have shifted them).

In `cmd_source_add`, REPLACE:

```python
def cmd_source_add(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    record = session.add_source(args.path)
    _save_session(session, args.session_dir)
    print(f"{record.status.value}\t{record.source_id}\t{record.source_type.value}\t{args.path}")
    if record.error:
        _eprint(f"  {record.error}")
    if record.unhandled_paths:
        for unhandled in record.unhandled_paths:
            _eprint(f"  unhandled: {unhandled}")
    return 0 if record.status.value in ("ingested", "already_ingested") else 1
```

with:

```python
def cmd_source_add(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    record = session.add_source(args.path, capture_type=args.capture_type)
    _save_session(session, args.session_dir)
    print(f"{record.status.value}\t{record.source_id}\t{record.source_type.value}\t{args.path}")
    if record.error:
        _eprint(f"  {record.error}")
    if record.unhandled_paths:
        for unhandled in record.unhandled_paths:
            _eprint(f"  unhandled: {unhandled}")
    return 0 if record.status.value in ("ingested", "already_ingested") else 1
```

In `cmd_source_inspect`, REPLACE:

```python
def cmd_source_inspect(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    matches = [s for s in session.sources() if s.source_id == args.source_id]
    if not matches:
        _eprint(f"unknown source id: {args.source_id!r}")
        return 1
    record = matches[0]
    print(f"source '{record.source_id}'")
    print(f"  path: {record.original_path}")
    print(f"  type: {record.source_type.value}")
    print(f"  status: {record.status.value}")
    print(f"  content hash: {record.content_hash}")
    print(f"  asset(s): {len(record.asset_ids)}")
    for asset_id in record.asset_ids:
        print(f"    {asset_id}")
    if record.unhandled_paths:
        print(f"  unhandled path(s): {len(record.unhandled_paths)}")
        for path in record.unhandled_paths:
            print(f"    {path}")
    if record.error:
        print(f"  error: {record.error}")
    return 0
```

with:

```python
def cmd_source_inspect(args: argparse.Namespace) -> int:
    session = _load_session(args.session_dir)
    matches = [s for s in session.sources() if s.source_id == args.source_id]
    if not matches:
        _eprint(f"unknown source id: {args.source_id!r}")
        return 1
    record = matches[0]
    print(f"source '{record.source_id}'")
    print(f"  path: {record.original_path}")
    print(f"  type: {record.source_type.value}")
    print(f"  status: {record.status.value}")
    print(f"  content hash: {record.content_hash}")
    print(f"  registration: {record.registration['status']}")
    print(f"  asset(s): {len(record.asset_ids)}")
    for asset_id in record.asset_ids:
        print(f"    {asset_id}")
    if record.components:
        print(f"  component(s): {len(record.components)}")
        for component, paths in sorted(record.components.items()):
            print(f"    {component}: {len(paths)} file(s)")
    if record.unhandled_paths:
        print(f"  unhandled path(s): {len(record.unhandled_paths)}")
        for path in record.unhandled_paths:
            print(f"    {path}")
    if record.error:
        print(f"  error: {record.error}")
    return 0
```

In `build_parser`, find `p_source_add = session_sub.add_parser(...)` and REPLACE:

```python
    p_source_add = session_sub.add_parser(
        "add-source", help="ingest one file/folder into an existing session (incremental; dedups by content)"
    )
    p_source_add.add_argument("session_dir")
    p_source_add.add_argument("path", help="file or folder to ingest")
    p_source_add.set_defaults(func=cmd_source_add)
```

with:

```python
    p_source_add = session_sub.add_parser(
        "add-source", help="ingest one file/folder into an existing session (incremental; dedups by content)"
    )
    p_source_add.add_argument("session_dir")
    p_source_add.add_argument("path", help="file or folder to ingest")
    p_source_add.add_argument(
        "--capture-type", choices=["phone", "drone"], default=None,
        help="declare a composite folder (rgb/video + gps/imu/depth/calibration/telemetry subfolders) "
             "as a phone or drone capture; ignored for plain files/folders",
    )
    p_source_add.set_defaults(func=cmd_source_add)
```

- [ ] **Step 4: Run the CLI test to verify it passes**

Run: `python -m pytest tests/test_cli.py -k TestCompositeCaptureCLI -v`
Expected: PASS.

- [ ] **Step 5: Write and pass the end-to-end test**

Append to `tests/test_multi_source_session.py` (at the end of the file):

```python
class TestCompositeEndToEnd:
    def test_full_composite_workflow(self, tmp_path):
        """create session -> add a plain photo source -> add a composite
        phone capture (rgb + gps + imu) -> verify both sources coexist,
        visual evidence is real and tagged, sidecar files are preserved
        as a manifest (not fabricated evidence), registration starts
        unknown and can be explicitly set -- then the whole session
        round-trips through serialization with everything intact."""
        from evidence.multi_source import CaptureComponent, SourceType
        from world_ir.coordinates import Frame, IDENTITY_MATRIX, Transform

        session = MultiSourceSession(session_id="Building_A", name="Building A")

        plain_photo = tmp_path / "survey.jpg"
        plain_photo.write_bytes(_jpeg(0x09))
        plain_record = session.add_source(str(plain_photo))
        assert plain_record.source_type is SourceType.IMAGE

        phone_folder = tmp_path / "phone_capture_001"
        (phone_folder / "rgb").mkdir(parents=True)
        (phone_folder / "rgb" / "0001.jpg").write_bytes(_jpeg(0x01))
        (phone_folder / "rgb" / "0002.jpg").write_bytes(_jpeg(0x02))
        (phone_folder / "gps").mkdir()
        (phone_folder / "gps" / "track.csv").write_text("lat,lon\n1.0,2.0\n")
        (phone_folder / "imu").mkdir()
        (phone_folder / "imu" / "log.csv").write_text("t,ax,ay,az\n0,0,0,9.8\n")

        phone_record = session.add_source(str(phone_folder), capture_type="phone")

        # Both sources coexist; neither was destroyed or merged.
        assert len(session.sources()) == 2
        assert phone_record.source_type is SourceType.PHONE_CAPTURE

        # Visual evidence is real (3 total photos: 1 survey + 2 rgb).
        assert len(session.package.all_assets()) == 3

        # Sidecar files preserved as a manifest, not fabricated evidence.
        assert phone_record.components[CaptureComponent.GPS.value] == ["gps/track.csv"]
        assert phone_record.components[CaptureComponent.IMU.value] == ["imu/log.csv"]

        # Visual assets traceable back to their component + composite source.
        for asset_id in phone_record.asset_ids:
            asset = session.package.assets[asset_id]
            tags = [p for p in asset.processing_history if p.operation == "composite_component_tag"]
            assert tags[0].detail["composite_source_id"] == phone_record.source_id

        # Registration starts unknown -- no fabricated alignment.
        assert phone_record.registration == {"status": "unknown", "transform": None}
        assert plain_record.registration == {"status": "unknown", "transform": None}

        # Explicit registration is possible and persists.
        transform = Transform(source_frame=Frame.SENSOR, target_frame=Frame.SESSION_LOCAL, matrix=IDENTITY_MATRIX)
        session.register_source_frame(phone_record.source_id, transform)

        # Full round trip preserves everything: both sources, components,
        # provenance tags, and the one registered transform.
        restored = MultiSourceSession.from_dict(session.to_dict())
        restored_phone = [s for s in restored.sources() if s.source_id == phone_record.source_id][0]
        restored_plain = [s for s in restored.sources() if s.source_id == plain_record.source_id][0]

        assert len(restored.sources()) == 2
        assert len(restored.package.all_assets()) == 3
        assert restored_phone.components[CaptureComponent.GPS.value] == ["gps/track.csv"]
        assert restored_phone.registration["status"] == "registered"
        assert restored_plain.registration == {"status": "unknown", "transform": None}

        # Downstream evidence summary still works unchanged (reuses the
        # real orchestrator gate -- 3 photos clears MIN_IMAGE_EVIDENCE=2).
        summary = restored.evidence_summary()
        assert summary["ready_for_reconstruction"] is True
```

Run: `python -m pytest tests/test_multi_source_session.py tests/test_cli.py -v`
Expected: ALL PASS (every test in both files, old and new).

- [ ] **Step 6: Run the full regression suite**

Run: `python -m pytest tests/ -q`
Expected: same pass count as the pre-plan baseline plus this plan's new tests, with no new failures. (The pre-existing `tests/test_sam_backend.py::TestSAMSegmentationBackendIntegration::test_real_model_load_and_inference` failure, caused by a missing cached `torch.hub` model unrelated to this plan, is expected to remain and is NOT this plan's responsibility to fix.) If a `MagicMock/` directory appears at the repo root afterward, it is that same unrelated SAM test's `torch.hub` mock leaking a cache-dir artifact — delete it (`rm -rf MagicMock`) as normal repo hygiene; it is not tracked by git.

- [ ] **Step 7: Commit**

```bash
git add apps/cli/main.py tests/test_cli.py tests/test_multi_source_session.py
git commit -m "feat: expose composite captures via CLI and add end-to-end coverage"
```
