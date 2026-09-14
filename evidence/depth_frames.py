"""DEPTH sidecar ingestion: 16-bit PNG depth frames (spec sec 5-8,
backlog P1.4; Decision 020 in docs/engineering/DESIGN_DECISIONS.md).

Closes the gap `evidence/sensors.py` deliberately left open: a
composite capture's ``depth/`` directory was detected and recorded as
a file manifest, never parsed. This module parses those files into
real, unit-explicit ``DepthFrame`` records a downstream consumer
(unprojection/fusion) can use without re-implementing device-format
logic.

FORMAT LADDER (Decision 020): 16-bit PNG first -- lossless integer,
inspectable, Pillow-native. EXR (float) and device-specific raw
formats are NOT parsed here; they need their own explicit adapters.

THE SCALE RULE (the reason this module exists):

    depth_meters = raw_value * depth_scale

A 16-bit PNG carries bare integers. TiffTag/PNG has no standard field
declaring what those integers mean, and devices disagree wildly
(millimeters at scale 0.001, 1/1000 m, arbitrary fixed-point...).
Raw values are therefore NEVER interpreted as meters: ``depth_scale``
is REQUIRED, and it must come from one of exactly two explicit
sources --

1. the caller passes ``depth_scale=`` directly, or
2. a ``depth/manifest.json`` sidecar (schema below) supplies it
   (per-file override > global default).

There is no default scale anywhere in this module. Neither source
present -> ``DepthScaleUnavailable`` naming the files, never a guess.

``depth/manifest.json`` schema (one JSON object; all keys optional
except that at least one scale source must resolve)::

    {
        "depth_scale": 0.001,       // meters per raw unit, > 0
        "invalid_value": 0,         // raw value meaning "no depth"
        "units": "meter",           // "meter" | "millimeter" only
        "per_file": {               // optional per-file overrides
            "frame_0001.png": {"depth_scale": 0.0005}
        }
    }

``units`` is whitelisted to "meter"|"millimeter" because inventing
conversions is how silent unit lies happen; a device reporting
anything else needs an explicit adapter, not a guessed factor.

Invalid pixels (raw == invalid_value) are MASKED -- counted and
excluded from meter iteration -- never zero-filled into geometry or
averaged over.

Honest failure: a malformed file (not a PNG, 8-bit gray, RGB color
image, truncated bytes) raises ``DepthFrameError`` naming the file and
reason; a missing scale raises ``DepthScaleUnavailable``. Nothing is
silently skipped or coerced. An unreadable file raises OSError from
the open() call, not swallowed.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from provenance import Provenance, Uncertainty
from evidence.sensors import SensorParseError  # base class for DepthFrameError

#: PNG modes this module accepts. "I;16" (native-endian 16-bit), its
#: explicit-endian variants, and PIL's "I" (32-bit signed int, which
#: some tools emit for float depth rounded to int). Anything else --
#: "L" (8-bit gray: almost certainly a colorized preview, not depth),
#: "RGB"/"RGBA" (color image) -- is rejected loudly.
_ACCEPTED_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N", "I"})
_MODE_DTYPE = {
    "I;16": "uint16",
    "I;16L": "uint16",
    "I;16B": "uint16",
    "I;16N": "uint16",
    "I": "int32",
}

_ALLOWED_UNITS = frozenset({"meter", "millimeter"})
_MANIFEST_NAME = "manifest.json"


class DepthFrameError(SensorParseError):
    """A depth sidecar file violated the documented format.

    Subclasses SensorParseError so callers already catching that for
    IMU/GNSS/telemetry parsing catch depth parsing too; OSError still
    means "file missing/unreadable".
    """


class DepthScaleUnavailable(ValueError):
    """No explicit depth_scale was provided and no depth/manifest.json
    resolved one. This is the deliberate failure of THIS module, not a
    data bug: raw 16-bit values cannot be honestly interpreted without
    a scale, so refusing is the correct behavior.
    """


@dataclass(frozen=True)
class DepthFrame:
    """One parsed depth image: raw values + the explicit scale that
    gives them meaning. Immutable; ``raw`` is never mutated post-load.

    ``raw`` is a tuple of rows of ints (row-major). Plain tuples keep
    the evidence layer dependency-free and structurally immutable --
    the same tradeoff ``perception/depth/interface.DepthMap`` documents
    for its List[List[float]] values. Consumers needing a dense array
    convert (their choice of numpy) at their own boundary.
    """

    frame_id: str
    source_path: str
    width: int
    height: int
    raw: Tuple[Tuple[int, ...], ...]
    depth_dtype: str
    depth_scale: float
    invalid_value: int
    units: str
    provenance: Provenance = Provenance.OBSERVED
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"dimensions must be positive, got {self.width}x{self.height}")
        if not math.isfinite(self.depth_scale) or self.depth_scale <= 0.0:
            raise ValueError(
                f"depth_scale must be finite and > 0, got {self.depth_scale!r}"
                " -- raw integers are never meters without an explicit scale"
            )
        if self.units not in _ALLOWED_UNITS:
            raise ValueError(
                f"units {self.units!r} not in {_ALLOWED_UNITS}"
                " -- unknown units need an explicit adapter, not a guessed conversion"
            )
        if len(self.raw) != self.height:
            raise ValueError(f"raw has {len(self.raw)} rows, expected height={self.height}")
        if self.raw and len(self.raw[0]) != self.width:
            raise ValueError(f"raw row length {len(self.raw[0])}, expected width={self.width}")

    @property
    def invalid_pixel_count(self) -> int:
        """Pixels whose raw value equals invalid_value (masked, not
        zero-filled). A large fraction means the sensor saw little --
        reported, never hidden."""
        return sum(row.count(self.invalid_value) for row in self.raw)

    def meters_at(self, row: int, col: int) -> Optional[float]:
        """Depth in meters at one pixel, or None for an invalid pixel.
        Raw integers are never returned as if they were meters."""
        value = self.raw[row][col]
        if value == self.invalid_value:
            return None
        return value * self.depth_scale

    def iter_meters(self) -> Iterator[Tuple[int, int, float]]:
        """Yield (row, col, meters) for every valid pixel. Invalid
        pixels are masked out here, not left for the caller to
        remember to filter."""
        for r, row in enumerate(self.raw):
            for c, value in enumerate(row):
                if value != self.invalid_value:
                    yield (r, c, value * self.depth_scale)

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "source_path": self.source_path,
            "width": self.width,
            "height": self.height,
            "raw": [list(row) for row in self.raw],
            "depth_dtype": self.depth_dtype,
            "depth_scale": self.depth_scale,
            "invalid_value": self.invalid_value,
            "units": self.units,
            "provenance": self.provenance.value,
            "uncertainty": {"confidence": self.uncertainty.confidence, "note": self.uncertainty.note},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DepthFrame":
        return cls(
            frame_id=data["frame_id"],
            source_path=data["source_path"],
            width=data["width"],
            height=data["height"],
            raw=tuple(tuple(int(v) for v in row) for row in data["raw"]),
            depth_dtype=data["depth_dtype"],
            depth_scale=float(data["depth_scale"]),
            invalid_value=int(data["invalid_value"]),
            units=data["units"],
            provenance=Provenance(data.get("provenance", "OBSERVED")),
            uncertainty=Uncertainty(
                confidence=float(data.get("uncertainty", {}).get("confidence", 1.0)),
                note=data.get("uncertainty", {}).get("note"),
            ),
        )


@dataclass(frozen=True)
class DepthSidecarManifest:
    """Parsed depth/manifest.json: the device's declaration of what its
    raw depth values mean. All fields optional in the file; validated
    on load (scale > 0 and finite, units whitelisted, per_file entries
    well-formed) so a bad manifest fails here, not downstream."""

    depth_scale: Optional[float] = None
    invalid_value: int = 0
    units: str = "meter"
    per_file: Dict[str, float] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def load(cls, path: str) -> "DepthSidecarManifest":
        with open(path, "r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError as exc:
                raise DepthFrameError(f"{path}: invalid JSON ({exc})") from exc
        if not isinstance(data, dict):
            raise DepthFrameError(f"{path}: expected a JSON object, got {type(data).__name__}")
        scale = data.get("depth_scale")
        if scale is not None:
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0.0:
                raise DepthFrameError(f"{path}: depth_scale must be finite and > 0, got {scale!r}")
        units = data.get("units", "meter")
        if units not in _ALLOWED_UNITS:
            raise DepthFrameError(f"{path}: units {units!r} not in {sorted(_ALLOWED_UNITS)}")
        invalid = int(data.get("invalid_value", 0))
        if invalid < 0:
            raise DepthFrameError(f"{path}: invalid_value must be >= 0, got {invalid}")
        per_file_raw = data.get("per_file", {})
        if not isinstance(per_file_raw, dict):
            raise DepthFrameError(f"{path}: 'per_file' must be an object")
        per_file: Dict[str, float] = {}
        for name, entry in per_file_raw.items():
            entry_scale = float(entry["depth_scale"]) if isinstance(entry, dict) else float(entry)
            if not math.isfinite(entry_scale) or entry_scale <= 0.0:
                raise DepthFrameError(f"{path}: per_file[{name!r}] scale must be finite and > 0")
            per_file[name] = entry_scale
        return cls(
            depth_scale=scale,
            invalid_value=invalid,
            units=units,
            per_file=per_file,
            source_path=os.path.basename(path),
        )

    def resolve(self, png_name: str) -> Tuple[float, int, str]:
        """(depth_scale, invalid_value, units) for one PNG: per-file
        override > global default. Raises DepthScaleUnavailable when
        neither exists -- the caller then surfaces the file's name,
        which is exactly the information needed to fix the capture."""
        scale = self.per_file.get(png_name, self.depth_scale)
        if scale is None:
            raise DepthScaleUnavailable(
                f"no depth_scale for {png_name!r}: not in per_file overrides and"
                " manifest has no global depth_scale"
            )
        return (scale, self.invalid_value, self.units)


def parse_depth_png(
    path: str,
    depth_scale: Optional[float] = None,
    *,
    invalid_value: int = 0,
    units: str = "meter",
    manifest: Optional[DepthSidecarManifest] = None,
    frame_id: Optional[str] = None,
) -> DepthFrame:
    """Parse one 16-bit depth PNG into a DepthFrame.

    ``depth_scale`` (meters per raw unit) is REQUIRED in practice:
    pass it directly, or pass ``manifest=`` and let the manifest
    resolve it (per-file override > global default). Neither present
    raises DepthScaleUnavailable -- never a guessed scale.

    Raises DepthFrameError for any content that is not an accepted
    depth image (8-bit gray, color, truncated, wrong mode); OSError if
    the file cannot be opened.
    """
    from PIL import Image  # lazy, matching evidence/importers.py's convention

    if manifest is not None:
        resolved_scale, manifest_invalid, manifest_units = manifest.resolve(os.path.basename(path))
        depth_scale = resolved_scale
        invalid_value = manifest_invalid
        units = manifest_units
    if depth_scale is None:
        raise DepthScaleUnavailable(
            f"{path}: no depth_scale available -- pass depth_scale= explicitly or"
            f" provide a {_MANIFEST_NAME} manifest; raw PNG values are never"
            " assumed to be meters (Decision 020)"
        )
    depth_scale = float(depth_scale)
    if not math.isfinite(depth_scale) or depth_scale <= 0.0:
        raise DepthFrameError(f"{path}: depth_scale must be finite and > 0, got {depth_scale!r}")

    try:
        img = Image.open(path)
    except Image.UnidentifiedImageError as exc:
        # File exists and reads, but is not an image at all -- a
        # content problem, not an availability problem.
        raise DepthFrameError(f"{path}: not an image ({exc})") from exc
    except OSError:
        # Missing file, permission denied, ... -- stays an OSError per
        # the documented contract, distinct from format errors.
        raise
    except Exception as exc:
        raise DepthFrameError(f"{path}: PNG open failed ({type(exc).__name__}: {exc})") from exc

    try:
        with img:
            mode = img.mode
            width, height = img.size
            if mode not in _ACCEPTED_MODES:
                raise DepthFrameError(
                    f"{path}: mode {mode!r} is not a supported depth encoding"
                    f" (accepted: {sorted(_ACCEPTED_MODES)}); an 8-bit gray or"
                    " color image is almost certainly not a depth map"
                )
            pixels = list(img.getdata())
    except DepthFrameError:
        raise
    except OSError as exc:
        # Header parsed but pixel data is truncated/unreadable.
        raise DepthFrameError(f"{path}: unreadable/truncated PNG ({exc})") from exc
    except Exception as exc:  # PIL raises assorted decode errors by type
        raise DepthFrameError(f"{path}: PNG decode failed ({type(exc).__name__}: {exc})") from exc

    if width <= 0 or height <= 0:
        raise DepthFrameError(f"{path}: degenerate dimensions {width}x{height}")
    rows: List[Tuple[int, ...]] = []
    idx = 0
    for _ in range(height):
        rows.append(tuple(int(v) for v in pixels[idx : idx + width]))
        idx += width

    return DepthFrame(
        frame_id=frame_id if frame_id is not None else os.path.splitext(os.path.basename(path))[0],
        source_path=path,
        width=width,
        height=height,
        raw=tuple(rows),
        depth_dtype=_MODE_DTYPE[mode],
        depth_scale=depth_scale,
        invalid_value=invalid_value,
        units=units,
        provenance=Provenance.OBSERVED,
        uncertainty=Uncertainty(
            confidence=1.0,
            note=f"scale from {'manifest' if manifest is not None else 'explicit argument'}",
        ),
    )


def parse_depth_frames(
    paths: Sequence[str],
    depth_scale: Optional[float] = None,
    *,
    invalid_value: int = 0,
    units: str = "meter",
    manifest: Optional[DepthSidecarManifest] = None,
) -> List[DepthFrame]:
    """Parse every ``.png`` in `paths` (a source's recorded DEPTH
    component files). Non-``.png`` files are skipped, matching the
    honest-subset behavior of parse_component_streams. Scale rules are
    parse_depth_png's; with a manifest, per-file overrides apply and
    files the manifest cannot resolve raise DepthScaleUnavailable
    naming that file."""
    return [
        parse_depth_png(
            p,
            depth_scale,
            invalid_value=invalid_value,
            units=units,
            manifest=manifest,
        )
        for p in paths
        if p.lower().endswith(".png")
    ]
