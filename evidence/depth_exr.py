"""EXR depth sidecar adapter (P1-03 depth integration; Decision 020
format ladder, rung 2).

The 16-bit PNG rung (evidence/depth_frames.py) is integer-only: a
device reporting float meters -- or a LiDAR reporting +inf for
no-return pixels -- cannot be represented losslessly. EXR is the float
rung. This adapter follows the established depth-sidecar discipline:

  - EXR depth IS float meters under this adapter's contract: there are
    no raw integers to misread, so the manifest scale machinery does
    not apply. A manifest whose resolved scale is not 1.0 contradicts
    the format and raises DepthFrameError (applying it would
    double-scale real meters).
  - NaN, +inf, and negative samples are INVALID pixels: masked and
    counted, never zero-filled into geometry.
  - DepthFrame.raw is integer-valued by contract, so float meters are
    kept at millimeter precision: raw = round(m * 1000),
    depth_scale = 0.001. Sub-mm precision is not representable in the
    canonical DepthFrame contract; the quantization is documented, not
    hidden.
  - Missing OpenEXR is an availability fact: DepthFrameError naming
    the dependency, never a silent skip or a guessed encoding.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from evidence.depth_frames import (
    DepthFrame,
    DepthFrameError,
    DepthSidecarManifest,
)
from evidence.sensors import SensorDescriptor
from provenance import Provenance, Uncertainty

#: Known EXR depth channel names, in priority order.
EXR_DEPTH_CHANNELS = ("D", "Z", "depth", "Depth")

try:  # pragma: no cover - import presence is environment-dependent
    import Imath as _Imath
    import OpenEXR as _OpenEXR

    _OpenEXRInputFile = _OpenEXR.InputFile
except Exception:  # noqa: BLE001 - any import failure means unavailable
    _Imath = None
    _OpenEXR = None
    _OpenEXRInputFile = None



def parse_depth_exr(
    path: str,
    *,
    manifest: Optional[DepthSidecarManifest] = None,
    frame_id: Optional[str] = None,
    identity: Optional[SensorDescriptor] = None,
) -> DepthFrame:
    """Parse one single-channel float EXR depth frame (meters).

    Raises DepthFrameError when OpenEXR is unavailable, the file is
    not a readable EXR, or no known depth channel exists; OSError
    propagates for missing/unreadable files.
    """
    if _OpenEXRInputFile is None:
        raise DepthFrameError(
            f"{path}: the OpenEXR dependency is not installed -- EXR depth"
            " parsing is unavailable (pip install OpenEXR); refusing to"
            " guess a depth encoding"
        )
    if manifest is not None:
        resolved_scale, _manifest_invalid, manifest_units = manifest.resolve(
            os.path.basename(path)
        )
        if resolved_scale != 1.0:
            raise DepthFrameError(
                f"{path}: manifest depth_scale {resolved_scale!r} contradicts"
                " EXR float meters (samples are already meters; a scale"
                " other than 1.0 would double-apply)"
            )
        if manifest_units != "meter":
            raise DepthFrameError(
                f"{path}: manifest units {manifest_units!r} contradict EXR"
                " float meters"
            )

    try:
        infile = _OpenEXRInputFile(str(path))
    except Exception as exc:  # noqa: BLE001 - OpenEXR raises bare Exception
        raise DepthFrameError(f"{path}: EXR open failed ({exc})") from exc

    try:
        header = infile.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = header.get("channels", {})
        if width <= 0 or height <= 0:
            raise DepthFrameError(
                f"{path}: degenerate dimensions {width}x{height}"
            )
        channel_name = None
        for candidate in EXR_DEPTH_CHANNELS:
            if candidate in channels:
                channel_name = candidate
                break
        if channel_name is None:
            raise DepthFrameError(
                f"{path}: no depth channel found (have {sorted(channels)};"
                f" looked for {list(EXR_DEPTH_CHANNELS)})"
            )
        pixel_type = _Imath.PixelType(_Imath.PixelType.FLOAT)
        raw_channel = infile.channel(channel_name, pixel_type)
    except DepthFrameError:
        raise
    except OSError:
        raise
    except Exception as exc:  # noqa: BLE001 - OpenEXR decode failures
        raise DepthFrameError(
            f"{path}: EXR decode failed ({type(exc).__name__}: {exc})"
        ) from exc
    finally:
        infile.close()

    meters = np.frombuffer(raw_channel, dtype=np.float32).reshape(
        height, width
    )
    invalid = 0
    rows: List[Tuple[int, ...]] = []
    for r in range(height):
        row = meters[r]
        int_row = []
        for c in range(width):
            v = float(row[c])
            if math.isnan(v) or math.isinf(v) or v < 0.0:
                int_row.append(invalid)
            else:
                int_row.append(int(round(v * 1000.0)))
        rows.append(tuple(int_row))

    return DepthFrame(
        frame_id=(
            frame_id
            if frame_id is not None
            else os.path.splitext(os.path.basename(path))[0]
        ),
        source_path=path,
        width=width,
        height=height,
        raw=tuple(rows),
        depth_dtype="float32",
        depth_scale=0.001,
        invalid_value=invalid,
        units="meter",
        provenance=Provenance.OBSERVED,
        uncertainty=Uncertainty(
            confidence=1.0,
            note="EXR float meters; quantized to DepthFrame mm precision",
        ),
        identity=identity,
    )
