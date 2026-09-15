"""Appearance features for multi-view object identity (P7-01): a real,
deterministic color-histogram descriptor over a detected region's own
pixels, and a similarity score between two descriptors.

This exists because `object_resolution.merge_hypotheses` previously
merged same-label, spatially-close hypotheses on label + 3D proximity
alone -- per `docs/future/perception/MULTI_VIEW_IDENTITY.md`, "same
label + proximity is not sufficient identity" (symmetric scenes, e.g.
four identical chairs, need a discriminating signal geometry alone
can't give). This is that signal's cheap half; `epipolar.py` is the
geometric half.

Deliberately NOT a learned embedding (no CLIP/DINOv2, no new ML
dependency) -- a normalized RGB color histogram over the region's own
mask, same "simple, real, deterministic feature" the P7-01 task scope
calls for. It will not discriminate two same-color same-shape objects;
that is an honest, documented limitation, not a hidden one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: 4 bins/channel = 64 total bins -- enough to separate "red chair" from
#: "blue chair" without overfitting to lighting noise on 8-bit color.
DEFAULT_BINS_PER_CHANNEL = 4


@dataclass(frozen=True)
class AppearanceDescriptor:
    """Normalized RGB color histogram (sums to 1.0) over one region's
    masked pixels. `pixel_count` is the number of pixels the histogram
    was built from -- 0 pixels never produces a descriptor (see
    `compute_color_histogram`), so this is always > 0 here."""

    histogram: Tuple[float, ...]
    bins_per_channel: int
    pixel_count: int

    def to_dict(self) -> dict:
        return {
            "histogram": list(self.histogram),
            "bins_per_channel": self.bins_per_channel,
            "pixel_count": self.pixel_count,
        }


def compute_color_histogram(
    image: Sequence[Sequence[Tuple[int, int, int]]],
    mask: Sequence[Sequence[bool]],
    bins_per_channel: int = DEFAULT_BINS_PER_CHANNEL,
) -> Optional[AppearanceDescriptor]:
    """Build a normalized RGB histogram over `image` pixels where `mask`
    is True. Returns None (not a fabricated all-zero histogram) when the
    mask selects no pixels or dimensions mismatch -- the honest
    "insufficient evidence" outcome, matching `lift_region_to_3d`."""
    if bins_per_channel <= 0:
        raise ValueError(f"bins_per_channel must be positive, got {bins_per_channel}")
    if len(image) != len(mask):
        return None

    bin_count = bins_per_channel ** 3
    counts: List[int] = [0] * bin_count
    total = 0
    bin_width = 256.0 / bins_per_channel

    for row in range(len(mask)):
        mask_row = mask[row]
        if row >= len(image) or len(image[row]) != len(mask_row):
            return None
        image_row = image[row]
        for col in range(len(mask_row)):
            if not mask_row[col]:
                continue
            r, g, b = image_row[col]
            rb = min(bins_per_channel - 1, int(r / bin_width))
            gb = min(bins_per_channel - 1, int(g / bin_width))
            bb = min(bins_per_channel - 1, int(b / bin_width))
            counts[(rb * bins_per_channel + gb) * bins_per_channel + bb] += 1
            total += 1

    if total == 0:
        return None

    histogram = tuple(c / total for c in counts)
    return AppearanceDescriptor(histogram=histogram, bins_per_channel=bins_per_channel, pixel_count=total)


def appearance_similarity(a: AppearanceDescriptor, b: AppearanceDescriptor) -> float:
    """Histogram intersection similarity in [0, 1]: 1.0 for identical
    normalized histograms, 0.0 for disjoint color distributions.
    Requires matching bin counts -- comparing histograms built with
    different `bins_per_channel` would silently misalign bins."""
    if a.bins_per_channel != b.bins_per_channel:
        raise ValueError(
            f"cannot compare histograms with different bins_per_channel "
            f"({a.bins_per_channel} vs {b.bins_per_channel})"
        )
    return sum(min(x, y) for x, y in zip(a.histogram, b.histogram))
