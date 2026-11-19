"""DUSt3R and MASt3R Multi-View Geometric 3D Regression Adapters.

Implements unconstrained stereoscopic 3D pointmap regression and dense correspondence
matching using DUSt3R and MASt3R.
IMPORTANT LICENCE NOTICE:
Both DUSt3R and MASt3R are released under CC-BY-NC-4.0 (Non-Commercial).
Commercial use is prohibited without an explicit licence from the authors.
`commercial_ok` is strictly False.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import numpy as np

from engine.multiengine.adapters.base import BaseModelEngineAdapter
from engine.multiengine.contract import (
    Availability,
    AvailabilityStatus,
    Device,
    EngineError,
    EngineErrorKind,
    EngineRequest,
    EngineSpec,
)
from engine.multiengine.observation import ObservationDraft


def probe_dust3r() -> Availability:
    """Honest availability probe for DUSt3R."""
    if importlib.util.find_spec("dust3r") is None:
        return Availability(
            AvailabilityStatus.BLOCKED,
            "dust3r package is not installed (CC-BY-NC-4.0 restricted licence)",
            "install dust3r from official repository for non-commercial research",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        "dust3r package importable (non-commercial use only)",
        evidence=("dust3r importable", "licence CC-BY-NC-4.0 acknowledged"),
    )


def probe_mast3r() -> Availability:
    """Honest availability probe for MASt3R."""
    if importlib.util.find_spec("mast3r") is None:
        return Availability(
            AvailabilityStatus.BLOCKED,
            "mast3r package is not installed (CC-BY-NC-4.0 restricted licence)",
            "install mast3r from official repository for non-commercial research",
        )
    return Availability(
        AvailabilityStatus.AVAILABLE,
        "mast3r package importable (non-commercial use only)",
        evidence=("mast3r importable", "licence CC-BY-NC-4.0 acknowledged"),
    )


class Dust3rEngineAdapter(BaseModelEngineAdapter):
    """Adapter for DUSt3R 3D pointmap regression."""

    def __init__(self, spec: EngineSpec) -> None:
        super().__init__(spec)

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("dust3r") is None:
            raise EngineError(
                EngineErrorKind.UNAVAILABLE,
                "dust3r package is not installed.",
            )
        import dust3r  # type: ignore

        self._model = dust3r

    def validate_input(self, request: EngineRequest) -> None:
        super().validate_input(request)
        if len(request.sources) < 2:
            raise EngineError(
                EngineErrorKind.INVALID_INPUT,
                f"DUSt3R requires at least 2 views for pairwise pointmap regression, got {len(request.sources)}.",
            )

    def infer(self, request: EngineRequest) -> Sequence[ObservationDraft]:
        drafts: list[ObservationDraft] = []
        sources = request.sources

        virtual_run = all(str(s).startswith("virtual:") for s in sources)
        if not virtual_run and self._model is None:
            raise EngineError(
                EngineErrorKind.RUNTIME_ERROR,
                f"Model {self.spec.engine_id} not loaded. Call load() before infer().",
            )
        if virtual_run:
            for i, src in enumerate(sources):
                extrinsic_4x4 = np.eye(4, dtype=float)
                extrinsic_4x4[0, 3] = float(i) * 0.6
                drafts.append(
                    ObservationDraft(
                        task="sfm",
                        quantity="camera_pose",
                        source_id=str(src),
                        coordinate_system="world:dust3r",
                        predicted={
                            "extrinsic_4x4": extrinsic_4x4.tolist(),
                            "pointmap_shape": [512, 512, 3],
                            "is_test_double": True,
                        },
                        original_coordinates={"pointmap_index": i},
                        confidence=None,
                    )
                )
            drafts.append(
                ObservationDraft(
                    task="depth",
                    quantity="dense_pointmap",
                    source_id=str(sources[0]),
                    coordinate_system="world:dust3r",
                    predicted={
                        "point_count": 32000,
                        "confidence_mean": 0.87,
                        "is_test_double": True,
                    },
                    original_coordinates={"pointmap_pts": 32000},
                    confidence=None,
                )
            )
            return drafts

        try:
            res = self._model.inference(list(sources), device=str(self._device))
            for i, src in enumerate(sources):
                conf_val = float(res["conf"][0]) if "conf" in res and res["conf"] is not None else None
                drafts.append(
                    ObservationDraft(
                        task="depth",
                        quantity="dense_pointmap",
                        source_id=str(src),
                        coordinate_system="world:dust3r",
                        predicted={"point_count": len(res["pts3d"][i])},
                        original_coordinates={"raw_pts": res["pts3d"][i]},
                        confidence=conf_val,
                    )
                )
        except Exception as e:
            raise EngineError(EngineErrorKind.MODEL_ERROR, f"DUSt3R inference failed: {e}") from e

        return drafts


class Mast3rEngineAdapter(Dust3rEngineAdapter):
    """Adapter for MASt3R matching and 3D stereo reconstruction."""

    def __init__(self, spec: EngineSpec) -> None:
        super().__init__(spec)

    def _load_model_weights(self, device: Device) -> None:
        if importlib.util.find_spec("mast3r") is None:
            raise EngineError(
                EngineErrorKind.UNAVAILABLE,
                "mast3r package is not installed.",
            )
        import mast3r  # type: ignore

        self._model = mast3r
