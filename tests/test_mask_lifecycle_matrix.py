"""B07: the mask lifecycle through the EvidenceMatrixStore — model output ->
artifact persistence -> EvidenceMatrixEntry -> reload -> retrieve -> verify
(mask dimensions, reference, coordinate frame, provenance, confidence) ->
cross-view identity via query.

Covers the directive's mask-lifecycle test list at the matrix level:
mask dimensions, coordinate frames, empty masks, missing masks, multiple
masks, mask references, persistence and reload, cross-view identity, and
evidence provenance. The mask artifact itself is RLE-encoded (shared with
tests/test_mask_store.py); the matrix carries the reference + inline
mask_width/mask_height metadata (a genuinely-missing field added to
EvidenceMatrixEntry for inline dimension checks without loading the
artifact).
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.multiengine.evidence_matrix import EvidenceMatrixEntry, EvidenceMatrixError, EvidenceMatrixStore


def _encode_mask(mask: np.ndarray) -> bytes:
    """RLE-encode a boolean mask to bytes for the EvidenceMatrixStore bytes
    artifact store. mask_store.encode() returns a uint32 numpy array; we
    serialise that array to bytes so save_artifact (bytes only) can store it,
    and the test reconstructs the array from those bytes before calling
    mask_store.decode()."""
    from world_ir.mask_store import encode

    rle = encode(mask)
    return rle.astype(np.uint32, copy=False).tobytes()


def _decode_mask(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Reconstruct a RLE-encoded mask from its bytes."""
    from world_ir.mask_store import decode

    rle = np.frombuffer(data, dtype=np.uint32)
    return decode(rle, shape)


class TestMaskLifecycleThroughMatrix:
    """Mask lifecycle: produce -> persist artifact + entry -> reload -> retrieve
    -> verify dimensions/reference/frame/provenance/confidence -> cross-view
    identity via query."""

    def test_mask_lifecycle_add_persist_reload_retrieve_verify(
        self, tmp_path: str,
    ) -> None:
        store = EvidenceMatrixStore(tmp_path)
        mask = np.random.default_rng(0).random((480, 640)) > 0.5
        w, h = mask.shape[1], mask.shape[0]

        # 1. Model output -> persist the mask artifact (bytes).
        rel = store.save_artifact("m-001", "mask.rle", _encode_mask(mask))

        # 2. Create the matrix entry carrying the reference + inline dimensions.
        entry = _make_mask_entry(
            raw_output_reference=rel,
            mask_width=w,
            mask_height=h,
            entity_association="entity-001",
            confidence=0.92,
        )
        assert store.add_entry(entry)

        # 3. Persist + reload (new store from the same disk).
        reloaded = EvidenceMatrixStore(tmp_path)

        # 4. Retrieve + verify the lifecycle fields.
        got = reloaded.get_entry("m-001")
        assert got.prediction_type == "segmentation_mask"
        assert got.coordinate_frame == "camera_local"
        assert got.mask_width == w
        assert got.mask_height == h
        assert got.confidence == 0.92
        assert got.entity_association == "entity-001"
        assert got.provenance == {"model": "sam2@1", "source": "model_output"}
        assert got.raw_output_reference == rel
        assert got.source_ids == ["img-m-001"]

        # 5. Mask reference resolves and the artifact shape matches the inline
        #    dimensions (mask dimensions verified end-to-end, not just asserted).
        data = reloaded.get_artifact(rel)
        decoded_mask = _decode_mask(data, (h, w))
        assert decoded_mask.shape == (h, w)
        assert got.mask_width == decoded_mask.shape[1]
        assert got.mask_height == decoded_mask.shape[0]
        assert np.array_equal(decoded_mask, mask)

        # 5b. Tamper check: a corrupted artifact is detected on retrieval.
        bad_path = reloaded.storage_root / rel
        bad_path.write_bytes(b"tampered")
        with pytest.raises(EvidenceMatrixError, match="corrupted"):
            reloaded.get_artifact(rel)

    def test_empty_mask_is_a_real_mask(self, tmp_path: str) -> None:
        """An all-False mask is a valid observation (nothing detected), not
        missing data. It must persist, reload, and carry its dimensions."""
        store = EvidenceMatrixStore(tmp_path)
        mask = np.zeros((240, 320), dtype=bool)
        rel = store.save_artifact("m-empty", "mask.rle", _encode_mask(mask))
        entry = _make_mask_entry(
            evidence_id="m-empty",
            raw_output_reference=rel,
            mask_width=mask.shape[1],
            mask_height=mask.shape[0],
            entity_association="entity-empty",
            confidence=0.0,  # model explicitly reports nothing
            provenance={"model": "sam2@1", "source": "model_output", "note": "nothing detected"},
        )
        store.add_entry(entry)
        reloaded = EvidenceMatrixStore(tmp_path)
        got = reloaded.get_entry("m-empty")
        assert got.mask_width == 320 and got.mask_height == 240
        assert got.confidence == 0.0
        assert got.entity_association == "entity-empty"
        assert np.array_equal(_decode_mask(reloaded.get_artifact(rel), (240, 320)), mask)

    def test_missing_mask_artifact_is_an_error_not_silent(self, tmp_path: str) -> None:
        """A reference to a mask artifact that does not exist is an error on
        retrieval, never a silent no-mask result."""
        store = EvidenceMatrixStore(tmp_path)
        entry = _make_mask_entry(
            evidence_id="m-missing",
            raw_output_reference="evidence_artifacts/m-missing/nope.rle",
            mask_width=640,
            mask_height=480,
        )
        store.add_entry(entry)
        reloaded = EvidenceMatrixStore(tmp_path)
        got = reloaded.get_entry("m-missing")
        assert got.raw_output_reference == "evidence_artifacts/m-missing/nope.rle"
        with pytest.raises(EvidenceMatrixError, match="not found"):
            reloaded.get_artifact(got.raw_output_reference)

    def test_multiple_masks_and_cross_view_identity(self, tmp_path: str) -> None:
        """Multiple masks across views share an entity association (cross-view
        identity) and are retrievable by it via query."""
        store = EvidenceMatrixStore(tmp_path)
        views = [
            ("m-v0", "img-a", "entity-chair-001", 640, 480),
            ("m-v1", "img-b", "entity-chair-001", 640, 480),
            ("m-v2", "img-c", "entity-chair-002", 640, 480),
        ]
        for eid, img, ent, w, h in views:
            mask = np.random.default_rng(hash(eid) % 2**31).random((h, w)) > 0.5
            rel = store.save_artifact(eid, "mask.rle", _encode_mask(mask))
            entry = _make_mask_entry(
                evidence_id=eid,
                source_ids=[img],
                raw_output_reference=rel,
                mask_width=w,
                mask_height=h,
                entity_association=ent,
                confidence=0.85 + 0.05 * (hash(eid) % 2),
            )
            store.add_entry(entry)

        reloaded = EvidenceMatrixStore(tmp_path)

        chair = reloaded.query(entity_association="entity-chair-001")
        assert len(chair) == 2
        assert {e.evidence_id for e in chair} == {"m-v0", "m-v1"}
        for e in chair:
            assert e.entity_association == "entity-chair-001"
            assert e.mask_width == 640 and e.mask_height == 480

        other = reloaded.query(entity_association="entity-chair-002")
        assert len(other) == 1 and other[0].evidence_id == "m-v2"

        # Query by source image (another cross-view identity axis).
        by_img = reloaded.query(source_id="img-b")
        assert len(by_img) == 1 and by_img[0].evidence_id == "m-v1"

    def test_mask_lifecycle_immutability_and_hash(self, tmp_path: str) -> None:
        """Entries are immutable and content-hash-verified: a reload validates
        every entry's hash, and a tampered index line is caught on load."""
        store = EvidenceMatrixStore(tmp_path)
        entry = _make_mask_entry()
        store.add_entry(entry)
        reloaded = EvidenceMatrixStore(tmp_path)
        assert reloaded.get_entry("m-001").evidence_id == "m-001"

        # Tamper the on-disk entry line: reload must raise (corrupt hash).
        index = tmp_path / "evidence_matrix.jsonl"
        text = index.read_text(encoding="utf-8")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if '"evidence_id": "m-001"' in line:
                lines[i] = line.replace('"mask_width": 640', '"mask_width": 999')
                break
        index.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(EvidenceMatrixError, match="Corrupt entry"):
            EvidenceMatrixStore(tmp_path)


def _make_mask_entry(
    *,
    evidence_id: str = "m-001",
    model_id: str = "sam2@1",
    architecture_id: str = "sam2",
    checkpoint_id: str = "sam2-large",
    run_id: str = "run-001",
    source_ids: list[str] | None = None,
    input_parameters: dict | None = None,
    prediction_type: str = "segmentation_mask",
    raw_output_reference: str = "evidence_artifacts/m-001/mask.rle",
    coordinate_frame: str = "camera_local",
    camera_reference: dict | None = None,
    world_transform: list[list[float]] | None = None,
    entity_association: str | None = None,
    mask_width: int = 640,
    mask_height: int = 480,
    confidence: float = 0.92,
    uncertainty: dict | None = None,
    provenance: dict | None = None,
    validation_results: dict | None = None,
    execution_status: str = "completed",
) -> EvidenceMatrixEntry:
    return EvidenceMatrixEntry(
        evidence_id=evidence_id,
        model_id=model_id,
        architecture_id=architecture_id,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        source_ids=source_ids or [f"img-{evidence_id}"],
        input_parameters=input_parameters or {"prompt": "chair"},
        prediction_type=prediction_type,
        raw_output_reference=raw_output_reference,
        coordinate_frame=coordinate_frame,
        camera_reference=camera_reference or {"intrinsics": [960.0, 960.0, 640.0, 360.0]},
        world_transform=world_transform,
        entity_association=entity_association or f"entity-{evidence_id}",
        mask_width=mask_width,
        mask_height=mask_height,
        confidence=confidence,
        uncertainty=uncertainty or {"method": "model", "value": 0.08},
        provenance=provenance or {"model": model_id, "source": "model_output"},
        validation_results=validation_results or {"mask_area_frac": 0.31},
        execution_status=execution_status,
    )
