"""Tests for evidence/clocks.py (P2.1 time synchronization:
ClockModel/Timestamp/TimeAlignment/SynchronizationDiagnostics per
docs/future/synchronization/TIME_SYNCHRONIZATION.md) and its wiring
into evidence.sensors.SensorStream.synchronized().

Spec rules under test: original sensor timestamps are never
overwritten; uncertainty is never fabricated (unknown stays unknown;
zero only where by construction); sync failure degrades honestly to
UNSYNCHRONIZED -- never a silent re-stamp; the spec's test list
(synthetic offset, synthetic drift, cross-stream alignment,
rejected-sample handling, round-trip provenance).
"""

from __future__ import annotations

import json

import pytest

from evidence.clocks import (
    ClockModel,
    ISynchronizationBackend,
    MetadataAlignmentBackend,
    SynchronizedSample,
    SynchronizationDiagnostics,
    SynchronizationError,
    SynchronizationUnavailable,
    SyncMethod,
    SyncState,
    SyncUncertainty,
    Timestamp,
    TimeAlignment,
    synchronize_stream,
)
from evidence.sensors import SensorStream, parse_imu_jsonl


def _imu_file(tmp_path, records, name="imu.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path)


def _imu_stream(tmp_path, ts):
    return parse_imu_jsonl(
        _imu_file(tmp_path, [
            {"t": t, "accel_mps2": [0.0, 0.0, 9.81], "gyro_rps": [0.0, 0.0, 0.0]}
            for t in ts
        ])
    )


class TestClockModel:
    def test_shared_clock_model_shape(self):
        model = ClockModel(clock_id="cam0", method=SyncMethod.SHARED_CLOCK)
        assert model.a == 1.0 and model.b == 0.0
        assert model.apply(12.5) == 12.5

    def test_apply_linear_map(self):
        model = ClockModel(clock_id="imu0", a=1.0002, b=2.5)
        assert model.apply(100.0) == pytest.approx(102.52)

    def test_nonpositive_drift_refused(self):
        with pytest.raises(SynchronizationError, match="a<=0|drift/scale"):
            ClockModel(clock_id="c", a=0.0, b=0.0)
        with pytest.raises(SynchronizationError):
            ClockModel(clock_id="c", a=-1.0, b=0.0)

    def test_nonfinite_refused(self):
        with pytest.raises(SynchronizationError):
            ClockModel(clock_id="c", a=float("nan"), b=0.0)
        with pytest.raises(SynchronizationError):
            ClockModel(clock_id="c", a=1.0, b=float("inf"))
        with pytest.raises(SynchronizationError):
            ClockModel(clock_id="c", a=1.0, b=0.0).apply(float("nan"))

    def test_empty_clock_id_refused(self):
        with pytest.raises(SynchronizationError):
            ClockModel(clock_id="", a=1.0, b=0.0)

    def test_roundtrip_to_dict_from_dict(self):
        model = ClockModel(
            clock_id="imu0", a=1.0002, b=2.5, method=SyncMethod.KNOWN_OFFSET,
            uncertainty=SyncUncertainty(offset_s=0.01, basis="declared"),
        )
        assert ClockModel.from_dict(model.to_dict()) == model

    def test_timestamp_type(self):
        ts = Timestamp(sensor_t=1.5, clock_id="c0")
        assert ts.sensor_t == 1.5 and ts.clock_id == "c0"
        with pytest.raises(SynchronizationError):
            Timestamp(sensor_t=float("nan"), clock_id="c0")
        with pytest.raises(SynchronizationError):
            Timestamp(sensor_t=1.0, clock_id="")


class TestUncertaintyHonesty:
    def test_zero_uncertainty_is_honest_only_by_construction(self):
        # Shared clock: zero uncertainty is a fact of the hardware, not
        # an estimate.
        unc = SyncUncertainty(offset_s=0.0, basis="shared clock by construction")
        assert unc.offset_s == 0.0

    def test_negative_uncertainty_refused(self):
        with pytest.raises(SynchronizationError, match="finite >= 0"):
            SyncUncertainty(offset_s=-0.5)

    def test_unknown_is_none_never_zero(self):
        # A declared offset without an estimate must NOT read as
        # "perfectly known". None is the honest encoding.
        unc = SyncUncertainty(offset_s=None, basis="declared without an uncertainty estimate")
        assert unc.offset_s is None
        assert unc.to_dict()["offset_s"] is None


class TestSharedClockBackend:
    def test_shared_clock_metadata(self, tmp_path):
        stream = _imu_stream(tmp_path, [10.0, 10.01, 10.02])

        alignment = stream.synchronized("cam0", {"shared_clock": True})

        assert alignment.model.method == SyncMethod.SHARED_CLOCK
        assert alignment.model.a == 1.0 and alignment.model.b == 0.0
        assert alignment.diagnostics.method == "shared_clock"
        assert all(s.sync_state is SyncState.SYNCHRONIZED for s in alignment.samples)
        assert alignment.global_times() == (10.0, 10.01, 10.02)
        # Honest zero: by construction, named in the basis.
        assert alignment.model.uncertainty.offset_s == 0.0
        assert "shared clock" in alignment.model.uncertainty.basis

    def test_pure_offset_synthetic_b(self, tmp_path):
        # Spec test: synthetic offset (pure b). Sensor clock runs 3.2 s
        # behind the global timeline.
        stream = _imu_stream(tmp_path, [0.0, 0.5, 1.0])

        alignment = stream.synchronized("cam1", {"clock_offset_s": 3.2})

        assert alignment.model.method == SyncMethod.KNOWN_OFFSET
        assert alignment.model.a == 1.0 and alignment.model.b == pytest.approx(3.2)
        assert alignment.global_times() == (3.2, 3.7, 4.2)
        # Declared without an uncertainty estimate -> UNKNOWN, not 0.
        assert alignment.model.uncertainty.offset_s is None
        assert "not zero" in alignment.model.uncertainty.basis

    def test_declared_uncertainty_carries(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        alignment = stream.synchronized(
            "cam1", {"clock_offset_s": 3.2, "clock_offset_uncertainty_s": 0.05}
        )

        assert alignment.model.uncertainty.offset_s == pytest.approx(0.05)
        assert alignment.model.uncertainty.basis == "declared in capture metadata"

    def test_synthetic_drift_a_ne_1(self, tmp_path):
        # Spec test: synthetic drift (a != 1). A clock running 200 ppm
        # fast with a 2.5 s offset.
        stream = _imu_stream(tmp_path, [100.0, 200.0])

        alignment = stream.synchronized(
            "imu0", {"clock_offset_s": 2.5, "clock_drift": 1.0002}
        )

        got = alignment.global_times()
        assert got[0] == pytest.approx(1.0002 * 100.0 + 2.5)
        assert got[1] == pytest.approx(1.0002 * 200.0 + 2.5)
        assert alignment.diagnostics.drift_estimate == pytest.approx(1.0002)
        assert alignment.diagnostics.offset_estimate_s == pytest.approx(2.5)

    def test_drift_alone_is_valid(self, tmp_path):
        stream = _imu_stream(tmp_path, [10.0])

        alignment = stream.synchronized("imu0", {"clock_drift": 0.999})

        assert alignment.model.a == pytest.approx(0.999) and alignment.model.b == 0.0
        assert alignment.samples[0].global_t == pytest.approx(9.99)

    def test_bad_metadata_types_are_unavailable_not_errors(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        alignment = stream.synchronized("cam1", {"clock_offset_s": "soon"})

        assert all(s.sync_state is SyncState.UNSYNCHRONIZED for s in alignment.samples)
        assert "must be a number" in alignment.diagnostics.backend_attempts[0]["reason"]


class TestHonestDegradation:
    def test_no_metadata_all_samples_unsynchronized(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0, 2.0, 3.0])

        alignment = stream.synchronized("cam2", {})

        # Every sample degraded, nothing re-stamped, originals intact.
        assert len(alignment.samples) == 3
        assert all(s.sync_state is SyncState.UNSYNCHRONIZED for s in alignment.samples)
        assert all(s.global_t is None for s in alignment.samples)
        assert [s.original_t for s in alignment.samples] == [1.0, 2.0, 3.0]
        assert alignment.diagnostics.method == "none"
        assert alignment.diagnostics.n_synchronized == 0
        assert alignment.diagnostics.n_rejected == 3
        assert any("UNSYNCHRONIZED" in n for n in alignment.diagnostics.notes)

    def test_unsynchronized_cannot_carry_global_t(self):
        with pytest.raises(SynchronizationError, match="fabricated re-stamp"):
            SynchronizedSample(
                index=0, original_t=1.0, clock_id="c", offset_s=0.0, drift=1.0,
                sync_method="none", sync_state=SyncState.UNSYNCHRONIZED, global_t=5.0,
            )

    def test_synchronized_must_carry_global_t(self):
        with pytest.raises(SynchronizationError, match="must carry global_t"):
            SynchronizedSample(
                index=0, original_t=1.0, clock_id="c", offset_s=0.0, drift=1.0,
                sync_method="shared_clock", sync_state=SyncState.SYNCHRONIZED,
                global_t=None,
            )

    def test_backend_attempts_recorded(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        alignment = stream.synchronized("cam2", {"irrelevant": True})

        # Every backend in the registry gets an honest attempt record:
        # metadata first (spec preference order), all declining when no
        # synchronization facts exist, with the reason for each.
        from evidence.clocks import DEFAULT_BACKENDS
        attempts = alignment.diagnostics.backend_attempts
        assert len(attempts) == len(DEFAULT_BACKENDS)
        assert [a["backend"] for a in attempts] == [b.name for b in DEFAULT_BACKENDS]
        assert all(a["ok"] is False for a in attempts)
        assert all(a["reason"] for a in attempts)
        assert "no synchronization metadata" in attempts[0]["reason"]


class TestBackendSelectionSeam:
    def test_preferred_backend_wins(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        class PriorityBackend(ISynchronizationBackend):
            name = "priority_test"

            def build_model(self, stream, context):
                return ClockModel(
                    clock_id=context.clock_id, method=SyncMethod.GNSS_PPS,
                    uncertainty=SyncUncertainty(offset_s=1e-6, basis="pps"),
                )

        alignment = synchronize_stream(
            stream, "cam0", {"shared_clock": True},
            backends=(PriorityBackend(), MetadataAlignmentBackend()),
        )

        assert alignment.model.method == SyncMethod.GNSS_PPS
        assert alignment.diagnostics.backend_attempts[0]["backend"] == "priority_test"
        assert alignment.diagnostics.backend_attempts[0]["ok"] is True

    def test_first_declining_backend_falls_through(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        class Declines(ISynchronizationBackend):
            name = "declines"

            def build_model(self, stream, context):
                raise SynchronizationUnavailable("not for this stream")

        alignment = synchronize_stream(
            stream, "cam0", {"shared_clock": True},
            backends=(Declines(), MetadataAlignmentBackend()),
        )

        assert alignment.model.method == SyncMethod.SHARED_CLOCK
        assert [a["backend"] for a in alignment.diagnostics.backend_attempts] == [
            "declines", "metadata_alignment",
        ]

    def test_all_declining_degrades_honestly(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        class Declines(ISynchronizationBackend):
            name = "declines"

            def build_model(self, stream, context):
                raise SynchronizationUnavailable("no anchor available")

        alignment = synchronize_stream(
            stream, "cam0", {}, backends=(Declines(), MetadataAlignmentBackend())
        )

        assert alignment.diagnostics.method == "none"
        assert len(alignment.diagnostics.backend_attempts) == 2
        assert all(s.sync_state is SyncState.UNSYNCHRONIZED for s in alignment.samples)

    def test_empty_backend_registry_refused(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])
        with pytest.raises(SynchronizationError, match="must not be empty"):
            synchronize_stream(stream, "cam0", {}, backends=())


class TestOriginalTimestampsPreserved:
    @pytest.mark.parametrize("meta", [
        {"shared_clock": True},
        {"clock_offset_s": 7.5, "clock_drift": 1.001},
        {},
    ])
    def test_stream_object_untouched_and_originals_retained(self, tmp_path, meta):
        stream = _imu_stream(tmp_path, [5.0, 6.0, 7.0])
        before = stream.to_dict()

        alignment = stream.synchronized("c0", meta)

        # The source stream is byte-identical after synchronization.
        assert stream.to_dict() == before
        # Every aligned sample rides the original sensor timestamp.
        for original, synced in zip(stream.samples, alignment.samples):
            assert synced.original_t == original.t
            assert synced.index == stream.samples.index(original)

    def test_rejected_samples_keep_originals(self, tmp_path):
        # A non-monotonic stream (real logging fault): samples stay in
        # file order, originals retained, fault recorded.
        stream = _imu_stream(tmp_path, [1.0, 3.0, 2.0])

        alignment = stream.synchronized("c0", {"shared_clock": True})

        assert alignment.diagnostics.stream_was_monotonic is False
        assert "not monotonic" in " ".join(alignment.diagnostics.notes)
        assert [s.original_t for s in alignment.samples] == [1.0, 3.0, 2.0]


class TestRejectedSampleHandling:
    def test_nonfinite_sample_time_rejected_individually(self, tmp_path):
        # A corrupt record sneaking a NaN into the file (the parser
        # guards its own path; this exercises the sync layer's
        # per-sample guard for streams assembled from other sources).
        stream = _imu_stream(tmp_path, [1.0, 2.0])

        alignment = synchronize_stream(stream, "c0", {"shared_clock": True})

        # Baseline: everything syncs; then inject a bad sample through
        # the public seam by replacing the stream's samples tuple via a
        # wrapper object (SensorStream is frozen; wrapper mirrors the
        # contract synchronize_stream depends on).
        class Wrapper:
            samples = stream.samples + (type("Bad", (), {"t": float("nan")})(),)
            provenance = stream.provenance
            kind = stream.kind
            source_path = stream.source_path

            def is_monotonic(self):
                return stream.is_monotonic()

        alignment_bad = synchronize_stream(Wrapper(), "c0", {"shared_clock": True})

        assert alignment_bad.diagnostics.n_synchronized == 2
        assert alignment_bad.diagnostics.n_rejected == 1
        assert alignment_bad.samples[2].rejection_reason is not None
        assert "finite" in alignment_bad.samples[2].rejection_reason
        # Good samples unaffected.
        assert alignment_bad.samples[0].global_t == 1.0


class TestRoundTripProvenance:
    def test_alignment_roundtrip(self, tmp_path):
        stream = _imu_stream(tmp_path, [0.0, 0.5])

        alignment = stream.synchronized("cam1", {"clock_offset_s": 3.2, "clock_drift": 1.0002})

        data = alignment.to_dict()
        model = ClockModel.from_dict(data["model"])
        assert model == alignment.model
        assert data["diagnostics"]["offset_estimate_s"] == pytest.approx(3.2)
        assert data["diagnostics"]["drift_estimate"] == pytest.approx(1.0002)
        assert data["samples"][0]["original_t"] == 0.0
        assert data["samples"][0]["sync_state"] == "SYNCHRONIZED"

    def test_diagnostics_dict_shape(self, tmp_path):
        stream = _imu_stream(tmp_path, [1.0])

        alignment = stream.synchronized("cam2", {})

        d = alignment.diagnostics.to_dict()
        for key in ("method", "n_samples", "n_synchronized", "n_rejected",
                    "offset_estimate_s", "drift_estimate", "residual_max_s",
                    "stream_was_monotonic", "rejected", "backend_attempts", "notes"):
            assert key in d


class TestCrossStreamAlignment:
    """The spec's cross-stream integration test: two streams on
    different device clocks map onto ONE timeline, and their samples
    interleave correctly."""

    def test_two_cameras_offset_by_known_value(self, tmp_path):
        cam_a = parse_imu_jsonl(
            _imu_file(tmp_path, [
                {"t": t, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]}
                for t in (0.0, 1.0, 2.0)
            ], name="cam_a.jsonl")
        )
        cam_b = parse_imu_jsonl(
            _imu_file(tmp_path, [
                {"t": t, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]}
                for t in (0.0, 1.0, 2.0)
            ], name="cam_b.jsonl")
        )

        al_a = cam_a.synchronized("cam_a", {"shared_clock": True})
        al_b = cam_b.synchronized("cam_b", {"clock_offset_s": 10.0})

        # cam_b's local 0/1/2 lands at 10/11/12 on the global timeline.
        assert al_a.global_times() == (0.0, 1.0, 2.0)
        assert al_b.global_times() == (10.0, 11.0, 12.0)

        # Cross-stream consumers use only synchronized samples.
        merged = sorted(
            [("a", t) for t in al_a.global_times()]
            + [("b", t) for t in al_b.global_times()]
        )
        assert merged[0][0] == "a" and merged[-1][0] == "b"
        assert [t for _, t in merged] == [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]

    def test_drifted_second_stream_diverges_linearly(self, tmp_path):
        cam_b = parse_imu_jsonl(
            _imu_file(tmp_path, [
                {"t": t, "accel_mps2": [0, 0, 9.81], "gyro_rps": [0, 0, 0]}
                for t in (0.0, 100.0, 200.0)
            ])
        )

        al_b = cam_b.synchronized("cam_b", {"clock_offset_s": 1.0, "clock_drift": 1.1})

        # Drift compounds: local 200 s lands at 221 s global.
        assert al_b.global_times() == pytest.approx((1.0, 111.0, 221.0))

# ==== Spec preference backends 3-6 (GNSS/PPS, trigger, correlation, optimization) ====

from evidence.clocks import (
    DEFAULT_BACKENDS,
    GnssPpsBackend,
    OptimizationBackend,
    SignalCorrelationBackend,
    TriggerBackend,
    synchronize_stream as _sync,
)


class _FakeSample:
    def __init__(self, t):
        self.t = t


class _FakeStream:
    def __init__(self, kind, ts):
        self.kind = kind
        self.samples = [_FakeSample(t) for t in ts]
        self.provenance = None
        self.identity = None

    def is_monotonic(self):
        ts = [s.t for s in self.samples]
        return all(b >= a for a, b in zip(ts, ts[1:]))


class _Ctx:
    def __init__(self, clock_id, metadata=None):
        self.clock_id = clock_id
        self.metadata = dict(metadata or {})


class TestGnssPpsBackend:
    def test_own_gnss_samples_identity_fit(self):
        # A PPS-disciplined GNSS stream is already in GNSS time: the
        # identity model with zero residual BY CONSTRUCTION.
        backend = GnssPpsBackend()
        model = backend.build_model(_FakeStream("gnss", [10.0, 11.0, 12.0, 13.0]), _Ctx("gnss0"))
        assert model.method is SyncMethod.GNSS_PPS
        assert abs(model.a - 1.0) < 1e-12 and abs(model.b) < 1e-12
        assert "by construction" in model.uncertainty.basis
        # Identity from the stream's own PPS-disciplined samples has no
        # measured fit residual: the honest report is None, not a
        # fabricated 0.0.
        assert model.uncertainty.offset_s is None

    def test_imu_stream_declines_without_anchors(self):
        backend = GnssPpsBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(_FakeStream("imu", [0.0, 1.0]), _Ctx("cam0"))

    def test_declared_anchors_recover_offset_and_drift(self):
        # Ground truth: t_global = 1.001 * t_sensor + 100.
        def truth(t):
            return 1.001 * t + 100.0

        anchors = [(t, truth(t)) for t in (0.0, 10.0, 20.0, 30.0)]
        backend = GnssPpsBackend()
        model = backend.build_model(
            _FakeStream("imu", [t for t, _ in anchors]),
            _Ctx("cam0", {"gnss_anchor_pairs": anchors}),
        )
        assert abs(model.a - 1.001) < 1e-9
        assert abs(model.b - 100.0) < 1e-6
        assert model.method is SyncMethod.GNSS_PPS

    def test_bad_anchor_shape_declines(self):
        backend = GnssPpsBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(
                _FakeStream("imu", [0.0, 1.0]),
                _Ctx("cam0", {"gnss_anchor_pairs": [(0.0, 1.0, 2.0)]}),
            )


class TestTriggerBackend:
    def test_trigger_pairs_known_answer(self):
        backend = TriggerBackend()
        model = backend.build_model(
            _FakeStream("imu", [1.0, 2.0, 3.0]),
            _Ctx("cam0", {"trigger_pairs": [(1.0, 50.0), (2.0, 51.0), (3.0, 52.0)]}),
        )
        assert model.method is SyncMethod.TRIGGER
        assert abs(model.a - 1.0) < 1e-12 and abs(model.b - 49.0) < 1e-9

    def test_two_triggers_exact_two_point_fit(self):
        backend = TriggerBackend()
        # Trigger pair = (sensor_t, global_t) at the same physical
        # trigger event: sensor 0.0 == global 7.5; sensor 5.0 == global
        # 7.505 (the sensor clock ran fast: 5 s of sensor time over 5 ms
        # of global time is absurd, so read it as the sensor clock being
        # ~1000x slow -- use a physically sensible drift instead).
        model = backend.build_model(
            _FakeStream("imu", [0.0, 5.0]),
            _Ctx("cam0", {"trigger_pairs": [(0.0, 7.5), (5.0, 7.505)]}),
        )
        # t_global = a*t_sensor + b with a = 0.001: 5 s of sensor time
        # spans 5 ms of global time. Known answer from the two pairs.
        assert abs(model.a - 0.001) < 1e-12
        assert abs(model.b - 7.5) < 1e-12
        # No redundancy -> no measured residual; uncertainty reports
        # the by-construction basis WITHOUT a fabricated 0.0 offset.
        assert "zero by construction" in model.uncertainty.basis
        assert model.uncertainty.offset_s is None

    def test_no_trigger_pairs_declines(self):
        backend = TriggerBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(_FakeStream("imu", [0.0, 1.0]), _Ctx("cam0"))


class TestSignalCorrelationBackend:
    def _ref(self):
        # Sparse, distinct flash times.
        return [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (5.0, 1.0), (7.0, 1.0), (9.0, 1.0)]

    def test_offset_recovered_from_event_train(self):
        ref = self._ref()
        obs = [(t + 3.7, 1.0) for t, _ in ref]
        backend = SignalCorrelationBackend()
        model = backend.build_model(
            _FakeStream("imu", [t for t, _ in obs]),
            _Ctx("cam0", {"reference_events": ref, "observed_events": obs}),
        )
        assert model.method is SyncMethod.SIGNAL_CORRELATION
        # Observed events at t_sensor = t_global + 3.7 map back with
        # t_global = 1.0 * t_sensor - 3.7 (model convention t_global =
        # a*t_sensor + b).
        assert abs(model.b + 3.7) < 0.05
        assert abs(model.a - 1.0) < 0.02

    def test_drift_recovered_from_event_train(self):
        ref = self._ref()
        # The sensor clock runs SLOW: its seconds are 1.01 global
        # seconds, so t_global = 1.01 * t_sensor recovers the reference
        # train from obs = t_ref / 1.01.
        a_true = 1.01
        b_true = 0.0
        obs = [(t / 1.01, 1.0) for t, _ in ref]
        backend = SignalCorrelationBackend()
        model = backend.build_model(
            _FakeStream("imu", [t for t, _ in obs]),
            _Ctx("cam0", {"reference_events": ref, "observed_events": obs}),
        )
        assert abs(model.a - a_true) < 0.005
        assert abs(model.b - b_true) < 0.05
        # The recovered model must actually re-align the trains: applying
        # it to the observed times reproduces the reference times.
        for (t_ref, _), s in zip(ref, [model.apply(t) for t, _ in obs]):
            assert abs(s - t_ref) < 0.02

    def test_missing_trains_decline(self):
        backend = SignalCorrelationBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(_FakeStream("imu", [0.0, 1.0]), _Ctx("cam0"))

    def test_single_event_trains_decline(self):
        backend = SignalCorrelationBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(
                _FakeStream("imu", [0.0, 1.0]),
                _Ctx("cam0", {
                    "reference_events": [(0.0, 1.0)],
                    "observed_events": [(1.0, 1.0)],
                }),
            )

    def test_unalignable_trains_decline(self):
        # Observed train with drift far outside the search space (0.5x
        # rate): no hypothesis in the grid aligns the trains; the
        # backend must decline, not best-effort. (A pure constant shift
        # IS alignable -- start-alignment recovers it exactly -- so drift
        # outside the searched band is the genuine decline case.)
        backend = SignalCorrelationBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(
                _FakeStream("imu", [0.5 * t + 500.0 for t, _ in self._ref()]),
                _Ctx("cam0", {
                    "reference_events": self._ref(),
                    "observed_events": [(0.5 * t + 500.0, 1.0) for t, _ in self._ref()],
                }),
            )


class TestOptimizationBackend:
    def test_two_clocks_offset_known_answer(self):
        # cam1 records the same events 5 s LATER on its own clock:
        # constraint (cam0, t, cam1, t+5) -> b_1 = -5 with the gauge on cam0.
        backend = OptimizationBackend()
        model = backend.build_model(
            _FakeStream("imu", [0.0, 10.0, 20.0]),
            _Ctx("cam1", {"cross_stream_constraints": [
                ("cam0", 0.0, "cam1", 5.0),
                ("cam0", 10.0, "cam1", 15.0),
                ("cam0", 20.0, "cam1", 25.0),
            ]}),
        )
        assert model.method is SyncMethod.OPTIMIZATION
        assert abs(model.a - 1.0) < 1e-12
        assert abs(model.b + 5.0) < 1e-9

    def test_two_clocks_drift_known_answer(self):
        expected_a = 100.0 / 100.2  # cam1's span is 0.2% longer
        backend = OptimizationBackend()
        model = backend.build_model(
            _FakeStream("imu", [0.0, 100.0, 200.0]),
            _Ctx("cam1", {"cross_stream_constraints": [
                ("cam0", 0.0, "cam1", 2.0),
                ("cam0", 100.0, "cam1", 102.2),
                ("cam0", 200.0, "cam1", 202.4),
            ]}),
        )
        assert abs(model.a - expected_a) < 1e-12
        assert abs(model.b + 2.0 * expected_a) < 1e-9

    def test_three_clocks_joint_fit(self):
        backend = OptimizationBackend()
        model = backend.build_model(
            _FakeStream("imu", []),
            _Ctx("cam2", {"cross_stream_constraints": [
                ("cam0", 0.0, "cam1", 5.0),
                ("cam0", 0.0, "cam2", 9.0),
                ("cam0", 10.0, "cam1", 15.0),
                ("cam0", 10.0, "cam2", 19.0),
                ("cam1", 0.0, "cam2", 4.0),
            ]}),
        )
        assert abs(model.a - 1.0) < 1e-12 and abs(model.b + 9.0) < 1e-9

    def test_residual_is_real(self):
        # One inconsistent constraint among consistent ones -> the max
        # residual reflects it (estimates and residuals are measured).
        backend = OptimizationBackend()
        model = backend.build_model(
            _FakeStream("imu", [0.0, 10.0, 20.0]),
            _Ctx("cam1", {"cross_stream_constraints": [
                ("cam0", 0.0, "cam1", 5.0),
                ("cam0", 10.0, "cam1", 15.0),
                ("cam0", 20.0, "cam1", 25.0),
                ("cam0", 30.0, "cam1", 36.0),  # 1 s off the true +5 offset
            ]}),
        )
        assert model.uncertainty.offset_s is not None
        assert model.uncertainty.offset_s > 0.1  # the inconsistent pair shows

    def test_underdetermined_declines(self):
        backend = OptimizationBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(
                _FakeStream("imu", [0.0]),
                _Ctx("cam1", {"cross_stream_constraints": [("cam0", 0.0, "cam1", 5.0)]}),
            )

    def test_uninvolved_clock_declines(self):
        backend = OptimizationBackend()
        with pytest.raises(SynchronizationUnavailable):
            backend.build_model(
                _FakeStream("imu", [0.0, 1.0]),
                _Ctx("cam9", {"cross_stream_constraints": [
                    ("cam0", 0.0, "cam1", 5.0), ("cam0", 1.0, "cam1", 6.0),
                ]}),
            )


class TestBackendPreferenceOrder:
    def test_registry_is_spec_ordered(self):
        names = [type(b).__name__ for b in DEFAULT_BACKENDS]
        assert names == [
            "MetadataAlignmentBackend",
            "GnssPpsBackend",
            "TriggerBackend",
            "SignalCorrelationBackend",
            "OptimizationBackend",
        ]

    def test_metadata_outranks_estimation(self):
        # A declared shared-clock fact must beat an estimated anchor fit:
        # with both available, the model is SHARED_CLOCK (preference 1).
        stream = _FakeStream("gnss", [10.0, 11.0, 12.0])
        alignment = _sync(
            stream, "gnss0",
            metadata={"shared_clock": True, "gnss_anchor_pairs": [(10.0, 20.0), (11.0, 21.0), (12.0, 22.0)]},
            backends=DEFAULT_BACKENDS,
        )
        assert alignment.model.method is SyncMethod.SHARED_CLOCK

    def test_gnss_backend_answers_before_trigger(self):
        stream = _FakeStream("gnss", [10.0, 11.0, 12.0])
        alignment = _sync(stream, "gnss0", backends=DEFAULT_BACKENDS)
        assert alignment.model.method is SyncMethod.GNSS_PPS
        # Identity-by-construction has no measured residual: None, not
        # a fabricated 0.0.
        assert alignment.diagnostics.residual_max_s is None

    def test_full_fallback_chain_to_unsynchronized(self):
        # No facts at all: every backend declines; samples are honestly
        # UNSYNCHRONIZED with one attempt record per registry backend.
        stream = _FakeStream("imu", [0.0, 1.0, 2.0])
        alignment = _sync(stream, "cam0", backends=DEFAULT_BACKENDS)
        assert alignment.diagnostics.method == "none"
        assert len(alignment.diagnostics.backend_attempts) == len(DEFAULT_BACKENDS)
        assert all(s.sync_state is SyncState.UNSYNCHRONIZED for s in alignment.samples)


class TestEstimatorOutlierRejection:
    def test_outlier_pair_rejected_and_refit(self):
        # Trigger pairs with one wildly wrong pair -> robust refit lands
        # on the truth and the rejected pair is recorded.
        backend = TriggerBackend()
        ctx = _Ctx("cam0", {"trigger_pairs": [
            (0.0, 100.0),
            (1.0, 101.0),
            (2.0, 102.0),
            (3.0, 103.0),
            (4.0, 999.0),  # outlier
        ]})
        model = backend.build_model(_FakeStream("imu", [0.0, 1.0, 2.0, 3.0, 4.0]), ctx)
        assert abs(model.a - 1.0) < 1e-9
        assert abs(model.b - 100.0) < 1e-9
        assert ctx.metadata.get("rejected_pairs"), "outlier pair must be recorded"
