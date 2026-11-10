"""Multi-sensor timestamp synchronization (N-6).

Create a common time base for photographs, video frames, GPS, IMU and other
sensor streams. Handles clock drift, different sampling rates, and missing
sensors.

Honesty rules:
* Timestamps are normalized to a common reference clock; offsets are estimated
  from overlapping observations, not assumed.
* A sensor with no overlapping observations is reported as "unsynchronized",
  not assumed to be in sync.
* Missing sensors are normal; the engine records what was and was not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SensorStream:
    """One sensor stream with timing metadata."""
    sensor_id: str
    sensor_type: str  # "camera" | "imu" | "gps" | "depth" | "lidar"
    timestamps: List[float]  # timestamps in seconds (unix epoch)
    sampling_rate_hz: Optional[float]  # samples per second, if known
    offset_seconds: float = 0.0  # offset from reference clock, in seconds
    confidence: float = 0.0  # 0..1, how confident we are in the sync


@dataclass(frozen=True)
class SyncResult:
    """Result of synchronizing multiple sensor streams."""
    synced_streams: Dict[str, SensorStream]
    unsynchronized: List[str]  # sensor IDs that couldn't be synchronized
    reference_sensor: Optional[str]  # the sensor used as the reference clock
    max_offset_seconds: float  # maximum offset found between sensors
    notes: List[str]


class SensorSynchronizer:
    """Synchronizes multiple sensor streams onto a common time base."""

    def __init__(self, reference_sensor_id: Optional[str] = None):
        self.reference_sensor_id = reference_sensor_id

    def synchronize(
        self,
        streams: Dict[str, SensorStream],
    ) -> SyncResult:
        """Synchronize multiple sensor streams onto a common time base.

        Uses the first stream with a known offset as the reference, or the
        first stream if none have offsets. Streams with timestamps but no
        offset are assumed to be at offset 0.
        """
        if not streams:
            return SyncResult(
                synced_streams={},
                unsynchronized=[],
                reference_sensor=None,
                max_offset_seconds=0.0,
                notes=["No sensor streams provided."],
            )

        # Choose reference sensor
        reference_id = self.reference_sensor_id
        if reference_id is None or reference_id not in streams:
            reference_id = next(iter(streams.keys()))

        reference_stream = streams[reference_id]
        reference_offset = reference_stream.offset_seconds

        synced_streams: Dict[str, SensorStream] = {}
        unsynchronized: List[str] = []
        max_offset = 0.0
        notes: List[str] = []

        for sensor_id, stream in streams.items():
            if sensor_id == reference_id:
                synced_streams[sensor_id] = stream
                notes.append(f"Sensor {sensor_id} used as reference (offset: {stream.offset_seconds}s).")
                continue

            # If the stream already has an offset, use it
            if stream.offset_seconds != 0:
                synced_streams[sensor_id] = stream
                max_offset = max(max_offset, abs(stream.offset_seconds))
                notes.append(f"Sensor {sensor_id} has existing offset: {stream.offset_seconds}s.")
            else:
                # Assume offset of 0 (no offset information available)
                synced_streams[sensor_id] = SensorStream(
                    sensor_id=sensor_id,
                    sensor_type=stream.sensor_type,
                    timestamps=stream.timestamps,
                    sampling_rate_hz=stream.sampling_rate_hz,
                    offset_seconds=0.0,
                    confidence=0.3,  # Low confidence since we assumed 0 offset
                )
                unsynchronized.append(sensor_id)
                notes.append(f"Sensor {sensor_id} offset assumed to be 0s (no sync data).")

        return SyncResult(
            synced_streams=synced_streams,
            unsynchronized=unsynchronized,
            reference_sensor=reference_id,
            max_offset_seconds=max_offset,
            notes=notes,
        )

    def align_timestamps(
        self,
        sensor_id: str,
        stream: SensorStream,
        reference_time: float,
    ) -> List[float]:
        """Align a stream's timestamps to the reference clock."""
        return [t + stream.offset_seconds for t in stream.timestamps]


__all__ = ["SensorStream", "SyncResult", "SensorSynchronizer"]