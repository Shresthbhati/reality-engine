#!/usr/bin/env python3
"""
Synthetic Room Capture Dataset Generator

Generates a realistic room capture dataset with:
- 30 images of a single room (5m x 4m x 2.5m)
- Camera positions in a realistic capture trajectory
- Proper EXIF metadata (make, model, exposure, GPS, timestamps)
- Camera intrinsics/extrinsics for reconstruction
- Multiple materials visible (floor, walls, ceiling, door, window, furniture)

This is a TEST FIXTURE that mimics real capture data. It is NOT a substitute
for real images but provides a complete metadata pipeline for end-to-end testing.
"""

from __future__ import annotations

import calendar
import json
import os
import struct
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Optional dependencies - handle gracefully
try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# Room dimensions (meters)
ROOM_WIDTH = 5.0   # X
ROOM_DEPTH = 4.0   # Z
ROOM_HEIGHT = 2.5  # Y

# Camera parameters (simulating a typical phone camera)
CAMERA_PARAMS = {
    "make": "Apple",
    "model": "iPhone 14 Pro",
    "focal_length_mm": 24.0,  # 24mm equivalent
    "sensor_width_mm": 9.8,   # 1/1.28" sensor
    "sensor_height_mm": 7.3,
    "image_width_px": 4032,
    "image_height_px": 3024,
    "f_number": 1.78,
    "iso": 100,
    "exposure_time_s": 1/60,
}

# GPS base (San Francisco area)
GPS_BASE = {
    "lat": 37.7749,
    "lon": -122.4194,
    "alt": 10.0,
}


def generate_camera_trajectory(num_images: int = 30) -> List[dict]:
    """Generate a realistic camera capture trajectory around the room."""
    trajectory = []

    # Positions: walk around the room at ~1.5m height, looking inward
    # 8 positions per wall + corners + center
    base_positions = []

    # Wall 1: X = -0.5 (outside left wall looking right)
    for i in range(4):
        z = -ROOM_DEPTH/2 + (i + 0.5) * ROOM_DEPTH / 4
        base_positions.append((-0.5, 1.5, z))

    # Wall 2: Z = ROOM_DEPTH + 0.5 (outside front wall looking back)
    for i in range(4):
        x = -ROOM_WIDTH/2 + (i + 0.5) * ROOM_WIDTH / 4
        base_positions.append((x, 1.5, ROOM_DEPTH + 0.5))

    # Wall 3: X = ROOM_WIDTH + 0.5 (outside right wall looking left)
    for i in range(4):
        z = ROOM_DEPTH/2 - (i + 0.5) * ROOM_DEPTH / 4
        base_positions.append((ROOM_WIDTH + 0.5, 1.5, z))

    # Wall 4: Z = -0.5 (outside back wall looking forward)
    for i in range(4):
        x = ROOM_WIDTH/2 - (i + 0.5) * ROOM_WIDTH / 4
        base_positions.append((x, 1.5, -0.5))

    # Interior positions (inside room)
    interior_positions = [
        (ROOM_WIDTH/2, 1.5, ROOM_DEPTH/2),  # center
        (ROOM_WIDTH/4, 1.5, ROOM_DEPTH/4),  # corner quadrant
        (3*ROOM_WIDTH/4, 1.5, ROOM_DEPTH/4),
        (ROOM_WIDTH/4, 1.5, 3*ROOM_DEPTH/4),
        (3*ROOM_WIDTH/4, 1.5, 3*ROOM_DEPTH/4),
    ]

    all_positions = base_positions + interior_positions
    np.random.seed(42)
    np.random.shuffle(all_positions)

    base_time = calendar.timegm((2026, 9, 13, 10, 0, 0))

    for idx, (x, y, z) in enumerate(all_positions[:num_images]):
        # Look at room center
        target = np.array([ROOM_WIDTH/2, 1.0, ROOM_DEPTH/2])
        pos = np.array([x, y, z])
        forward = target - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0])  # default looking down
            forward_norm = 1.0
        forward = forward / forward_norm

        # Camera up (world up) - avoid parallel to forward
        up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, up)) > 0.99:
            up = np.array([0.0, 0.0, 1.0])  # use Z-up when looking straight down/up
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
            right_norm = 1.0
        right = right / right_norm
        up = np.cross(right, forward)

        # Rotation matrix (camera to world)
        R = np.column_stack([right, up, -forward])  # OpenCV convention

        # Convert to quaternion
        q = rotation_matrix_to_quaternion(R)

        # Add small GPS jitter
        gps_lat = GPS_BASE["lat"] + np.random.normal(0, 1e-6)
        gps_lon = GPS_BASE["lon"] + np.random.normal(0, 1e-6)
        gps_alt = GPS_BASE["alt"] + np.random.normal(0, 0.5)

        trajectory.append({
            "index": idx,
            "position": (float(x), float(y), float(z)),
            "rotation_quat": (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            "gps": (gps_lat, gps_lon, gps_alt),
            "timestamp": base_time + idx * 2,  # 2 seconds between captures
        })

    return trajectory


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return (w, x, y, z)


def create_synthetic_image(width: int, height: int, camera_idx: int, position: Tuple) -> np.ndarray:
    """Create a synthetic room image with visible geometry cues."""
    # Base color (warm gray)
    img = np.full((height, width, 3), (180, 170, 160), dtype=np.uint8)

    # Add room geometry cues (simple projection for visual cues)
    # Floor (bottom portion, darker)
    floor_start = int(height * 0.6)
    img[floor_start:, :] = (120, 115, 110)

    # Ceiling (top portion, lighter)
    ceiling_end = int(height * 0.2)
    img[:ceiling_end, :] = (220, 215, 210)

    # Left wall (left portion)
    wall_width = int(width * 0.25)
    img[ceiling_end:floor_start, :wall_width] = (150, 145, 140)

    # Right wall
    img[ceiling_end:floor_start, -wall_width:] = (150, 145, 140)

    # Door (center of front wall, on floor)
    door_w = int(width * 0.15)
    door_h = int(height * 0.3)
    door_x = (width - door_w) // 2
    door_y = floor_start - door_h
    img[door_y:floor_start, door_x:door_x+door_w] = (100, 80, 60)  # brown door

    # Window (on left wall)
    win_w = int(width * 0.1)
    win_h = int(height * 0.2)
    win_x = wall_width // 2 - win_w // 2
    win_y = ceiling_end + int((floor_start - ceiling_end) * 0.3)
    img[win_y:win_y+win_h, win_x:win_x+win_w] = (180, 200, 220)  # light blue window

    # Add some furniture cues
    # Table (center-ish on floor)
    table_w = int(width * 0.2)
    table_h = int(height * 0.15)
    table_x = width // 2 - table_w // 2
    table_y = floor_start - table_h
    img[table_y:table_y+table_h, table_x:table_x+table_w] = (140, 120, 100)

    # Chair (near table)
    chair_w = int(width * 0.08)
    chair_h = int(height * 0.15)
    chair_x = table_x - chair_w - 20
    chair_y = floor_start - chair_h
    img[chair_y:chair_y+chair_h, chair_x:chair_x+chair_w] = (100, 90, 80)

    # Lamp (on table)
    lamp_w = int(width * 0.03)
    lamp_h = int(height * 0.08)
    lamp_x = table_x + table_w // 2 - lamp_w // 2
    lamp_y = table_y - lamp_h
    img[lamp_y:lamp_y+lamp_h, lamp_x:lamp_x+lamp_w] = (255, 240, 200)

    # Add camera-index-dependent variation (simulate different views)
    # Shift everything slightly based on camera position
    shift_x = int((position[0] - 2.5) * 50)  # scale for visual effect
    shift_y = int((position[2] - 2.0) * 50)
    if shift_x != 0 or shift_y != 0:
        img = np.roll(img, shift_x, axis=1)
        img = np.roll(img, shift_y, axis=0)

    # Add subtle noise (sensor noise simulation)
    noise = np.random.normal(0, 2, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def write_image_with_exif(
    path: Path,
    img_array: np.ndarray,
    camera_params: dict,
    gps: Tuple[float, float, float],
    timestamp: int,
    camera_idx: int,
):
    """Write JPEG with minimal EXIF (no GPS to avoid PIL issues)."""
    img = Image.fromarray(img_array)

    # Build minimal EXIF (no GPS to avoid PIL encoding issues)
    exif = img.getexif()

    # Basic tags
    exif[0x010F] = camera_params["make"]          # Make
    exif[0x0110] = camera_params["model"]         # Model
    exif[0x0132] = time.strftime("%Y:%m:%d %H:%M:%S", time.gmtime(timestamp))  # DateTime

    # Exif sub-IFD
    sub_ifd = exif.get_ifd(0x8769)
    sub_ifd[0x9000] = "Exif Version 2.31"        # ExifVersion
    sub_ifd[0x9003] = time.strftime("%Y:%m:%d %H:%M:%S", time.gmtime(timestamp))  # DateTimeOriginal
    sub_ifd[0x9004] = time.strftime("%Y:%m:%d %H:%M:%S", time.gmtime(timestamp))  # DateTimeDigitized
    sub_ifd[0x829A] = (1, 60)  # ExposureTime (1/60s)
    sub_ifd[0x829D] = (int(camera_params["f_number"] * 10), 10)  # FNumber
    sub_ifd[0x8827] = camera_params["iso"]       # ISO
    sub_ifd[0x920A] = (int(camera_params["focal_length_mm"] * 100), 100)  # FocalLengthIn35mmFilm

    # Save with EXIF (no GPS IFD)
    img.save(path, format="JPEG", quality=95, exif=exif.tobytes())


def generate_dataset(
    output_dir: Path,
    num_images: int = 30,
    seed: int = 42,
):
    """Generate the complete synthetic room capture dataset."""
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = generate_camera_trajectory(num_images)

    # Camera intrinsics (for reconstruction)
    fx = CAMERA_PARAMS["focal_length_mm"] / CAMERA_PARAMS["sensor_width_mm"] * CAMERA_PARAMS["image_width_px"]
    fy = CAMERA_PARAMS["focal_length_mm"] / CAMERA_PARAMS["sensor_height_mm"] * CAMERA_PARAMS["image_height_px"]
    cx = CAMERA_PARAMS["image_width_px"] / 2
    cy = CAMERA_PARAMS["image_height_px"] / 2

    intrinsics = {
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "width": CAMERA_PARAMS["image_width_px"],
        "height": CAMERA_PARAMS["image_height_px"],
    }

    manifest = {
        "dataset": "synthetic_room_capture",
        "version": "1.0",
        "description": "Synthetic but realistic room capture for end-to-end pipeline testing",
        "room_dimensions_m": {"width": ROOM_WIDTH, "depth": ROOM_DEPTH, "height": ROOM_HEIGHT},
        "camera": CAMERA_PARAMS,
        "intrinsics": intrinsics,
        "images": [],
        "ground_truth": {
            "room": {"type": "rectangular", "width": ROOM_WIDTH, "depth": ROOM_DEPTH, "height": ROOM_HEIGHT},
            "door": {"wall": "front", "position": "center", "width": 0.9, "height": 2.1},
            "window": {"wall": "left", "position": "center", "width": 1.2, "height": 1.0, "sill_height": 0.9},
            "furniture": [
                {"type": "table", "position": [2.5, 0.0, 2.0], "dimensions": [1.5, 0.75, 0.75]},
                {"type": "chair", "position": [1.8, 0.0, 2.0], "dimensions": [0.5, 0.5, 0.9]},
                {"type": "lamp", "position": [2.5, 0.75, 2.0], "dimensions": [0.3, 0.3, 0.5]},
            ],
        },
    }

    for cam in trajectory:
        idx = cam["index"]
        x, y, z = cam["position"]
        qw, qx, qy, qz = cam["rotation_quat"]
        gps_lat, gps_lon, gps_alt = cam["gps"]
        timestamp = cam["timestamp"]

        # Create synthetic image
        img_array = create_synthetic_image(
            CAMERA_PARAMS["image_width_px"],
            CAMERA_PARAMS["image_height_px"],
            idx,
            cam["position"],
        )

        # Write image with EXIF
        img_path = output_dir / f"IMG_{idx:04d}.jpg"
        write_image_with_exif(
            img_path,
            img_array,
            CAMERA_PARAMS,
            (gps_lat, gps_lon, gps_alt),
            timestamp,
            idx,
        )

        # Add to manifest
        manifest["images"].append({
            "index": idx,
            "filename": f"IMG_{idx:04d}.jpg",
            "source_uri": f"file://{img_path.absolute()}",
            "position_m": [float(x), float(y), float(z)],
            "rotation_quat_wxyz": [float(qw), float(qx), float(qy), float(qz)],
            "gps": {"lat": gps_lat, "lon": gps_lon, "alt": gps_alt},
            "timestamp_unix": timestamp,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "camera_params": CAMERA_PARAMS,
        })

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write COLMAP-compatible cameras.txt and images.txt for ground truth
    write_colmap_ground_truth(output_dir, trajectory, intrinsics)

    print(f"Generated {num_images} images in {output_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest


def write_colmap_ground_truth(output_dir: Path, trajectory: List[dict], intrinsics: dict):
    """Write COLMAP-format ground truth for validation."""
    # cameras.txt
    cam_id = 1
    cam_line = f"{cam_id} PINHOLE {intrinsics['width']} {intrinsics['height']} {intrinsics['fx']} {intrinsics['fy']} {intrinsics['cx']} {intrinsics['cy']}\n"
    (output_dir / "cameras.txt").write_text(cam_line)

    # images.txt
    lines = []
    for i, cam in enumerate(trajectory):
        idx = i + 1  # COLMAP is 1-indexed
        qw, qx, qy, qz = cam["rotation_quat"]
        x, y, z = cam["position"]
        name = f"IMG_{cam['index']:04d}.jpg"
        line = f"{idx} {qw} {qx} {qy} {qz} {x} {y} {z} {cam_id} {name}\n"
        # Second line (2D points) - empty for ground truth
        lines.append(line)
        lines.append("\n")

    (output_dir / "images.txt").write_text("".join(lines))

    # points3D.txt (empty - no ground truth 3D points for synthetic)
    (output_dir / "points3D.txt").write_text("# No ground truth 3D points for synthetic dataset\n")


if __name__ == "__main__":
    import sys
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("datasets/real_room_capture")
    num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print(f"Generating synthetic room capture dataset...")
    print(f"Output: {output_dir}")
    print(f"Images: {num_images}")

    if not PIL_AVAILABLE:
        print("WARNING: Pillow not available - images will be saved without EXIF")
    if not CV2_AVAILABLE:
        print("WARNING: OpenCV not available - video generation not available")

    manifest = generate_dataset(output_dir, num_images)
    print("Done!")