"""Writes small, deterministic local media for the local-import e2e: <dir>/photos/*.jpg and <dir>/walk.mp4.

The pictures are unrelated random textures on purpose: they decode and pass the photo checks, but cannot be
registered into one scene -- the e2e asserts the UI reports THAT honestly instead of showing an empty state.
"""
import sys
from pathlib import Path

import cv2
import numpy as np


def textured(i, w=320, h=240):
    rng = np.random.default_rng(i)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 1.4)
    return cv2.putText(img, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 4)


out = Path(sys.argv[1])
(out / "photos").mkdir(parents=True, exist_ok=True)
for i in range(4):
    cv2.imwrite(str(out / "photos" / f"shot{i}.jpg"), textured(i))
vw = cv2.VideoWriter(str(out / "walk.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
for i in range(30):
    vw.write(textured(10 + i // 3))
vw.release()
print("ok")
