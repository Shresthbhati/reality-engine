"""Browser-level end-to-end test running built Reality Studio against real FastAPI server."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_http(url: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1.0) as resp:
                if resp.status < 500:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {url}")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_reality_studio_browser_e2e_against_real_api(tmp_path: Path) -> None:
    studio_dir = Path("apps/studio").resolve()
    if not (studio_dir / "dist" / "index.html").exists():
        pytest.skip("apps/studio/dist not built")

    img_path = tmp_path / "capture.jpg"
    arr = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(arr).save(img_path)

    api_port = _free_port()
    studio_port = _free_port()

    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{(tmp_path / 'studio_e2e.db').as_posix()}"
    env["WORLDSTORE_DIR"] = str(tmp_path / "worldstore")
    env["ARTIFACTS_DIR"] = str(tmp_path / "artifacts")
    env["VITE_API_ORIGIN"] = f"http://127.0.0.1:{api_port}"

    api_log = tmp_path / "api.log"
    vite_log = tmp_path / "vite.log"
    api_log_f = api_log.open("wb")
    vite_log_f = vite_log.open("wb")

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", str(api_port)],
        env=env,
        stdout=api_log_f,
        stderr=subprocess.STDOUT,
    )
    vite_proc = subprocess.Popen(
        ["npx", "vite", "preview", "--host", "127.0.0.1", "--port", str(studio_port), "--strictPort"],
        cwd=str(studio_dir),
        env=env,
        shell=(os.name == "nt"),
        stdout=vite_log_f,
        stderr=subprocess.STDOUT,
    )

    try:
        try:
            _wait_http(f"http://127.0.0.1:{api_port}/api/health", timeout_s=45.0)
        except TimeoutError as exc:
            api_log_f.flush()
            raise AssertionError(
                f"API failed to start: {exc}\nAPI output:\n{api_log.read_text(errors='replace')}"
            ) from exc
        _wait_http(f"http://127.0.0.1:{studio_port}/", timeout_s=20.0)

        e2e_env = env.copy()
        e2e_env["STUDIO_URL"] = f"http://127.0.0.1:{studio_port}/"
        e2e_env["SAMPLE_IMAGE_PATH"] = str(img_path)
        e2e_log = tmp_path / "e2e.log"
        with e2e_log.open("wb") as e2e_f:
            try:
                res = subprocess.run(
                    ["node", "e2e/studio.e2e.mjs"],
                    cwd=str(studio_dir),
                    env=e2e_env,
                    stdout=e2e_f,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
            except subprocess.TimeoutExpired as exc:
                raise AssertionError(
                    f"E2E timed out:\n{e2e_log.read_text(errors='replace')}"
                ) from exc
        e2e_out = e2e_log.read_text(errors="replace")
        assert res.returncode == 0, f"E2E failed:\n{e2e_out}"
        assert "STUDIO_BROWSER_E2E_OK" in e2e_out
    finally:
        vite_proc.terminate()
        api_proc.terminate()
        vite_proc.wait(timeout=5)
        api_proc.wait(timeout=5)
        api_log_f.close()
        vite_log_f.close()
