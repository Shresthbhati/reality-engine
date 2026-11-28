"""Where is COLMAP? One answer for the whole repo.

Order (first existing file wins; every miss is recorded so "not found" can say exactly where it looked):

    1. the caller's explicit path (``colmap_binary`` that is a real file / has a directory part)
    2. env ``REALITY_COLMAP_PATH`` then ``COLMAP_PATH`` (a file, or a directory holding ``colmap`` / ``bin/colmap``)
    3. the persisted setting ``colmap_path`` (``settings.json``, see :func:`settings_file`; set from the Settings page)
    4. ``PATH``
    5. common install directories (Windows ``Program Files``/``tools``/scoop, ``/usr/local``, ``/opt``, Homebrew)

Nothing here runs COLMAP; "found" means a file exists, not that it works (callers probe capabilities separately).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ENV_VARS = ("REALITY_COLMAP_PATH", "COLMAP_PATH")
_EXE = "colmap.exe" if os.name == "nt" else "colmap"


@dataclass(frozen=True)
class ColmapLocation:
    path: Optional[str]
    source: str  # explicit | env:<NAME> | settings | PATH | common_dir | not_found
    tried: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None

    def to_dict(self) -> dict:
        return {"found": self.found, "path": self.path, "source": self.source, "tried": list(self.tried),
                "warnings": list(self.warnings)}


def settings_file() -> Path:
    """``$REALITY_SETTINGS_FILE``, else ``~/.reality-engine/settings.json``."""
    env = os.environ.get("REALITY_SETTINGS_FILE")
    return Path(env) if env else Path.home() / ".reality-engine" / "settings.json"


def read_settings() -> dict:
    try:
        data = json.loads(settings_file().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_settings(update: dict) -> dict:
    """Merge ``update`` into the settings file (a ``None`` value removes the key). Returns the new settings."""
    cur = read_settings()
    for k, v in update.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    p = settings_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return cur


def as_binary(candidate: str) -> Optional[str]:
    """``candidate`` as a colmap executable: a file, or a dir with ``colmap`` / ``bin/colmap``."""
    p = Path(os.path.expandvars(os.path.expanduser(candidate)))
    if p.is_file():
        return str(p)
    if p.is_dir():
        for sub in (p / _EXE, p / "bin" / _EXE):
            if sub.is_file():
                return str(sub)
    return None


def _common_dirs() -> List[str]:
    home = str(Path.home())
    pats: List[str] = []
    if os.name == "nt":
        for base in filter(None, (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"),
                                  os.environ.get("LOCALAPPDATA"), "C:\\", "D:\\")):
            pats += [os.path.join(base, "COLMAP*"), os.path.join(base, "colmap*"),
                     os.path.join(base, "Programs", "COLMAP*"), os.path.join(base, "tools", "colmap*")]
        pats += [os.path.join(home, "tools", "colmap*"), os.path.join(home, "tools", "colmap*", "*"),
                 os.path.join(home, "scoop", "apps", "colmap", "current"),
                 os.path.join(home, "Downloads", "colmap*"), os.path.join(home, "Desktop", "colmap*")]
    else:
        pats += ["/usr/local/bin", "/usr/bin", "/opt/colmap", "/opt/homebrew/bin", "/opt/local/bin",
                 os.path.join(home, ".local", "bin"), os.path.join(home, "colmap*")]
    out: List[str] = []
    for pat in pats:
        out += sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
    return out


def locate_colmap(binary: str = "colmap") -> ColmapLocation:
    tried: List[str] = []
    warnings: List[str] = []

    def hit(path: Optional[str], source: str) -> Optional[ColmapLocation]:
        return ColmapLocation(path, source, tried, warnings) if path else None

    has_dir = os.path.dirname(binary) != "" or os.path.isabs(binary)
    if has_dir:
        tried.append(f"explicit: {binary}")
        r = hit(as_binary(binary), "explicit")
        if r:
            return r
        warnings.append(f"configured colmap binary {binary!r} does not exist")
    for name in ENV_VARS:
        val = os.environ.get(name)
        if val:
            tried.append(f"env {name}={val}")
            r = hit(as_binary(val), f"env:{name}")
            if r:
                return r
            warnings.append(f"{name}={val!r} does not exist (ignored)")
    saved = read_settings().get("colmap_path")
    if saved:
        tried.append(f"settings colmap_path={saved}")
        r = hit(as_binary(str(saved)), "settings")
        if r:
            return r
        warnings.append(f"saved colmap_path {saved!r} does not exist (ignored)")
    if not has_dir:
        tried.append(f"PATH ({binary})")
        r = hit(shutil.which(binary), "PATH")
        if r:
            return r
    for d in _common_dirs():
        b = as_binary(d)
        if b:
            tried.append(f"common dir {d}")
            return ColmapLocation(b, "common_dir", tried, warnings)
    tried.append("common install directories (Program Files, ~/tools, /usr/local, /opt, Homebrew, scoop)")
    return ColmapLocation(None, "not_found", tried, warnings)


def which(binary: str) -> Optional[str]:
    """Drop-in for ``shutil.which`` that understands COLMAP's extra search locations; other binaries pass through."""
    stem = Path(binary).stem.lower()
    if stem == "colmap":
        return locate_colmap(binary).path
    return shutil.which(binary)


def not_found_message(binary: str = "colmap") -> str:
    loc = locate_colmap(binary)
    return ("COLMAP was not found. Looked in: " + "; ".join(loc.tried)
            + ". Install COLMAP and set REALITY_COLMAP_PATH (or Settings -> Reconstruction tools) to colmap.exe.")
