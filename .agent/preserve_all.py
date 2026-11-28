"""Preserve all uncommitted work from every Reality Engine worktree.

For each worktree with dirty files:
  - create branch preserve/<slug> from that worktree's HEAD
  - copy all dirty tracked + untracked files (excluding ephemeral junk) into it
  - commit + push

NEVER modifies the source worktree. Only reads it and writes new refs in the
shared git object database via the current worktree.
"""
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(r"C:\Users\shres\.local\share\opencode\worktree\c779b1\clever-panda")
SCAN_ROOTS = [
    Path(r"C:\Users\shres\.cline\worktrees"),
    Path(r"C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.antigravity\worktrees"),
    Path(r"C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.claude\worktrees"),
    Path(r"C:\Users\shres\.local\share\opencode\worktree\c779b1"),
    Path(r"C:\Users\shres\OneDrive\Desktop\codes"),
]
# ephemeral / non-code junk not worth preserving as implementation
SKIP = re.compile(
    r"(\.pyc$|\.log$|/__pycache__/|^MagicMock/|sqlite-shm$|sqlite-wal$|"
    r"\.txt$|\.json\.partial$|\.err$|baseline_pytest|push_err|push_out|_suite_)",
    re.I,
)


def run(args, cwd=None, timeout=60):
    c = cwd or REPO
    r = subprocess.run(args, cwd=str(c), capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def worktrees():
    out, _, _ = run(["git", "worktree", "list", "--porcelain"])
    result, cur = [], {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            if cur: result.append(cur)
            cur = {"path": Path(line.split(" ", 1)[1])}
        elif line.startswith("HEAD "):
            cur["head"] = line.split(" ", 1)[1]
        elif line.startswith("branch "):
            cur["branch"] = line.split(" ", 1)[1].replace("refs/heads/", "")
    if cur: result.append(cur)
    return result


def slug(wt):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(wt["path"])).strip("-").lower()
    return "preserve/" + s[-60:]


def main():
    report = []
    for wt in worktrees():
        p = wt["path"]
        if not (p / ".git").exists() or p == REPO:
            continue
        status, _, _ = run(["git", "status", "--porcelain", "-z"], p)
        entries = [e for e in status.split("\0") if e.strip()]
        files = []
        for e in entries:
            code, name = e[:2], e[3:]
            if SKIP.search(name):
                continue
            files.append((code, name))
        if not files:
            continue
        head = wt.get("head", "")
        br = slug(wt)
        # create branch from this worktree's HEAD in shared object db
        _, err, rc = run(["git", "branch", "-f", br, head])
        if rc != 0:
            report.append((str(p), "BRANCH_FAIL", err)); continue
        run(["git", "checkout", "-q", br])
        ok = 0
        for code, name in files:
            src = p / name
            dst = REPO / name
            try:
                if "D" in code:
                    if dst.exists(): dst.unlink()
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_dir():
                        if dst.exists(): shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
                ok += 1
            except Exception as exc:
                report.append((str(p), f"COPY_FAIL {name}", str(exc)))
        run(["git", "add", "-A"])
        msg = f"preserve: uncommitted work from {wt.get('branch','?')} @ {head[:8]} ({ok} paths)"
        _, err, rc = run(["git", "commit", "-q", "-m", msg])
        if rc != 0:
            report.append((str(p), "COMMIT_FAIL", err)); continue
        _, err, rc = run(["git", "push", "-q", "-u", "origin", br])
        report.append((str(p), f"PRESERVED {ok} paths -> {br}", err if rc else "pushed"))
        run(["git", "checkout", "-q", "-"])
    for r in report:
        print(r[0])
        print("   " + r[1] + ((" | " + r[2][:100]) if r[2] else ""))


if __name__ == "__main__":
    main()
