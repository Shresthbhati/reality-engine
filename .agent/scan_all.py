"""Read-only comprehensive scan of ALL Reality Engine worktrees.

Reports: branch, HEAD, dirty file count + sample, unpushed commits,
and commits ahead of origin/main (with subject). Never modifies anything.
"""
import subprocess
from pathlib import Path

SEARCH_ROOTS = [
    Path(r"C:\Users\shres\.cline\worktrees"),
    Path(r"C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.antigravity\worktrees"),
    Path(r"C:\Users\shres\OneDrive\Desktop\codes\reality-engine\.claude\worktrees"),
    Path(r"C:\Users\shres\.local\share\opencode\worktree\c779b1"),
    Path(r"C:\Users\shres\.local\share\kilo\worktree-pool"),
    Path(r"C:\Users\shres\AppData\Local\Temp\claude"),
    Path(r"C:\Users\shres\OneDrive\Desktop\codes"),
]


def run(args, cwd, timeout=25):
    try:
        r = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception as exc:
        return f"<err:{type(exc).__name__}>", 1


def main():
    seen = set()
    rows = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        # find any dir that is a git worktree (has .git file or dir)
        for wt in [root] + [p for p in root.rglob(".git") if p.is_file() or p.is_dir()]:
            base = wt.parent if wt.name == ".git" else wt
            key = str(base).lower()
            if key in seen:
                continue
            seen.add(key)
            if not (base / ".git").exists():
                continue
            status, _ = run(["git", "status", "--porcelain"], base)
            branch, _ = run(["git", "branch", "--show-current"], base)
            head, _ = run(["git", "rev-parse", "--short", "HEAD"], base)
            ahead_main, _ = run(["git", "rev-list", "--count", "origin/main..HEAD"], base)
            up, rc = run(["git", "rev-parse", "--abbrev-ref", "--verify", "@{upstream}"], base)
            unpushed = "?"
            if rc == 0 and up and not up.startswith("<err"):
                unpushed, _ = run(["git", "rev-list", "--count", f"{up}..HEAD"], base)
            dirty = [l for l in status.splitlines() if l.strip()]
            if not dirty and (ahead_main in ("0", "") or ahead_main.startswith("<")) and unpushed in ("0", ""):
                continue
            rows.append((str(base), branch, head, len(dirty), ahead_main, unpushed, dirty[:6]))

    for path, branch, head, nd, ahead, unp, sample in rows:
        tag = []
        if nd: tag.append(f"DIRTY:{nd}")
        if ahead not in ("0", "", "?") and not ahead.startswith("<"): tag.append(f"AHEAD_MAIN:{ahead}")
        if unp not in ("0", "", "?") and not unp.startswith("<"): tag.append(f"UNPUSHED:{unp}")
        if not tag: continue
        print(f"[{' '.join(tag)}] {branch:42} {head:9} {path}")
        if nd and sample:
            for l in sample:
                print("      " + l[:120])


if __name__ == "__main__":
    main()
