"""Read-only: categorize pushed preserve/* branches by real content vs junk."""
import re
import subprocess

REPO = r"C:\Users\shres\.local\share\opencode\worktree\c779b1\clever-panda"

CODE = re.compile(r"\.(py|ts|tsx|js|jsx|json|md|yaml|yml|toml|txt|sql|html|css|sh|bat|ini|cfg)$", re.I)
JUNK_DIR = re.compile(r"(^|/)(MagicMock|__pycache__|\.codex|\.serena|\.claude/data|--out|build/)", re.I)
JUNK_FILE = re.compile(r"(\.log$|\.err$|^_suite|^baseline_pytest|^push_err|^push_out|^0$|sqlite-shm$|sqlite-wal$)", re.I)


def run(args):
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


branches = [b.strip() for b in run(["git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/preserve/"]).splitlines() if b.strip()]
print(f"pushed preserve branches: {len(branches)}")
for br in branches:
    short = br.replace("origin/", "")
    files = [f for f in run(["git", "diff", "--name-only", "origin/main...origin/" + short]).splitlines() if f.strip()]
    code = [f for f in files if CODE.search(f) and not JUNK_DIR.search(f) and not JUNK_FILE.search(f)]
    parent = run(["git", "rev-parse", "origin/" + short + "^"])
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", parent, "origin/main"], cwd=REPO).returncode
    print(f"{short[:70]:70} files={len(files):3} code={len(code):3} parent_in_main={'Y' if ancestor==0 else 'N'}")
