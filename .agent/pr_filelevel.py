"""PR remaining diverged preserve branches via file-level checkout onto fresh main."""
import re
import subprocess

REPO = r"C:\Users\shres\.local\share\opencode\worktree\c779b1\clever-panda"


def run(args, timeout=90):
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


DIV = [
    "preserve/c-users-shres-.cline-worktrees-216df-reality-engine",
    "preserve/des-reality-engine-.claude-worktrees-agent-acc7cb9b5cad631f3",
    "preserve/p-codes-reality-engine-.claude-worktrees-vram-release-ca4968",
    "preserve/p-codes-reality-engine-.claude-worktrees-worldir-corrections",
    "preserve/ty-engine-.claude-worktrees-reality-engine-1-0-sprint-e5b98f",
    "preserve/lity-engine-.claude-worktrees-reality-engine-sprint-5-b377bf",
]


def main():
    made = []
    for src in DIV:
        tip, _, _ = run(["git", "rev-parse", "origin/" + src])
        files, _, _ = run(["git", "diff-tree", "--no-commit-id", "--name-only", "-r", tip])
        code = [f for f in files.splitlines() if re.search(r"\.(py|ts|tsx)$", f)]
        if not code:
            continue
        newbr = "preserve-files/" + src.replace("preserve/", "")[:50]
        run(["git", "checkout", "-q", "-B", newbr, "origin/main"])
        # take the preserved version of each code file (and any md/yaml/json in same commit)
        allf = [f for f in files.splitlines() if f.strip() and
                re.search(r"\.(py|ts|tsx|md|yaml|yml|json|sql)$", f)]
        ok = 0
        for f in allf:
            out, _, rc = run(["git", "checkout", tip, "--", f])
            if rc == 0:
                ok += 1
        if ok == 0:
            continue
        run(["git", "add", "-A"])
        _, err, rc = run(["git", "commit", "-q", "-m",
                          f"preserve(files): {src[:60]} @ {tip[:8]} ({ok} paths onto current main)"])
        if rc != 0:
            continue
        run(["git", "push", "-q", "-f", "origin", newbr])
        title = "Preserve (file-level): " + src.replace("preserve/", "")[:55]
        body = (
            "## Preservation PR (file-level onto current main)\n\n"
            f"Source: `{src}` — uncommitted work from an agent worktree. The original base "
            "was old/divergent, so this PR takes the preserved versions of the changed files "
            "onto current `origin/main` (no history reversion). Files: " + str(ok) + ".\n\n"
            "**Honest status:** preservation snapshot. NOT a claim of integration/correctness/"
            "passing tests. These files may overwrite newer main versions of the same paths — "
            "review carefully before merge. Source worktrees untouched; no merge by me."
        )
        o, e, rc4 = run(["gh", "pr", "create", "--base", "main", "--head", newbr,
                         "--title", title, "--body", body])
        made.append((src, newbr, ok, o.strip().splitlines()[-1] if o.strip() else e[:60]))
    run(["git", "checkout", "-q", "origin/main"])
    print("CREATED:", len(made))
    for s, b, n, u in made: print(" ", s[:48], b[:44], n, u)


if __name__ == "__main__":
    main()
