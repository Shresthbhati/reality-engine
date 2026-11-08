# Branch audit (plan task 13.1)

Date: 2026-11-07 · Base: `origin/main` · Repo: Shresthbhati/reality-engine · Method: `git rev-list origin/main..origin/<branch>` plus PR state from `gh pr list`.

| Group | Count | Meaning |
|-------|-------|---------|
| Remote branches (excluding `main`) | 127 | — |
| Zero commits not in `main` | 108 | Nothing unique on them. Listed in [branch_delete_candidates.txt](branch_delete_candidates.txt). |
| Merged PR but commits ahead | 19 | Almost certainly squash merges (commit hashes differ). Review before deleting; list below. |
| No PR, zero commits ahead | 2 | `claude/reality-engine-ml-reconstruction-c6a6fe`, `dev/universal-e2e-and-studio-integration` — included in the 108. |
| Open PRs | 0 at audit time | — |

`main` branch protection: **off** (`protected: false`). Enabling it is a repository-owner action.

## Cleanup completed (2026-11-08)

All 108 zero-commit candidate branches listed in [branch_delete_candidates.txt](branch_delete_candidates.txt) were safely deleted from `origin` and pruned from remote tracking references.

## Review before deleting (commits ahead of `main`, PR already merged)

Includes `claude/vram-release-ca4968`, the active worktree for this work. Count is commits not reachable from `main`; for squash merges this is expected and does not by itself mean lost work.

- `claude/frontend-audit-11f0f4` (1)
- `claude/integration-2026-11-05` (35)
- `claude/p0-no-fake-backend` (37)
- `claude/p7-04-evidence-quality` (3)
- `claude/reality-engine-audit-0a07f0` (1)
- `claude/reality-engine-ui-research-b1087b` (1)
- `claude/tenant-isolation-9a47d9` (1)
- `claude/verification-checklist-c8ccb9` (1)
- `claude/vram-release-ca4968` (2)
- `docs-canon` (1)
- `docs/image-to-world-lifecycle` (1)
- `feat/rules-11-12-universal-spatial-intelligence-pipeline` (3)
- `feat/scene-complete-eval-8-6` (4)
- `feat/task-11-2-world-package-zip` (24)
- `feat/task-4-1-model-registry-and-licenses` (1)
- `fix/exif-orientation-pixel-cap` (1)
- `fix/p0-multiworker-production` (1)
- `fix/sam-load-path` (1)
- `trajectory` (1)

To check one branch for post-merge work: `git log origin/main..origin/<branch> --oneline` and compare against its merged PR.

## Owner actions still needed (cannot be done from a branch)

1. Review and run the delete command above (or delete in the GitHub UI).
2. Enable protection on `main`: require pull request, require the CI checks `Core Tests`, `Frontend`, and the fidelity gate, block force-push.
3. Turn on "Automatically delete head branches" in repo settings so merged branches do not pile up again.
