# Worktree Organization

## Current worktrees

| Worktree dir | Branch | Topic |
|---|---|---|
| `4dvarnet-fm-opencode` | `master` | Integration / PR landing |
| `4dvarnet-fm-joint-da` | `feature/l96-joint-da-benchmark` | Joint DA (ETKF) topic |
| `4dvarnet-fm-cfm-v2v3` | `feature/l96-v2v3-pure` | V2/V3 (TweedieCFM + PredictStateCFM) topic |

## Rules

1. One active topic = one dedicated `git worktree`.
2. A branch is never checked out in two worktrees simultaneously.
3. Commit before switching branches; never leave uncommitted edits when switching.
4. Topic branches fork from `origin/master`.

## Future expansions

Separate languages / case studies get their own worktrees (e.g. `4dvarnet-fm-qg`, `4dvarnet-fm-sw`).
