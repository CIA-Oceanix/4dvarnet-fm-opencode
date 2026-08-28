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

### Session bootstrap

When starting a new opencode session in a topic worktree:

1. Start from the topic directory:
   ```bash
   # Joint-DA topic:
   cd .../4dvarnet-fm-opencode/4dvarnet-fm-joint-da
   opencode

   # V2/V3 topic:
   cd .../4dvarnet-fm-opencode/4dvarnet-fm-cfm-v2v3
   opencode
   ```

2. If this worktree's docs (AGENTS.md/PLAN.md/CHANGELOG.md) are outdated, sync from master:
   ```bash
   git pull origin master --dry-run  # quick check
   git pull origin master             # if pull updates docs
   ```

3. Start the session — opencode will auto-read the local AGENTS.md/PLAN.md.
