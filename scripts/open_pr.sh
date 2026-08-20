#!/usr/bin/env bash
#
# open_pr.sh — GitHub PR wrapper for the multi-agent review workflow (Option A).
#
# Turns each subtask into a real GitHub PR that a reviewer agent screens via
# gh pr review, gated by CI, then merged with gh pr merge.
#
# Each subtask maps to an implementer -> reviewer -> verifier sequence:
#
#   scripts/open_pr.sh create R4 "da_J=None assertion"        # push + open PR
#   scripts/open_pr.sh review <PR#>                           # reviewer approve/request
#   scripts/open_pr.sh verify <PR#>                           # wait for CI + merge
#
# Requires: gh authenticated (gh auth login) + write access to the repo.
#
# Reviewer identity (distinct from the implementer, required so a PR is not
# self-approved):
#   On a single GitHub account, GitHub blocks self-approval. To run the
#   reviewer automatically under a SECOND account, set REVIEWER_GH_TOKEN to
#   a PAT of that account (write scope). The `review` command then runs as
#   that account while create/verify run as the default (rfablet) account.
#
set -euo pipefail

MAIN_BRANCH="${MAIN_BRANCH:-feat/l96-fast-weights-randomization}"
REPO="CIA-Oceanix/4dvarnet-fm-opencode"
REMOTE="origin"
REVIEWER_GH_TOKEN="${REVIEWER_GH_TOKEN:-}"
REVIEWER_TOKEN_FILE="${REVIEWER_TOKEN_FILE:-${HOME}/.config/opencode/reviewer-token}"
if [ -z "$REVIEWER_GH_TOKEN" ] && [ -f "$REVIEWER_TOKEN_FILE" ]; then
    REVIEWER_GH_TOKEN="$(cat "$REVIEWER_TOKEN_FILE" | tr -d '[:space:]')"
fi

CMD="${1:?usage: open_pr.sh <create|review|verify> ...}"
shift

require_gh() {
    gh auth status >/dev/null 2>&1 || {
        echo "ERROR: gh not authenticated. Run: gh auth login" >&2
        exit 1
    }
}

case "$CMD" in
    create)
        STEP_ID="${1:?create: <STEP_ID> <description>}"
        DESC="${2:?create: <STEP_ID> <description>}"
        BRANCH="fix/$(printf '%s' "$STEP_ID" | tr '[:upper:]' '[:lower:]')-$(printf '%s' "$DESC" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
        BRANCH="$(printf '%s' "$BRANCH" | tr -cd '[:alnum:]_-' | cut -c1-80)"

        echo "=== IMPLEMENTER: pushing $BRANCH ==="
        git switch "$BRANCH" 2>/dev/null || { echo "No branch '$BRANCH'. Run agent_review_loop.sh $STEP_ID first."; exit 1; }
        git push -u "$REMOTE" "$BRANCH"

        echo "=== IMPLEMENTER: opening PR ==="
        gh pr create \
            --repo "$REPO" \
            --base "$MAIN_BRANCH" \
            --head "$BRANCH" \
            --title "$STEP_ID: $DESC" \
            --body "## What
$DESC

## Verification
- [ ] pytest fast passes (CI)
- [ ] reviewer approval required"
        echo "PR created. Review it:  scripts/open_pr.sh review <PR#>"
        ;;

    review)
        require_gh
        PR="${1:?review: <PR#> [approve|request] [message]}"
        DECISION="${2:-approve}"
        MESSAGE="${3:-}"
        echo "=== REVIEWER: PR #$PR ==="
        if [ -n "$REVIEWER_GH_TOKEN" ]; then
            REVIEWER_ACCOUNT=$(GH_TOKEN="$REVIEWER_GH_TOKEN" gh api user --jq '.login')
            echo "  reviewer account: ${REVIEWER_ACCOUNT:-<invalid REVIEWER_GH_TOKEN>}"
        else
            echo "  reviewer account: <default gh account> (NOTE: self-approval is blocked on one account)"
        fi
        echo ""
        gh pr diff "$PR" --repo "$REPO"
        echo ""
        # Reviewer identity: approve/request-changes run as the reviewer token.
        # gh authenticates via GH_TOKEN (NOT REVIEWER_GH_TOKEN), so we must set
        # GH_TOKEN when acting as the reviewer account.
        if [ -n "$REVIEWER_GH_TOKEN" ]; then
            if [ "$DECISION" = "approve" ]; then
                GH_TOKEN="$REVIEWER_GH_TOKEN" gh pr review "$PR" --repo "$REPO" --approve ${MESSAGE:+--body "$MESSAGE"}
            else
                GH_TOKEN="$REVIEWER_GH_TOKEN" gh pr review "$PR" --repo "$REPO" --request-changes ${MESSAGE:+--body "$MESSAGE"}
            fi
        else
            if [ "$DECISION" = "approve" ]; then
                gh pr review "$PR" --repo "$REPO" --approve ${MESSAGE:+--body "$MESSAGE"}
            else
                gh pr review "$PR" --repo "$REPO" --request-changes ${MESSAGE:+--body "$MESSAGE"}
            fi
        fi
        echo "Review submitted as ${REVIEWER_ACCOUNT:-default account}."
        ;;

    verify)
        require_gh
        PR="${1:?verify: <PR#>}"
        echo "=== VERIFIER: waiting for CI on PR #$PR ==="
        # The watcher exits non-zero if ANY check fails (incl. informational
        # ruff, which is continue-on-error). That must not abort the merge.
        gh pr checks "$PR" --repo "$REPO" --watch --interval 20 || true
        echo ""
        # Confirm the ruleset gate (approval + required checks) is satisfied
        # before merging. The ruleset itself enforces pytest + 1 review.
        MERGEABLE=$(gh pr view "$PR" --repo "$REPO" --json mergeable --jq '.mergeable')
        MSTATE=$(gh pr view "$PR" --repo "$REPO" --json mergeStateStatus --jq '.mergeStateStatus')
        echo "=== VERIFIER: mergeable=$MERGEABLE mergeState=$MSTATE ==="
        if [ "$MERGEABLE" != "MERGEABLE" ] || [ "$MSTATE" = "BLOCKED" ] || [ "$MSTATE" = "DIRTY" ]; then
            echo "ERROR: PR #$PR not cleanly mergeable (mergeable=$MERGEABLE, mergeState=$MSTATE). Aborting." >&2
            exit 1
        fi
        echo "=== VERIFIER: merging PR #$PR ==="
        # Note: this gh version rejects --yes; --squash --delete-branch already
        # performs the merge non-interactively.
        gh pr merge "$PR" --repo "$REPO" --squash --delete-branch
        echo "Merged."
        ;;

    *)
        echo "Usage: open_pr.sh <create|review|verify> ..." >&2
        exit 1
        ;;
esac
