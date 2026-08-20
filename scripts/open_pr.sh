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
set -euo pipefail

MAIN_BRANCH="${MAIN_BRANCH:-feat/l96-fast-weights-randomization}"
REPO="CIA-Oceanix/4dvarnet-fm-opencode"
REMOTE="origin"

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
        echo "=== REVIEWER: PR #$PR diff ==="
        gh pr view "$PR" --repo "$REPO"
        echo ""
        gh pr diff "$PR" --repo "$REPO"
        echo ""
        if [ "$DECISION" = "approve" ]; then
            gh pr review "$PR" --repo "$REPO" --approve ${MESSAGE:+--body "$MESSAGE"}
            echo "Approved."
        else
            gh pr review "$PR" --repo "$REPO" --request-changes ${MESSAGE:+--body "$MESSAGE"}
            echo "Requested changes. Implementer must push fixes, then re-review."
        fi
        ;;

    verify)
        require_gh
        PR="${1:?verify: <PR#>}"
        echo "=== VERIFIER: waiting for CI on PR #$PR ==="
        gh pr checks "$PR" --repo "$REPO" --watch --interval 20
        echo ""
        echo "=== VERIFIER: merging PR #$PR ==="
        gh pr merge "$PR" --repo "$REPO" --squash --delete-branch --yes
        echo "Merged."
        ;;

    *)
        echo "Usage: open_pr.sh <create|review|verify> ..." >&2
        exit 1
        ;;
esac
