#!/usr/bin/env bash
#
# agent_review_loop.sh — local multi-agent review loop (no GitHub required)
#
# Mirrors the implementer -> reviewer -> verifier cycle using local git only.
# Each step creates a feature branch, the reviewer gates a local diff, the
# verifier runs lint+tests, and a squash merge lands the change.
#
# Usage:
#   scripts/agent_review_loop.sh <STEP_ID> <description>            # create branch
#   scripts/agent_review_loop.sh <STEP_ID> <description> --review   # commit+review+verify+merge
#
# Example:
#   scripts/agent_review_loop.sh R4 "da_J=None assertion"
#   ... make changes on the branch ...
#   scripts/agent_review_loop.sh R4 "da_J=None assertion" --review
#
set -euo pipefail

STEP_ID="${1:?Usage: agent_review_loop.sh <STEP_ID> <desc> [--review]}"
DESC="${2:?missing description}"
MODE="${3:-}"

MAIN_BRANCH="${MAIN_BRANCH:-feat/l96-fast-weights-randomization}"
BRANCH="fix/$(printf '%s' "$STEP_ID" | tr '[:upper:]' '[:lower:]')-$(printf '%s' "$DESC" | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
BRANCH="$(printf '%s' "$BRANCH" | tr -cd '[:alnum:]_-' | cut -c1-80)"

# Python env that has the project deps (torch, hydra, lightning). The base
# miniforge env's pytest cannot import torch, so default to the fdv env.
PYTHON_BIN="${PYTHON_BIN:-/Odyssey/private/rfablet/miniforge3/envs/fdv/bin/python}"

if [ "$MODE" != "--review" ]; then
    git switch "$MAIN_BRANCH"
    git switch -c "$BRANCH"
    echo "=== IMPLEMENTER ==="
    echo "Branch: $BRANCH"
    echo "Make your changes now, then re-run with --review:"
    echo "  scripts/agent_review_loop.sh $STEP_ID \"$DESC\" --review"
    exit 0
fi

echo "=== IMPLEMENTER: committing changes ==="
git add -A
git commit -m "$STEP_ID: $DESC"

echo ""
echo "=== REVIEWER: reviewing diff ==="
git diff "$MAIN_BRANCH..$BRANCH" --stat
echo ""
git diff "$MAIN_BRANCH..$BRANCH"

echo ""
read -r -p "Approve? (y/n): " APPROVE
if [ "$APPROVE" != "y" ]; then
    echo "Rejected by reviewer. Fix on branch, then re-run --review."
    exit 1
fi

echo ""
echo "=== VERIFIER: running checks ==="
if command -v ruff >/dev/null 2>&1; then
    ruff check . && echo "  ruff: PASS"
else
    echo "  ruff: SKIPPED (not installed)"
fi
if [ -x "$PYTHON_BIN" ]; then
    "$PYTHON_BIN" -m pytest tests/test_lorenz96_training.py -m "not slow" -q >/dev/null 2>&1 \
        && echo "  pytest: PASS" \
        || { echo "  pytest: FAIL"; exit 1; }
else
    echo "  pytest: SKIPPED (PYTHON_BIN not found: $PYTHON_BIN)"
fi

echo ""
echo "=== MERGING ==="
git switch "$MAIN_BRANCH"
git merge --squash "$BRANCH"
git commit -m "$STEP_ID: $DESC"
git branch -d "$BRANCH"
echo "Done! $STEP_ID merged into $MAIN_BRANCH"
