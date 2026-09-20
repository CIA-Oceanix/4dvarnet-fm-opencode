#!/usr/bin/env bash
#
# sync-overleaf.sh -- two-way sync between this paper directory and an
# Overleaf project, using Overleaf's git bridge.
#
# The paper lives at docs/papers/p1_structural_hypotheses/ in this repo, but
# Overleaf needs main.tex at the PROJECT ROOT. `git subtree` bridges that:
# it rewrites this subdirectory's history as if it were its own repo.
#
#   ./sync-overleaf.sh init <overleaf-git-url>   # one-time: add the remote
#   ./sync-overleaf.sh push                      # repo  -> Overleaf
#   ./sync-overleaf.sh pull                      # Overleaf -> repo
#   ./sync-overleaf.sh status                    # what's configured
#
# The Overleaf git URL comes from the project's Menu -> Git, and looks like
#   https://git.overleaf.com/<project-id>
# Authentication is a git token (Overleaf Account Settings -> Git integration),
# used as the PASSWORD; any username works. Requires a paid Overleaf plan --
# the git bridge is not on the free tier. If you are on the free tier, use the
# zip route in README.md instead.
#
# NOTE: never put the token in the URL you pass to `init` -- it would be
# written into .git/config in plaintext. Let git prompt, and cache it with
#   git config --global credential.helper 'cache --timeout=86400'
#
set -euo pipefail

PREFIX="docs/papers/p1_structural_hypotheses"
REMOTE="overleaf"
BRANCH="master"   # Overleaf's git bridge uses master, not main

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

die() { echo "ERROR: $*" >&2; exit 1; }

require_clean() {
    # subtree push/pull rewrite history; a dirty tree makes failures hard to undo
    [ -z "$(git status --porcelain -- "$PREFIX")" ] \
        || die "uncommitted changes under $PREFIX -- commit them first"
}

case "${1:-}" in
    init)
        URL="${2:?usage: sync-overleaf.sh init <overleaf-git-url>}"
        case "$URL" in
            *@*) die "the URL appears to embed credentials; pass the bare https://git.overleaf.com/<id> URL" ;;
        esac
        git remote remove "$REMOTE" 2>/dev/null || true
        git remote add "$REMOTE" "$URL"
        echo "Added remote '$REMOTE' -> $URL"
        echo "Next: ./sync-overleaf.sh push"
        ;;
    push)
        git remote get-url "$REMOTE" >/dev/null 2>&1 || die "no '$REMOTE' remote; run: $0 init <url>"
        require_clean
        echo "Pushing $PREFIX -> $REMOTE/$BRANCH ..."
        git subtree push --prefix="$PREFIX" "$REMOTE" "$BRANCH"
        ;;
    pull)
        git remote get-url "$REMOTE" >/dev/null 2>&1 || die "no '$REMOTE' remote; run: $0 init <url>"
        require_clean
        echo "Pulling $REMOTE/$BRANCH -> $PREFIX ..."
        # --squash keeps this repo's history readable: one merge commit per
        # sync instead of every Overleaf autosave commit.
        git subtree pull --prefix="$PREFIX" "$REMOTE" "$BRANCH" --squash
        ;;
    status)
        echo "prefix: $PREFIX"
        echo -n "remote: "; git remote get-url "$REMOTE" 2>/dev/null || echo "(not configured)"
        echo -n "local changes under prefix: "
        [ -z "$(git status --porcelain -- "$PREFIX")" ] && echo "none" || git status --short -- "$PREFIX"
        ;;
    *)
        sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
