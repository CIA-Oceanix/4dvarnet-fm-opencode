#!/usr/bin/env bash
#
# sync-overleaf.sh -- two-way sync between this paper directory and an
# Overleaf project, over Overleaf's git bridge.
#
#   ./sync-overleaf.sh init <overleaf-git-url>   # one-time: add the remote
#   ./sync-overleaf.sh push                      # repo  -> Overleaf
#   ./sync-overleaf.sh pull                      # Overleaf -> repo
#   ./sync-overleaf.sh status                    # what is configured, and drift
#
# ---------------------------------------------------------------------------
# Two facts about the Overleaf git bridge drive this script's design; both were
# established the hard way against a live project on 2026-09-20.
#
#   1. The branch is `main`, not `master`.
#
#   2. Overleaf REFUSES force pushes, server-side:
#        "remote: hint: You can't git push --force to a Overleaf project."
#      This rules out `git subtree push`. A subtree split has no ancestor in
#      common with whatever Overleaf already has (a fresh project ships with a
#      starter main.tex), so subtree push can only land via a force -- which is
#      rejected -- or a merge Overleaf has no way to make.
#
# So `push` does not push the split history at all. It takes the split's TREE,
# commits it with Overleaf's current head as parent, and pushes that: an
# ordinary fast-forward, always legal, and it survives Overleaf autosaves that
# land between syncs.
#
# The paper is a subdirectory here but must be the project ROOT on Overleaf;
# `git subtree split` is what bridges that.
# ---------------------------------------------------------------------------
#
# Auth: a git token (Overleaf -> Account Settings -> Git integration) in a 0600
# file, read by scripts/overleaf-askpass.sh. Never put it in the remote URL --
# that writes it into .git/config in plaintext, and `init` rejects such a URL.
#
#   umask 077 && printf '%s' '<token>' > ~/.config/overleaf-token
#
set -euo pipefail

PREFIX="docs/papers/p1_structural_hypotheses"
REMOTE="overleaf"
BRANCH="${OVERLEAF_BRANCH:-main}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

ASKPASS="$REPO_ROOT/scripts/overleaf-askpass.sh"
[ -x "$ASKPASS" ] && export GIT_ASKPASS="$ASKPASS"

die() { echo "ERROR: $*" >&2; exit 1; }

have_remote() {
    git remote get-url "$REMOTE" >/dev/null 2>&1 || die "no '$REMOTE' remote; run: $0 init <url>"
}

require_clean() {
    [ -z "$(git status --porcelain -- "$PREFIX")" ] \
        || die "uncommitted changes under $PREFIX -- commit them first"
}

fetch_remote() {
    git fetch -q "$REMOTE" "+refs/heads/$BRANCH:refs/remotes/$REMOTE/$BRANCH" 2>/dev/null || true
}

split_sha() { git subtree split --prefix="$PREFIX" HEAD 2>/dev/null | tail -1; }

case "${1:-}" in
    init)
        URL="${2:?usage: sync-overleaf.sh init <overleaf-git-url>}"
        case "$URL" in
            *@*) die "that URL appears to embed credentials; pass the bare https://git.overleaf.com/<id> form and let the askpass helper supply the token" ;;
        esac
        git remote remove "$REMOTE" 2>/dev/null || true
        git remote add "$REMOTE" "$URL"
        echo "Added remote '$REMOTE' -> $URL"
        echo "Next: $0 push"
        ;;

    push)
        have_remote; require_clean; fetch_remote
        SPLIT="$(split_sha)"
        [ -n "$SPLIT" ] || die "subtree split produced nothing for $PREFIX"
        TREE="$(git rev-parse "$SPLIT^{tree}")"

        if git rev-parse --verify -q "refs/remotes/$REMOTE/$BRANCH" >/dev/null; then
            PARENT="$(git rev-parse "refs/remotes/$REMOTE/$BRANCH")"
            if [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then
                echo "Overleaf already matches $PREFIX at $(git rev-parse --short HEAD) -- nothing to push."
                exit 0
            fi
            NEW="$(git commit-tree "$TREE" -p "$PARENT" \
                -m "Sync from $(basename "$REPO_ROOT") @ $(git rev-parse --short HEAD)

Source: $PREFIX")"
        else
            echo "Remote branch '$BRANCH' does not exist yet; pushing an initial commit."
            NEW="$(git commit-tree "$TREE" -m "Import from $(basename "$REPO_ROOT") @ $(git rev-parse --short HEAD)")"
        fi

        echo "Pushing $PREFIX ($(git rev-parse --short HEAD)) -> $REMOTE/$BRANCH ..."
        git push "$REMOTE" "$NEW:refs/heads/$BRANCH"
        git update-ref "refs/remotes/$REMOTE/$BRANCH" "$NEW"
        echo "Done."
        ;;

    pull)
        have_remote; require_clean; fetch_remote
        git rev-parse --verify -q "refs/remotes/$REMOTE/$BRANCH" >/dev/null \
            || die "remote branch '$BRANCH' not found -- has anything been pushed yet?"
        REMOTE_TREE="$(git rev-parse "refs/remotes/$REMOTE/$BRANCH^{tree}")"
        LOCAL_TREE="$(git rev-parse "$(split_sha)^{tree}")"
        if [ "$REMOTE_TREE" = "$LOCAL_TREE" ]; then
            echo "$PREFIX already matches Overleaf -- nothing to pull."
            exit 0
        fi
        echo "Overleaf differs from $PREFIX. Applying its contents to the working tree:"
        git diff --stat "$LOCAL_TREE" "$REMOTE_TREE" | sed 's/^/  /'
        # Check out the remote tree INTO the prefix. Deliberately not
        # `git subtree pull`: that merges Overleaf's unrelated history into this
        # repo, and Overleaf's history is one autosave commit per keystroke-ish.
        # We want its content, not its log.
        git rm -rq --cached "$PREFIX" >/dev/null
        rm -rf "${REPO_ROOT:?}/$PREFIX"
        git read-tree --prefix="$PREFIX/" "$REMOTE_TREE"
        git checkout -- "$PREFIX"
        echo
        echo "Working tree updated. Review with 'git diff --cached -- $PREFIX', then commit."
        ;;

    status)
        echo "prefix: $PREFIX"
        echo "branch: $BRANCH"
        echo -n "remote: "; git remote get-url "$REMOTE" 2>/dev/null || echo "(not configured)"
        echo -n "token:  "
        TF="${OVERLEAF_TOKEN_FILE:-$HOME/.config/overleaf-token}"
        [ -r "$TF" ] && echo "$TF (present)" || echo "$TF (MISSING)"
        echo -n "local changes under prefix: "
        [ -z "$(git status --porcelain -- "$PREFIX")" ] && echo "none" || { echo; git status --short -- "$PREFIX"; }
        if git remote get-url "$REMOTE" >/dev/null 2>&1; then
            fetch_remote
            if git rev-parse --verify -q "refs/remotes/$REMOTE/$BRANCH" >/dev/null; then
                R="$(git rev-parse "refs/remotes/$REMOTE/$BRANCH^{tree}")"
                L="$(git rev-parse "$(split_sha)^{tree}" 2>/dev/null || echo none)"
                [ "$R" = "$L" ] && echo "drift:  none (Overleaf matches $PREFIX)" \
                                || echo "drift:  DIFFERS -- run push or pull"
            else
                echo "drift:  remote branch '$BRANCH' not found"
            fi
        fi
        ;;

    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
