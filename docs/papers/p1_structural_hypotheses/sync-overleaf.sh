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
# So `push` does not push any synthetic history at all. It takes the TREE we
# want Overleaf to have, commits it with Overleaf's current head as parent, and
# pushes that: an
# ordinary fast-forward, always legal, and it survives Overleaf autosaves that
# land between syncs.
#
# The paper is a subdirectory here but must be the project ROOT on Overleaf.
# That needs no `git subtree split` at all: `HEAD:$PREFIX` IS a tree whose root
# is the paper directory. We filter out NO_SYNC and push that tree.
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

# Everything in $PREFIX is mirrored to Overleaf EXCEPT these -- repo-side
# tooling with no business in a LaTeX project. sync-overleaf.sh in particular
# lives in $PREFIX, so mirroring it would let a pull overwrite this script with
# whatever Overleaf holds: it would clobber itself.
#
# A deny-list, not an allow-list, and deliberately so. An earlier version named
# the files TO sync; the moment a previous_draft.tex was uploaded in Overleaf
# (2026-09-21) it fell outside that list, so `pull` fetched it while local_tree
# ignored it -- permanent drift, and a next `push` that would have silently
# DELETED it from Overleaf. A paper directory gains .tex files, .bib files and
# figures over time; the set that must NOT travel is the stable one.
NO_SYNC=(sync-overleaf.sh README.md .gitignore)

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

# The tree we mirror: $PREFIX at HEAD, minus NO_SYNC. Overleaf needs
# these at the project ROOT, which is exactly what HEAD:$PREFIX already is --
# no `git subtree split` needed, and no synthetic history to reconcile.
local_tree() {
    local idx; idx="$(mktemp -u)"
    (
        export GIT_INDEX_FILE="$idx"
        git read-tree "HEAD:$PREFIX"
        for f in "${NO_SYNC[@]}"; do
            git update-index --force-remove "$f" 2>/dev/null || true
        done
        git write-tree
    )
    rm -f "$idx"
}

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
        TREE="$(local_tree)"
        [ -n "$TREE" ] || die "could not build a tree for $PREFIX"

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
        LOCAL_TREE="$(local_tree)"
        if [ "$REMOTE_TREE" = "$LOCAL_TREE" ]; then
            echo "$PREFIX already matches Overleaf -- nothing to pull."
            exit 0
        fi
        echo "Overleaf differs from $PREFIX. Applying its contents:"
        git diff --stat "$LOCAL_TREE" "$REMOTE_TREE" | sed 's/^/  /'

        # Index-based, and NO_SYNC-aware. Deliberately NOT `git subtree
        # pull` (it would merge Overleaf's per-keystroke autosave history into
        # this repo -- we want its content, not its log), and deliberately NOT
        # `rm -rf "$PREFIX"` before re-populating. An earlier version did the
        # latter; on 2026-09-20 the rm failed ("directory not empty" -- this repo
        # is on NFS, where an open file leaves a .nfs* entry), `set -e` aborted,
        # and the paper directory was left GUTTED with nothing restored.
        # Never remove the worktree copy before the replacement is in hand.
        local_idx="$(mktemp -u)"
        (
            export GIT_INDEX_FILE="$local_idx"
            git read-tree --prefix="$PREFIX/" "$REMOTE_TREE"
            git checkout-index -f -a
        )
        rm -f "$local_idx"

        # Delete files Overleaf no longer has. NO_SYNC entries are skipped, so
        # sync-overleaf.sh / README.md / .gitignore are never touched.
        remote_files="$(git ls-tree -r --name-only "$REMOTE_TREE")"
        git ls-files -- "$PREFIX" | while read -r f; do
            rel="${f#"$PREFIX/"}"
            skip=no
            for n in "${NO_SYNC[@]}"; do [ "$rel" = "$n" ] && skip=yes; done
            [ "$skip" = yes ] && continue
            grep -qxF "$rel" <<<"$remote_files" || { echo "  removed $rel"; rm -f "$f"; }
        done
        git add -- "$PREFIX"

        echo "Working tree updated. Review with 'git diff --cached -- $PREFIX', then commit."
        ;;

    status)
        echo "prefix: $PREFIX"
        echo "branch: $BRANCH"
        echo -n "remote: "; git remote get-url "$REMOTE" 2>/dev/null || echo "(not configured)"
        echo -n "token:  "
        TF="${OVERLEAF_TOKEN_FILE:-$HOME/.config/overleaf-token}"
        if [ ! -r "$TF" ]; then echo "$TF (MISSING)"
        elif [ ! -s "$TF" ]; then echo "$TF (EMPTY -- auth will fail)"
        else echo "$TF (present)"; fi
        echo -n "local changes under prefix: "
        [ -z "$(git status --porcelain -- "$PREFIX")" ] && echo "none" || { echo; git status --short -- "$PREFIX"; }
        if git remote get-url "$REMOTE" >/dev/null 2>&1; then
            fetch_remote
            if git rev-parse --verify -q "refs/remotes/$REMOTE/$BRANCH" >/dev/null; then
                R="$(git rev-parse "refs/remotes/$REMOTE/$BRANCH^{tree}")"
                L="$(local_tree 2>/dev/null || echo none)"
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
