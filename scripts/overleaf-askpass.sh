#!/usr/bin/env bash
#
# GIT_ASKPASS helper for the Overleaf git bridge.
#
# git invokes this with the prompt text as $1 and expects the answer on stdout.
# The Overleaf git token is read from a 0600 file, so it never appears on a
# command line, in shell history, or in .git/config.
#
#   umask 077 && printf '%s' '<token>' > ~/.config/overleaf-token
#
# Token: Overleaf -> Account Settings -> Git integration. Overleaf accepts any
# username, so "git" is returned unconditionally.
#
# Lives under scripts/ rather than beside the paper deliberately: sync-overleaf.sh
# refuses to run with uncommitted changes under the paper prefix, so a helper
# stored there could not be edited and used in the same breath.
set -euo pipefail

TOKEN_FILE="${OVERLEAF_TOKEN_FILE:-$HOME/.config/overleaf-token}"

case "${1:-}" in
    Username*|username*) printf 'git' ;;
    *)
        if [ ! -r "$TOKEN_FILE" ]; then
            # Returning nothing makes git fail with an auth error rather than
            # hang; the message goes to stderr so it is visible but not
            # mistaken for the password itself.
            echo "overleaf-askpass: no readable token at $TOKEN_FILE" >&2
            exit 1
        fi
        tr -d '\r\n' < "$TOKEN_FILE"
        ;;
esac
