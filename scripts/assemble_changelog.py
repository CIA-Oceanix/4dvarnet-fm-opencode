#!/usr/bin/env python3
"""Fold CHANGELOG.d/ fragment files into CHANGELOG.md (newest first) and
delete the consumed fragments.

Every PR adds its own fragment file to CHANGELOG.d/ instead of editing
CHANGELOG.md directly (see CHANGELOG.d/README.md) -- that's what lets
concurrent PRs land without a merge conflict (new files never conflict with
each other; edits to one shared file's top section always do). This script
is a separate, occasional maintenance step, not part of any individual PR:
run it whenever convenient, e.g. after a batch of PRs has merged.

Usage:
    python scripts/assemble_changelog.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
FRAGMENTS_DIR = ROOT / "CHANGELOG.d"
MARKER = ("<!-- AUTO-ASSEMBLED FROM CHANGELOG.d/ -- do not hand-edit above "
          "this line; add new files to CHANGELOG.d/ instead (see "
          "CHANGELOG.d/README.md) -->")


def main() -> None:
    fragments = sorted(
        (p for p in FRAGMENTS_DIR.glob("*.md") if p.name != "README.md"),
        reverse=True,
    )
    if not fragments:
        print("No fragments to assemble.")
        return

    text = CHANGELOG.read_text()
    if MARKER not in text:
        raise SystemExit(f"Marker not found in {CHANGELOG} -- expected:\n{MARKER}")
    head, _, tail = text.partition(MARKER)
    tail = tail.lstrip("\n")

    fragment_text = "\n\n".join(f.read_text().strip() for f in fragments)

    new_text = f"{head}{MARKER}\n\n{fragment_text}\n\n{tail}"
    CHANGELOG.write_text(new_text)

    for f in fragments:
        f.unlink()

    print(f"Assembled {len(fragments)} fragment(s) into {CHANGELOG}:")
    for f in fragments:
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
