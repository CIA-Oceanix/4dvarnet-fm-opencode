## 2026-11-08: L96 archive script covers the 2D CFM / SDA1 runs

**Summary:** `scripts/archive_l96_benchmark_default.py` now lists the 2D runs: VanillaCFM and SDA1 at S, M and L (seed 1), and the DirectUNet-2D-M -> SDA1-2D-M hybrid. They are already hard-linked into `experiments/l96/`.
**Files modified:** `scripts/archive_l96_benchmark_default.py`: new `FDV2D` source root, `HYB2D` eval sub-directory and seven runs.
**Rationale:** These runs back the 2D rows of the extended report and the P1 tables. Their checkpoints existed only in the `4dvarnet-fm-l96-fdv-2d` worktree, and listing them in the script makes the archive reproducible.
**Verification:** Dry run: the 45 files of the 2D runs and the 2 test-set files are reported `same`, with no conflict. `ruff check` passes.
