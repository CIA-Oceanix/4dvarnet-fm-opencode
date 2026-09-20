## 2026-09-19: Plan for moving QG onto Hydra

**Summary:** Design doc only — no code. Plans the migration of
`train_qg_neural.py` (argparse + flat `OmegaConf.load`) onto Hydra, the way
`train.py` already is, under the constraint that the archived Q1-Q4 checkpoints
stay evaluable and the published benchmark stays reproducible.

**Files modified:** `docs/scoping/qg_hydra_migration.md` (new),
`docs/scoping/README.md` — index row.

**Rationale:** The key finding is that the migration's compatibility surface is
small and already in place: `resolved_config.yaml`'s `model`/`data`/`training`
sections (shipped 2026-09-19, deliberately given L96's own layout) are readable
by the current evaluation code, so a Hydra config that keeps those key names
makes a Hydra-trained run indistinguishable from today's as far as evaluation is
concerned. The plan therefore adopts the existing resolved-config schema as the
Hydra schema rather than designing a new one.

The caller set that must move with the entry point is **16 sbatch scripts**, all
tracked and none building a flag name dynamically, so the conversion is
mechanical and lands in the same PR as the entry point; a missed script fails
loudly, because Hydra rejects an unrecognized `--flag`. Staged as three
independently revertible PRs; PR 3 is the one that pays, because wiring
`training/resume.py` in needs exactly the `DictConfig` the schema defines, and
that closes the gap that cost the Q5 sweep three 24-hour jobs.

**Verification:** Documentation only. Claims checked against the tree:
`grep -l train_qg_neural.py batch/*.sbatch` → 16 scripts, 37 `add_argument`
calls, 13 QG experiment configs, `model.hidden_channels` read by nothing,
`train.py` pinned at `version_base="1.3"`.

**Correction (same day, pre-merge):** the first draft said 67 scripts and built
the A-vs-B recommendation on it. That count came from `grep -rl ... batch/`,
which swept in 51 job logs under `batch/logs/` that echo the command line they
ran. Caught in review. At the real count the hard cut is the better option, so
the recommendation was re-derived rather than restated.
