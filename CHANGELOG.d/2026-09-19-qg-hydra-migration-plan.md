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

The cost centre is not the entry point but the **67 sbatch scripts** that pass
the 37 CLI flags, most of which are the reproducible record of experiments
already run — which rules out a hard cut and argues for a translation shim.
Staged as four independently revertible PRs; PR 3 is the one that pays, because
wiring `training/resume.py` in needs exactly the `DictConfig` the schema
defines, and that closes the gap that cost the Q5 sweep three 24-hour jobs.

**Verification:** Documentation only. Claims checked against the tree: 67 sbatch
scripts call `train_qg_neural.py`, 37 `add_argument` calls, 13 QG experiment
configs, `model.hidden_channels` read by nothing, `train.py` pinned at
`version_base="1.3"`.
