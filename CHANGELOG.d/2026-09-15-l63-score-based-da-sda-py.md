## 2026-09-15: Replace L63 score-based DA with the master-branch models/sda.py implementation

**Summary:** Deleted `score_based_inference.py`'s bespoke unconditional-VanillaCFM prior +
linear-Gaussian Kalman-gain guided sampler and replaced it with the same model class and
sampler the L96 SDA benchmark uses: `models/sda.py::UnconditionalPriorCFM` (a purely
unconditional x -> v(x, tau) prior, no obs/forcing/params conditioning at all, trained via
new `config/models/sda_prior.yaml`) guided at inference time by
`evaluation/sda_sampler.py::sda_guided_sample` (DPS/Pi-GDM normalized-gradient guidance),
run via new `eval_sda_l63.py`. `guidance_weight` (no principled default) was tuned via an
S0 sweep over {0,1,3,10,20,40,80,150}: bell curve peaking at 20 (R2=0.940 vs 0.10 at the
placeholder default of 1.0), same qualitative shape as the L96 SDA1 sweep.
**Files modified:** `score_based_inference.py` (deleted), `eval_sda_l63.py` (new),
`config/models/sda_prior.yaml` (new), `config/lorenz63.yaml` (added `training.loss` --
no L63 config declared it, which the merged `train.py`'s `LitModel` construction now
requires unconditionally for every non-tweedie model, silently breaking all L63 training
until this fix), `reports/README.md` (score-based row regenerated: s0 RMSE 1.958/R2
0.941/CRPS 0.874, s1 RMSE 1.647/R2 0.958/CRPS 0.761).
**Rationale:** user request -- "follow the main branch version of the code for l63,
delete the merge/victor version."
**Verification:** full production run (`train.py --config-name models/sda_prior`, both
s0/s1, early-stopped at 710/934 of 3000 epochs) then `eval_sda_l63.py` (N_ensemble=50,
N_outer=10, guidance_weight=20) end-to-end on the real 200-window S0/S1 test sets;
correct `experiments/l63/<model>/results.json` schema, matches every other L63 model row.
