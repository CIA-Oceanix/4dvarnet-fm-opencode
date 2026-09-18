## 2026-09-18: P2 method-paper scoping + literature check + ML-doc correction

**Summary:** New scoping doc for the third paper — the DA-venue method paper on
unrolled variational solvers and conditional flows — preceded by a literature
check that materially narrowed its claims. Also corrects the ML doc, whose
2026-09-17 revision drew the wrong conclusion from a correct premise.

**Files modified:**
`docs/scoping/p2_unrolled_solvers_and_flows.md` — new, ~400 lines.
`docs/scoping/ml_paper_exploiting_prior_knowledge.md` — new §8.1 correction;
§10 decisions 5 revised and 6 closed.
`docs/scoping/README.md` — P2 index entry; the three-paper split recorded.

**Rationale:** P2 is the nearest-term paper of the three — S0 only, no
Weak-4DVar, no symmetric model-error arm, and #211 plus the affine-decomposition
doc are already most of its results section. Restricting to S0 is a design
decision, not a simplification: the claims are about operator structure, so model
error is irrelevant, and it immunizes P2 against P1's R1 asymmetry.

**The literature check changed the paper.** Two of the three things P2 was going
to claim are already published: the Tweedie/Kalman/score identification
(arXiv 2605.15902) and the closed-form posterior covariance for flow matching
(arXiv 2605.00941) cover the decomposition itself, and Warm-Start Diffusion
(arXiv 2507.09212) published the deterministic warm start. Retrieval confirmed
WSD is a **hard initial condition, not a continuous velocity blend**, that its
"warmth blending" is a training-level weight rather than a `τ`-schedule, and that
it **explicitly excludes DA and chaotic dynamics** — so the blend and the DA-side
taxonomy survive. The word "unification" was removed from the title and the
contribution narrowed to: the DA-method taxonomy via the partition, the unrolled
solver as the degenerate member, the `p >= 1` continuous blend, and the negatives
(notably `Psi_NG` non-invariance under a gain change). The existing 4DVarNet UQ
line (ensemble-based; SPDE priors, AIES 2025; VAE prior) is recorded as R3: P2
must position as *explaining what each route supplies*, not as a fourth
competitor.

**ADDA assessment** (arXiv 2608.23297, m-dml/ADDA), requested separately: it is
infrastructure, **not** an evaluation framework — its own paper says it is "not a
benchmarking study" and it ships RMSE / error growth / filter divergence with no
CRPS, energy score, rank histogram or spread-skill, so it does not close this
project's measurement gap. Recorded as worth a bounded cross-check (P2 M5), not a
port: early-stage (3 stars, 62 commits, under review), and porting risks
invalidating published numbers.

**ML-doc correction:** §8.0 reasoned "family D cannot host E2, therefore retarget
the headline". The premise is right, the inference wrong, because that paper is
deliberately non-DA. New §8.1: **drop family D, keep C1/E2 as the headline**;
A and B host an exact-prior arm exactly and C gives an exactly-known H. The
matched-budget mean-slot comparison moves to P2 (M2), where it belongs — it is
inherently L96/QG — which also closes §10 decision 6.

**Verification:** Docs only, no code touched. Every figure re-checked against
`reports/l96/outputs/l96_fm_sampler_benchmark.md` and
`docs/results/cfm_affine_velocity_decomposition.md` (blend 0.3764 vs warm 0.3978;
decoupled 0.6117/0.1913 vs 0.3898/0.2224 at matched settings; ratio/target
1.20/1.22/1.24; `dep_fit = 0.147` vs `R = 0.163`); the referenced probe
`reports/l96/probe_cfm_affine_decomposition.py` confirmed to exist. All URLs
verified by retrieval on 2026-09-18. Cross-directory reference checker: 0 broken.

**Caution recorded:** one search returned this repo's own PR #211 and echoed its
`Psi_mean` terminology back as if it were external literature. Prior-art claims
in §7 come from retrieved papers only, never from search summaries.
