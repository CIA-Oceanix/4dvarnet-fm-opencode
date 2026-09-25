# Affine decomposition of the CFM velocity, and what it costs to exploit it

**Status:** RESULTS, draft for review (2026-09-15). Derivation + a measurement on two
trained L96 checkpoints, via `reports/l96/probe_cfm_affine_decomposition.py`.
Analysis only — no model, training or evaluation code is changed by this note.

Goal: make precise the proposed parameterization of the CFM velocity as
(amortized mean) + (linear term) + (nonlinear remainder), decide whether the
remainder is small enough for the intended compute saving, and record what was
actually measured.

---

## 1. Setup and notation

Matching the code exactly:

- Linear interpolant (`models/interpolant.py:8-26`): $x_\tau = (1-\tau)x_0 + \tau x_1$, $\tau\in[0,1]$.
- Source: $x_0 \sim \mathcal N(0,\sigma_0^2 I)$, $\sigma_0 = $ `sigma_prior` $= 0.5$ (`conf/schema.py:168`), independent of $(x_1,y)$.
- CFM velocity target: $v = x_1 - x_0$ (`models/interpolant.py:28-34`).
- $y$ = observations; $x_1$ = the state trajectory (here the 24-D observed subspace, $T=3000$).

Write

$$\mu(x_\tau,\tau,y) := \mathbb E[x_1 \mid x_\tau,\tau,y], \qquad m(y) := \mathbb E[x_1\mid y].$$

$m(y)$ is the **amortized estimator** — the slot the unrolled 4DVarNet-style solver is meant to fill.

---

## 2. Step 1 — the velocity is entirely determined by the posterior mean

*Exact, no assumptions.* For $\tau<1$ the interpolant can be inverted pointwise,
$x_0 = (x_\tau - \tau x_1)/(1-\tau)$, so conditioning on $(x_\tau,\tau,y)$ gives

$$\mathbb E[x_0\mid x_\tau,\tau,y] = \frac{x_\tau - \tau\,\mu}{1-\tau}.$$

Hence

$$v^\star(x_\tau,\tau,y) = \mathbb E[x_1-x_0\mid x_\tau,\tau,y] = \mu - \frac{x_\tau-\tau\mu}{1-\tau} = \frac{(1-\tau)\mu - x_\tau + \tau\mu}{1-\tau}$$

$$\boxed{\;v^\star = \frac{\mu - x_\tau}{1-\tau}\;}$$

This is the identity already inlined as `x1_hat` ($\hat x_1 = x_\tau + (1-\tau)v$)
and used by `PredictStateCFM.sample`. Everything below is therefore a statement
about $\mu$.

---

## 3. Step 2 — projecting $\mu$ onto $\{m(y),\,x_\tau\}$

Since $\mathbb E[x_0]=0$ we have $\mathbb E[x_\tau\mid y] = \tau m(y)$. Define the centered regressor

$$z := x_\tau - \tau m(y) = (1-\tau)x_0 + \tau\,(x_1 - m(y)).$$

Seek the $L^2$-optimal linear coefficient $G(\tau)$ in

$$\mu(x_\tau,\tau,y) = m(y) + G(\tau)\,z + \rho(x_\tau,\tau,y), \qquad \rho \perp z .$$

Normal equations give $G = \mathbb E[(\mu-m)z^\top]\,\big(\mathbb E[zz^\top]\big)^{-1}$. Two simplifications:

1. $z$ is $(x_\tau,y)$-measurable and $\mu = \mathbb E[x_1\mid x_\tau,y]$, so by the tower property $\mathbb E[\mu z^\top] = \mathbb E[x_1 z^\top]$, hence $\mathbb E[(\mu-m)z^\top] = \mathbb E[(x_1-m)z^\top]$.
2. Using $z=(1-\tau)x_0+\tau(x_1-m)$ and $x_0\perp(x_1,y)$, $\mathbb E[x_0]=0$:

$$\mathbb E[(x_1-m)z^\top] = \tau\,\bar P,\qquad \mathbb E[zz^\top] = (1-\tau)^2\sigma_0^2 I + \tau^2\bar P,$$

with $\bar P := \mathbb E\big[(x_1-m(y))(x_1-m(y))^\top\big] = \mathbb E_y\!\big[\mathrm{Cov}(x_1\mid y)\big]$ the **average posterior covariance**. Therefore

$$\boxed{\;G(\tau) = \tau\,\bar P\,\big[(1-\tau)^2\sigma_0^2 I + \tau^2\bar P\big]^{-1}\;}$$

**Interpretation of $\rho$.** If $(x_1,y)$ are jointly Gaussian (so that
$p(x_1\mid y)$ is Gaussian with $y$-independent covariance), the same formula is
obtained from exact Gaussian conditioning and $\rho\equiv 0$. So

> $\rho$ is exactly the non-Gaussianity of the posterior $p(x_1\mid y)$, expressed in flow coordinates.

**Note.** $y$ enters only through $m(y)$. A *separate* linear-in-$y$ channel is
redundant in this limit; what is not redundant is an **innovation** term
$H^\top R^{-1}(y - Hx_\tau)$ — linear in $(y,x_\tau)$ jointly, and exactly the
observation-cost gradient the `grad+state`/`subgrad+state` solvers already form.

---

## 4. Step 3 — the three-term velocity

Substituting into $v^\star=(\mu-x_\tau)/(1-\tau)$:

$$\mu - x_\tau = (I-\tau G)m - (I-G)x_\tau + \rho.$$

Define $K(\tau) := (I-G(\tau))/(1-\tau)$, so $I-G=(1-\tau)K$ and

$$I-\tau G = (I-G) + (1-\tau)G = (1-\tau)(K+G).$$

Hence

$$\boxed{\;v^\star(x_\tau,\tau,y) = \underbrace{\big[K(\tau)+G(\tau)\big]m(y)}_{\text{amortized}} \;-\; \underbrace{K(\tau)\,x_\tau}_{\text{linear}} \;+\; \underbrace{\frac{\rho(x_\tau,\tau,y)}{1-\tau}}_{\text{nonlinear}}\;}$$

### Two parameterizations — do not confuse them

| target | form | coefficients |
|---|---|---|
| posterior mean | $\mu = A(\tau)\,m + B(\tau)\,x_\tau + \rho$ | $A = I-\tau G$, $B = G$ |
| velocity | $v = a(\tau)\,m + b(\tau)\,x_\tau + \rho/(1-\tau)$ | $a = K+G$, $b = -K$ |

related by $a = A/(1-\tau)$, $b = (B-I)/(1-\tau)$. **§6 fits the $\mu$ form**, so
the fitted $B$ is compared against $G$.

### Isotropic closed form

With $\bar P = p\,I$ and $s:=\sigma_0^2$:

$$\gamma(\tau) = \frac{\tau p}{(1-\tau)^2 s + \tau^2 p}, \qquad \kappa(\tau)=\frac{1-\gamma(\tau)}{1-\tau},$$

$$A = 1-\tau\gamma,\quad B=\gamma,\qquad a = \kappa+\gamma = \frac{1-\tau\gamma}{1-\tau},\quad b=-\kappa .$$

**Endpoint checks** (all four hold):

- $\tau=0$: $\gamma=0\Rightarrow \mu=m$ ✓ (indeed $\mathbb E[x_1\mid x_0,y]=m$ since $x_0\perp x_1$); and $a=1,b=-1\Rightarrow v=m-x_0$ ✓.
- $\tau\to1$: $\gamma\to1\Rightarrow\mu\to x_\tau=x_1$ ✓. With $\varepsilon=1-\tau$, $1-\gamma\simeq-\varepsilon$ so $\kappa\to-1$, giving $a\to0$, $b\to1$, $v\to x_1$ ✓ (since $\mathbb E[x_1-x_0\mid x_1,y]=x_1$).

**Peak of the gain.** $\gamma$ can exceed 1. Setting $\gamma'=0$ gives $s(1-\tau^2)=\tau^2 p$, i.e.

$$\tau^\star=\sqrt{\tfrac{s}{s+p}},\qquad \gamma_{\max}=\frac{\tau^\star p}{2s(1-\tau^\star)} .$$

---

## 5. Consequence for the proposed architecture

$$v_\theta(x_\tau,\tau,y) = a_\theta(\tau)\,\mu_\phi(y) \;+\; b_\theta(\tau)\,x_\tau \;+\; c_\theta(\tau)\,H^\top R^{-1}\!\big(y-Hx_\tau\big) \;+\; r_\psi(x_\tau,\tau,y)$$

with $\mu_\phi$ the unrolled solver, $(a,b,c)$ scalar functions of $\tau$
initialized at $(\kappa+\gamma,\,-\kappa,\,0)$, and $r_\psi$ the residual net.
The compute argument: $\mu_\phi(y)$ does **not** depend on $x_\tau$, so it is
evaluated once per sample rather than inside every Euler step.

Practical: regress $\mu$ (or the $\mu$-residual) and divide by $(1-\tau)$ at the
end — never regress $v$ directly, or the $1/(1-\tau)$ inflates the loss near
$\tau=1$. `PredictStateCFM` already does this.

---

## 6. What was measured

Probe: `reports/l96/probe_cfm_affine_decomposition.py`. To reproduce §7:

```bash
python reports/l96/probe_cfm_affine_decomposition.py \
    --checkpoint experiments/V3_predict_state_cfm_l96/checkpoints/stage1_best.ckpt \
    --case s0 --batch-size 25 --output cfm_affine_v3_s0.json
```

Cached 200-window test set `l96_datasets_obsj2_int100_nwin200.pt`; all
$200\times3000\times24 = 1.44\times10^7$ entries pooled; $\tau$ on a 12-point grid.

- $\hat m(y) = \frac1K\sum_{k=1}^K \mu_\theta(x_0^{(k)},\,\tau{=}0,\,y)$, $K=4$ draws — justified by $\mathbb E[x_1\mid x_0,y]=m(y)$.
- For each $\tau$: one draw $x_0$, $x_\tau=(1-\tau)x_0+\tau x_1$ with $x_1$ the **true** cached state; $\mu = \mu_\theta(x_\tau,\tau,y)$ the trained network's output.
- Fit **scalars** $(A,B)$ minimizing $\sum\|\mu - A\hat m - Bx_\tau\|^2$; residual sum of squares at the optimum is $\mathrm{RSS} = S_{\mu\mu} - A\,S_{m\mu} - B\,S_{x\mu}$.
- $\hat p = \overline{(x_1-\hat m)^2}$ (the mean estimator's own MSE), $s=\sigma_0^2=0.25$.

Reported ratios (note the second one):

$$\frac{\|\rho\|}{\|\mu\|},\qquad \frac{\|\rho\|}{\|\mu-x_\tau\|}\equiv\frac{\|\rho/(1-\tau)\|}{\|v\|},\qquad \frac{\|\rho\|}{\|x_1\|}.$$

The middle one is the **velocity-relative** residual — the decision-relevant one,
since the ODE integrates $v$, not $\mu$.

**What $\rho$ is and is not.** It measures the non-affineness of the *learned*
$\mu_\theta$, not of the true posterior mean. That is the right target for the
architecture question ("can $\mu_\theta$ be replaced by affine + small residual?"),
but it inherits the network's own error and is not a statement about $p(x_1\mid y)$.
Since $(A,B)$ are fitted post hoc on the test set, these residuals are a **lower
bound** on what any affine architecture could achieve.

---

## 7. Results

Checkpoints: **V3** = `PredictStateCFM` (`experiments/V3_predict_state_cfm_l96`,
$\hat m$ RMSE 0.652, $\hat p=0.425$) and **FDV1CFM** =
`FourDVarNetPredictStateCFM`, unrolled $K_{\rm inner}=5$ solver as $\mu$
(`4dvarnet-fm-fdv1cfm/experiments/FDV1CFM_predict_state_l96`, $\hat m$ RMSE 0.516,
$\hat p=0.266$). S1 reproduces S0 to 3 decimals throughout (expected: obs-only models).

### 7.1 The closed form predicts the fitted gain

$B_{\rm fit}$ vs $\gamma(\tau)=\tau\hat p/[(1-\tau)^2s+\tau^2\hat p]$ — a **one-parameter** formula:

| $\tau$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|
| V3 $B_{\rm fit}$ | 0.207 | 0.481 | 0.792 | 1.068 | 1.246 | 1.306 | 1.274 | 1.188 | 1.083 |
| V3 $\gamma$ | 0.206 | 0.480 | 0.793 | 1.076 | 1.259 | 1.321 | 1.289 | 1.206 | 1.103 |
| FDV1CFM $B_{\rm fit}$ | 0.131 | 0.308 | 0.530 | 0.774 | 0.988 | 1.125 | 1.173 | 1.147 | 1.074 |
| FDV1CFM $\gamma$ | 0.130 | 0.312 | 0.545 | 0.802 | 1.031 | 1.176 | 1.218 | 1.181 | 1.098 |

Agreement ~1.5% (V3) / ~4% (FDV1CFM). Two further confirmations:

- **Constrained form holds**: fitted $A$ matches $1-\tau B_{\rm fit}$ to $<0.015$ at every $\tau$ (max deviation 0.0073 V3, 0.0139 FDV1CFM) — the free 2-parameter fit collapses onto $\mu = m + G(x_\tau-\tau m)$.
- **Predicted peak location**: $\tau^\star=\sqrt{s/(s+\hat p)}$ gives 0.609 (V3) and 0.696 (FDV1CFM); the observed maxima of $B_{\rm fit}$ sit at $\tau=0.6$ and $\tau=0.7$ respectively — on the grid, both exact.
- **Scalar suffices**: refitting $(A,B)$ per channel changes $\|\rho\|/\|\mu\|$ from 0.2381 to 0.2356 (V3, $\tau{=}0.2$).

### 7.2 Residuals

| $\tau$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V3 $\|\rho\|/\|\mu\|$ | .232 | .238 | .222 | .193 | .156 | .118 | .084 | .055 | .037 | .033 | .032 |
| **V3 $\|\rho\|/\|\mu-x_\tau\|$** | .253 | .291 | .310 | .314 | .305 | .288 | .271 | .269 | .355 | .567 | .959 |
| **FDV1CFM $\|\rho\|/\|\mu-x_\tau\|$** | .190 | .227 | .250 | .266 | .273 | .271 | .262 | .253 | .281 | .405 | .716 |

---

## 8. Conclusions

1. **The decomposition is the right coordinate system, and the schedules are free.**
   $B_{\rm fit}\approx\gamma(\tau)$ to a few percent from a one-parameter formula whose
   parameter $\hat p$ is just the mean estimator's MSE — already known. $A$ is then
   determined by $B$. Scalar (not per-channel) schedules suffice. So $a(\tau),b(\tau)$
   need not be learned at all; at most they need fine-tuning around the closed form.

2. **The affine part captures nearly all of $\mu$ but a much smaller share of $v$.**
   Since $\rho\perp\mathrm{span}\{m,x_\tau\}$, the affine part explains
   $1-(\|\rho\|/\|\mu\|)^2$ of $\mu$'s energy: **94.3–99.9%** (V3), **96.6–99.98%**
   (FDV1CFM). But $\mu$ is dominated by a large common "move toward the data"
   component that cancels in $\mu-x_\tau$. The ODE integrates that difference, and
   there the residual is **25–31% of the velocity amplitude (V3) / 19–27% (FDV1CFM),
   roughly flat over $\tau\in[0.1,0.8]$** — i.e. 90–94% of velocity *energy* explained
   in the bulk — degrading steeply past $\tau=0.9$ (to 8% energy explained at
   $\tau=0.99$ for V3).

   Amplitude is the relevant currency for sizing $r_\psi$ (it must *output* something
   of that magnitude); energy is the relevant one for "how much is explained". Both
   are quoted to avoid the two being conflated.

3. **Therefore $r_\psi$ cannot be a thin correction head**, and the compute claim must
   be restated. The saving is not "one solve + $N_{\rm outer}$ cheap evals". It is that
   $\mu_\phi(y)$ stops being recomputed inside every Euler step:
   $N_{\rm outer}\times K_{\rm inner}=50$ becomes $K_{\rm inner} + N_{\rm outer}\times\mathrm{cost}(r_\psi)\approx 5+10=15$
   for a same-sized residual net — about **3×**, not 5×.

4. **The approximation improves with a better mean model** (V3 → FDV1CFM: peak velocity
   residual 0.314 → 0.273), which supports putting the unrolled solver in the $m$ slot.

5. ~~**Actionable, and a consequence of the flat-then-steep profile**: integrate coarsely
   where the affine model is accurate ($\tau<0.8$) and concentrate network evaluations
   near $\tau\to1$.~~ **RETRACTED 2026-09-16 — this was a model artifact, not a property
   of the problem.** The late-$\tau$ blow-up that motivated it is present in V3 and
   FDV1CFM but **absent in V2rerun** (`TweedieCFM`, multi-$\tau$), whose
   $\|\rho\|/\|\mu-x_\tau\|$ at $\tau=0.95$ is 0.204 and *falls* to 0.174 under sparse
   observations, against V3's 0.57-0.62. Since V2 and FDV1CFM have near-identical mean
   quality ($\hat m$ RMSE 0.522 vs 0.515) but very different late-$\tau$ residuals, the
   blow-up is a property of the **residual/velocity stage**, not of the posterior — V2's
   explicitly two-stage design (mean estimator + residual CFM) handles $\tau\to1$
   cleanly. See `docs/results/l96_psi_decomposition.md` §1.

### Open / caveats

- $\hat m$ carries $x_0$-dispersion 0.17 (V3) / 0.14 (FDV1CFM) $\Rightarrow$ $\approx$0.07–0.085 on the 4-draw mean against a state scale $\approx$3.4, i.e. ~2% regressor noise. Too small to explain a 25–30% residual, but it inflates it slightly. Raising $K$ would tighten the numbers.
- Both checkpoints are non-normalized `UNet1D`-family. The current best schemes use the monai backbone; re-run there before committing.
- $\bar P$ was treated as isotropic ($\bar P = pI$). The per-channel refit suggests this is adequate, but a diagonal $\bar P$ is a cheap check.
- $\rho$ is measured against $\mu_\theta$, not the true posterior mean (see §6).
