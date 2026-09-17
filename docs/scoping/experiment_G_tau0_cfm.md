# Experiment G: VanillaCFM with τ=0 Training

## Motivation

Test whether VanillaCFM's advantage over DirectUNet comes from multi-τ
training (random noise levels) or from the residual loss formulation.
At τ=0, the CFM optimal predictor satisfies:

    v_θ(x₀, obs, 0) = E[states | obs] - x₀

One Euler step gives x = x₀ + x₀ + v = E[states | obs], i.e., the
conditional mean — same target as DirectUNet. The remaining 9 Euler
steps sample the posterior without improving mean accuracy.

If G ≈ F in RMSE  → multi-τ training is NOT the source of CFM's advantage
If G ≈ E (DirectUNet) → multi-τ training IS the source

## Changes

### 1. `conf/schema.py` — Add flag to VanillaCFMConfig

```python
@dataclass
class VanillaCFMConfig:
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    time_emb_dim: int = 64
    N_outer: int = 10
    sigma_prior: float = 0.5
    dropout: float = 0.1
    train_tau_0_only: bool = False   # NEW
```

### 2. `models/vanilla_cfm.py` — Modify __init__, compute_cfm_loss, sample

__init__: add parameter `train_tau_0_only=False`, store as self.train_tau_0_only

compute_cfm_loss: replace `tau = torch.rand(B, device=device)` with:
```python
if self.train_tau_0_only:
    tau = torch.zeros(B, device=device)
else:
    tau = torch.rand(B, device=device)
```

sample: add early return at top of method:
```python
if self.train_tau_0_only:
    tau = torch.zeros(B, device=device)
    v = self.forward(x, obs, tau)
    return x + v   # single Euler step with dt=1.0
```

### 3. `train.py` — model_factory, pass new flag

```python
train_tau_0_only=vc.get("train_tau_0_only", False),
```

### 4. New configs: `config/experiment/G{1,2,3}.yaml`

| ID | Channels | Train mix | Analogue |
|---|---|---|---|
| G1 | [64,128,256] | cs1+cs2 | F1 with τ=0 |
| G2 | [32,64,128] | cs1+cs2 | F2 with τ=0 |
| G3 | [32,64,128] | cs1_rand+cs2_rand | F3 with τ=0 |

Each is identical to its F counterpart except `train_tau_0_only: true`.

## Expected outcomes

| Comparison | If true, means... |
|---|---|
| G1 ≈ F1 | Multi-τ training irrelevant; advantage is from residual loss or noisy input |
| G1 ≈ E2 (DirectUNet) | Multi-τ training IS the source of CFM's advantage |
| G1 RMSE between F1 and E2 | Both factors contribute |

## Next steps after G

If G1 ≈ E2 (multi-τ is the source), the question is settled.
If G1 ≈ F1 (multi-τ is not the source), add experiment H:

**H: CFM τ=0 + deterministic prior (x₀=0)**

Controls for the input difference (noise vs zeros) to isolate whether
the residual loss formulation `MSE(v, states - x₀)` itself helps vs
direct `MSE(pred, states)`.

## Run command

```bash
python train.py --config-name experiment/G1_vanilla_cfm_t0_default
```
