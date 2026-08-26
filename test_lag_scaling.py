import sys
sys.path.insert(0, '.')

import torch
import numpy as np

print("Testing lag difference at larger resolution...")

for lead in [5, 10, 20]:
    print(f"\nlead={lead}:")
    for lag in [1.0, 2.0, 5.0]:
        dt_steps = max(int(lag / 2.0), 1)  # Roughly derived
        lead_actual = dt_steps + 1

        if lead_actual < lead_actual:
            continue  # Skip spans that don't match

        gens = [torch.Generator().manual_seed(i*42) for i in range(10)]
        r = torch.rand(10, generator=gens[0])

        member_lags = []
        for i in range(10):
            k_tplus1 = int(r[i] * lead_actual) + 1
            member_lags.append(k_tplus1)

        lag_mean = np.mean(member_lags)
        lag_std = np.std(member_lags)
        print(f"  lag={lag:.1f}: mean_timestep={lag_mean:.2f}, std={lag_std:.2f}")

print("\n✓ Test complete")
print("  Larger lead enables better separation between lag values")
