"""
generate_data.py
----------------
Simulates battery degradation data inspired by the NASA PCoE Battery Dataset.
Real dataset: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

Each battery degrades following an empirical model:
    SOH(k) = a * exp(b * k) + c * exp(d * k) + noise

where k is the cycle number and parameters are drawn from
a hierarchical distribution to model unit-to-unit variability.
"""

import os
import numpy as np
import pandas as pd

N_BATTERIES = 8
N_CYCLES = 150
EOL_THRESHOLD = 0.8  # End-of-life: SOH drops below 80%

def simulate_battery(battery_id, seed=None):
    rng = np.random.default_rng(seed)
    # Parameters from NASA empirical model (double-exponential)
    a = rng.normal(0.97, 0.01)
    b = rng.normal(-0.0008, 0.0001)
    c = rng.normal(0.03, 0.005)
    d = rng.normal(-0.05, 0.005)
    noise_std = rng.uniform(0.003, 0.008)

    cycles = np.arange(1, N_CYCLES + 1)
    soh = a * np.exp(b * cycles) + c * np.exp(d * cycles)
    soh += rng.normal(0, noise_std, size=N_CYCLES)
    soh = np.clip(soh, 0.5, 1.0)

    df = pd.DataFrame({
        "battery_id": f"B{battery_id:04d}",
        "cycle": cycles,
        "SOH": soh,
        "capacity_Ah": soh * 2.0,  # nominal 2Ah
    })
    return df, {"a": a, "b": b, "c": c, "d": d, "noise_std": noise_std}


def main():
    np.random.seed(42)
    all_data = []
    true_params = {}
    for i in range(N_BATTERIES):
        df, params = simulate_battery(i + 1, seed=i * 7)
        all_data.append(df)
        true_params[f"B{i+1:04d}"] = params

    dataset = pd.concat(all_data, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    dataset.to_csv(os.path.join("data", "battery_degradation.csv"), index=False)
    print(f"Dataset saved: {len(dataset)} rows, {N_BATTERIES} batteries, {N_CYCLES} cycles each.")
    print(dataset.head(10))
    return dataset, true_params


if __name__ == "__main__":
    main()
