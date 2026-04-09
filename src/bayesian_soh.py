"""
bayesian_soh.py
---------------
Bayesian Hierarchical Model for Battery State-of-Health (SOH) Prediction.
Uses PyMC with the NumPyro/JAX backend — fast even without a C++ compiler.

References:
    [1] Zhou & Howey (2023), IFAC-PapersOnLine 56(2), 6117-6123.
    [2] Vanem et al. (2021), Journal of Energy Storage 43, 103158.

Author : Hafid IDRISSI
Context: CIFRE PhD preparation — Stellantis / CentraleSupélec
"""

import os, warnings
warnings.filterwarnings("ignore")
os.environ["PYTENSOR_FLAGS"] = "cxx="   # silence g++ warning on Windows

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "battery_degradation.csv")
OUTPUT_DIR  = "results"
EOL         = 0.80
TRAIN_FRAC  = 0.50
N_SAMPLES   = 500
N_TUNE      = 300
TARGET_BATT = "B0001"

os.makedirs(OUTPUT_DIR, exist_ok=True)
df        = pd.read_csv(DATA_PATH)
batteries = df["battery_id"].unique()
print(f"Loaded {len(df)} observations — {len(batteries)} batteries.\n")


# ── 1. Single-battery model ──────────────────────────────────────────────────
def fit_single_battery(battery_id):
    data    = df[df["battery_id"] == battery_id].copy()
    n_train = int(len(data) * TRAIN_FRAC)
    train, test = data.iloc[:n_train], data.iloc[n_train:]
    k_train = train["cycle"].values.astype(float)
    y_train = train["SOH"].values.astype(float)
    k_all   = data["cycle"].values.astype(float)

    with pm.Model() as model:
        a     = pm.Normal("a",     mu=1.0,    sigma=0.05)
        b     = pm.Normal("b",     mu=-0.001, sigma=0.001)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        mu    = a * pm.math.exp(b * k_train)
        _     = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_train)
        k_sh  = pm.Data("k_all", k_all)
        mu_p  = pm.Deterministic("mu_pred", a * pm.math.exp(b * k_sh))
        trace = pm.sample(N_SAMPLES, tune=N_TUNE, chains=1, cores=1,
                          nuts_sampler="numpyro", progressbar=True,
                          random_seed=42)
        ppc   = pm.sample_posterior_predictive(trace, var_names=["mu_pred"],
                                               progressbar=False)
    return trace, ppc, train, test, data


# ── 2. Hierarchical model ────────────────────────────────────────────────────
def fit_hierarchical():
    batt_idx = {b: i for i, b in enumerate(batteries)}
    frames   = [df[df["battery_id"]==b].iloc[:int(len(df[df["battery_id"]==b])*TRAIN_FRAC)]
                for b in batteries]
    train_df = pd.concat(frames)
    k_vals   = train_df["cycle"].values.astype(float)
    y_vals   = train_df["SOH"].values.astype(float)
    batt_ids = np.array([batt_idx[b] for b in train_df["battery_id"]])
    nb       = len(batteries)

    with pm.Model():
        mu_a    = pm.Normal("mu_a",    mu=1.0,    sigma=0.05)
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.02)
        mu_b    = pm.Normal("mu_b",    mu=-0.001, sigma=0.0005)
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.0005)
        a_off   = pm.Normal("a_offset", 0, 1, shape=nb)
        b_off   = pm.Normal("b_offset", 0, 1, shape=nb)
        a_i     = pm.Deterministic("a_i", mu_a + a_off * sigma_a)
        b_i     = pm.Deterministic("b_i", mu_b + b_off * sigma_b)
        sigma   = pm.HalfNormal("sigma", sigma=0.02)
        mu      = a_i[batt_ids] * pm.math.exp(b_i[batt_ids] * k_vals)
        _       = pm.Normal("obs", mu=mu, sigma=sigma, observed=y_vals)
        trace   = pm.sample(N_SAMPLES, tune=N_TUNE, chains=1, cores=1,
                            nuts_sampler="numpyro", progressbar=True,
                            random_seed=42)
    return trace


# ── 3. RUL ───────────────────────────────────────────────────────────────────
def compute_rul(ppc, data):
    s   = ppc.posterior_predictive["mu_pred"].values.reshape(-1, len(data))
    cyc = data["cycle"].values
    rul = []
    for row in s:
        below = np.where(row < EOL)[0]
        rul.append(cyc[below[0]] if len(below) else cyc[-1])
    return np.array(rul)


# ── 4. Figures ───────────────────────────────────────────────────────────────
def plot_single(trace, ppc, train, test, data, rul, bid):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Bayesian SOH Estimation — Battery {bid}\n"
                 f"Physics-Informed Data-Driven | Hafid IDRISSI",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    s      = ppc.posterior_predictive["mu_pred"].values.reshape(-1, len(data))
    mu_m   = s.mean(0);  lo = np.percentile(s, 5, 0);  hi = np.percentile(s, 95, 0)
    cyc    = data["cycle"].values

    ax = fig.add_subplot(gs[0,0])
    ax.scatter(train["cycle"], train["SOH"], s=12, c="#2196F3", alpha=.7, label="Train")
    ax.scatter(test["cycle"],  test["SOH"],  s=12, c="#FF9800", alpha=.7, label="Test")
    ax.plot(cyc, mu_m, c="#E53935", lw=2, label="Posterior mean")
    ax.fill_between(cyc, lo, hi, alpha=.25, color="#E53935", label="90% CI")
    ax.axhline(EOL, ls="--", c="gray", lw=1.2, label=f"EOL={EOL}")
    ax.axvline(train["cycle"].max(), ls=":", c="#2196F3", lw=1.2)
    ax.set(xlabel="Cycle", ylabel="SOH", title="SOH Prediction", ylim=(.70,1.05))
    ax.legend(fontsize=7.5)

    ax2 = fig.add_subplot(gs[0,1])
    a_p = trace.posterior["a"].values.flatten()
    b_p = trace.posterior["b"].values.flatten()
    ax2.hist(a_p, bins=40, color="#4CAF50", alpha=.75, density=True,
             label=f"a μ={a_p.mean():.4f}")
    ax2b = ax2.twinx()
    ax2b.hist(b_p*1000, bins=40, color="#9C27B0", alpha=.55, density=True,
              label=f"b×1000 μ={b_p.mean()*1000:.4f}")
    ax2.set(xlabel="Value", ylabel="Density(a)", title="Posterior Parameters")
    h1,l1=ax2.get_legend_handles_labels(); h2,l2=ax2b.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, fontsize=8)

    ax3 = fig.add_subplot(gs[1,0])
    p5,p95 = np.percentile(rul,[5,95])
    ax3.hist(rul, bins=35, color="#FF5722", alpha=.8, density=True, edgecolor="w")
    ax3.axvline(rul.mean(), c="k", lw=2, label=f"Mean {rul.mean():.0f}")
    ax3.axvline(p5,  c="gray", lw=1.5, ls="--", label=f"P5  {p5:.0f}")
    ax3.axvline(p95, c="gray", lw=1.5, ls="-.", label=f"P95 {p95:.0f}")
    ax3.set(xlabel="EOL cycle (SOH<0.80)", ylabel="Density",
            title="RUL Distribution")
    ax3.legend(fontsize=8)
    ax3.text(.97,.95, f"μ={rul.mean():.0f} σ={rul.std():.1f}\n90%CI [{p5:.0f},{p95:.0f}]",
             transform=ax3.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", alpha=.8))

    ax4 = fig.add_subplot(gs[1,1])
    for i in np.random.choice(s.shape[0], min(100,s.shape[0]), replace=False):
        ax4.plot(cyc, s[i], c="#03A9F4", alpha=.05, lw=.8)
    ax4.plot(cyc, mu_m, c="#E53935", lw=2.5, label="Posterior mean", zorder=5)
    ax4.scatter(data["cycle"], data["SOH"], s=8, c="black", alpha=.4,
                label="Observations", zorder=4)
    ax4.axhline(EOL, ls="--", c="gray", lw=1.2)
    ax4.set(xlabel="Cycle", ylabel="SOH", title="Posterior Predictive Check",
            ylim=(.70,1.05))
    ax4.legend(fontsize=8)

    out = os.path.join(OUTPUT_DIR, f"bayesian_soh_{bid}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Figure saved → {out}")


def plot_hierarchical(trace):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    fig.suptitle("Hierarchical Model — Population Parameters\n"
                 "Partial Pooling Across 8 Batteries",
                 fontsize=12, fontweight="bold")
    colors  = plt.cm.tab10(np.linspace(0,1,len(batteries)))
    a_pop   = trace.posterior["mu_a"].values.flatten()
    sa_pop  = trace.posterior["sigma_a"].values.flatten()
    a_i_all = trace.posterior["a_i"].values.reshape(-1, len(batteries))
    b_i_all = trace.posterior["b_i"].values.reshape(-1, len(batteries))

    for i,(b,c) in enumerate(zip(batteries, colors)):
        axes[0].hist(a_i_all[:,i], bins=30, alpha=.5, density=True, color=c, label=b)
    x = np.linspace(a_pop.mean()-4*sa_pop.mean(), a_pop.mean()+4*sa_pop.mean(), 200)
    axes[0].plot(x, norm.pdf(x, a_pop.mean(), sa_pop.mean()), "k--", lw=2, label="Pop.")
    axes[0].set(title="Per-Battery a_i + Population", xlabel="a_i", ylabel="Density")
    axes[0].legend(fontsize=7, ncol=2)

    for i,(b,c) in enumerate(zip(batteries, colors)):
        axes[1].hist(b_i_all[:,i]*1000, bins=30, alpha=.5, density=True, color=c)
    axes[1].set(title="Per-Battery b_i × 1000", xlabel="b_i×1000", ylabel="Density")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "hierarchical_population.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Figure saved → {out}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  STEP 1 — Single-battery (NumPyro backend, ~1 min)")
    print("="*60)
    tr, ppc, trn, tst, dat = fit_single_battery(TARGET_BATT)
    rul = compute_rul(ppc, dat)
    plot_single(tr, ppc, trn, tst, dat, rul, TARGET_BATT)
    print(f"  a={tr.posterior['a'].values.mean():.4f}  "
          f"b={tr.posterior['b'].values.mean():.6f}")
    print(f"  RUL: {rul.mean():.0f} ± {rul.std():.1f} cycles  "
          f"| 90% CI [{np.percentile(rul,5):.0f}, {np.percentile(rul,95):.0f}]")

    print("\n"+"="*60)
    print("  STEP 2 — Hierarchical (8 batteries, ~2 min)")
    print("="*60)
    tr_h = fit_hierarchical()
    plot_hierarchical(tr_h)
    print(f"  μ_a={tr_h.posterior['mu_a'].values.mean():.4f}  "
          f"μ_b={tr_h.posterior['mu_b'].values.mean():.6f}")
    print("\n  ✓ Done — check results/ folder.")
