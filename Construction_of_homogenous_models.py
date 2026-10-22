#!/usr/bin/env python
"""
compute_analytical_models_linear_method2.py

Method 2 (primary): rates and reference fields from pipeline_method2.py
(global mean first, no mask; P_L from the full pooled wet-day variance with a
fixed 1980-2009 baseline; P_L rate fitted with an intercept), *_method2 folders.

Analytical Bernoulli-Gamma exceedance, Models 1, 2, 3 (LINEAR scaling):
  Model 1: P_wet fixed, P_L scales, shape s = P_wet_ref / P_L(dT) drops naturally
  Model 2: P_L + P_wet scale, f_wet fixed
  Model 3: P_L + P_wet + f_wet scale
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import linregress, pearsonr
from scipy.stats import gamma as gamma_dist

sys.path.insert(0, "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP")
import config as cfg

# ── Pipeline run: GLOBAL MEAN FIRST (Method 2, full P_L variance, fixed baseline)
RUN_LABEL = "method2"
SUFFIX    = "_method2"

# ── Settings ──────────────────────────────────────────────────────────────
BASE_DIR    = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP"
GEN_DIR     = os.path.join(BASE_DIR, "general" + SUFFIX)
OUT_DIR     = os.path.join(BASE_DIR, "analytical_models_linear" + SUFFIX)
deltaT_list = np.arange(-1.0, 7.10, 0.1)

# ── Empirical extreme-area file per model (EDIT to match your pipeline) ───
EMP_FILE = os.path.join(GEN_DIR, "{model}", "{model}_area_annual.nc")
EMP_VARS = {"dA": "dA_rel_global", "dT": "dT_annual"}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def load_ref_grid(path, var_candidates):
    """Load (data, lat, lon) from reference NetCDF. Tries multiple var names."""
    if not os.path.exists(path):
        return None, None, None
    ds  = xr.open_dataset(path)
    lat = ds["lat"].values if "lat" in ds.coords else ds["latitude"].values
    lon = ds["lon"].values if "lon" in ds.coords else ds["longitude"].values
    for v in var_candidates:
        if v in ds:
            data = ds[v].values.astype(float)
            ds.close()
            return data, lat, lon
    ds.close()
    return None, None, None


def load_model_rates(model, gen_dir):
    """
    Return dict with fw, pw, pl (+ pl_ic) — per-°C fractional rates derived
    from each model's own CMIP6 pipeline output files.
    """
    rates = {}

    # f_wet rate (OLS vs annual dT)
    f = os.path.join(gen_dir, model, f"{model}_fwet_annual.nc")
    if os.path.exists(f):
        ds   = xr.open_dataset(f)
        mask = np.isfinite(ds["dT_annual"].values) & \
               np.isfinite(ds["fwet_rel_global"].values)
        sl, *_ = linregress(ds["dT_annual"].values[mask],
                            ds["fwet_rel_global"].values[mask])
        rates["fw"] = sl
        ds.close()
    else:
        rates["fw"] = None

    # P_wet rate (OLS vs annual dT) — fractional change of global-mean P_wet
    f = os.path.join(gen_dir, model, f"{model}_Pwet_annual.nc")
    if os.path.exists(f):
        ds   = xr.open_dataset(f)
        mask = np.isfinite(ds["dT_annual"].values) & \
               np.isfinite(ds["Pwet_rel_global"].values)
        sl, *_ = linregress(ds["dT_annual"].values[mask],
                            ds["Pwet_rel_global"].values[mask])
        rates["pw"] = sl
        ds.close()
    else:
        rates["pw"] = None

    # P_L rate (OLS WITH intercept vs 31yr dT; fixed baseline does not force
    # dP_L/P_L = 0 at dT_31yr = 0)
    f_pl   = os.path.join(gen_dir, model, f"{model}_PL_31yr.nc")
    f_taup = os.path.join(gen_dir, model, f"{model}_TauP_31yr.nc")
    if os.path.exists(f_pl) and os.path.exists(f_taup):
        ds_pl   = xr.open_dataset(f_pl)
        ds_taup = xr.open_dataset(f_taup)
        pl_da, dT_da = xr.align(ds_pl["PL_rel"], ds_taup["dT_31yr"], join="inner")
        mask = np.isfinite(pl_da.values) & np.isfinite(dT_da.values)
        fit  = linregress(dT_da.values[mask], pl_da.values[mask])
        rates["pl"]    = fit.slope
        rates["pl_ic"] = fit.intercept      # diagnostic: should be close to 0
        ds_pl.close(); ds_taup.close()
    else:
        rates["pl"] = None

    return rates


def print_multimodel_rates():
    """Multi-model mean scaling rates of P_L, P_wet, f_wet."""
    rows, ics = [], []
    for model in cfg.MODELS:
        r = load_model_rates(model, GEN_DIR)
        if any(r[k] is None for k in ("pl", "pw", "fw")):
            continue
        rows.append((r["pl"] * 100, r["pw"] * 100, r["fw"] * 100))
        ics.append(r["pl_ic"] * 100)
    if not rows:
        print("No complete rate sets found.")
        return
    rows = np.array(rows)
    mm = rows.mean(axis=0)
    print(f"Multi-model mean slopes ({len(rows)} models, run: {RUN_LABEL})")
    print(f"  P_L   : {mm[0]:+.2f} %/°C")
    print(f"  P_wet : {mm[1]:+.2f} %/°C")
    print(f"  f_wet : {mm[2]:+.2f} %/°C")
    print(f"\nP_L fit intercept (check, should be ≈ 0): "
          f"mean {np.mean(ics):+.2f} %, range [{np.min(ics):+.2f}, {np.max(ics):+.2f}] %")


# ══════════════════════════════════════════════════════════════════════════
# Core analytical computation — one CMIP6 model
# ══════════════════════════════════════════════════════════════════════════
def compute_analytical_one_model(model, PL, TauP, fwet, Pwet, lat,
                                 sl_fw, sl_pw, sl_pl, deltaT_list):
    """
    Analytical Models 1, 2, 3 for one model — closed-form Gamma exceedance
    using AREA-WEIGHTED MEAN baseline parameters. Shape a_bar = Pwet_bar/PL_bar
    (ratio of means, consistent with a_dT below). TauP is unused; kept for
    call-signature compatibility.
    """
    nx, ny = PL.shape

    # Area weights
    weights      = np.cos(np.deg2rad(lat))[:, None]
    weights_2d   = np.tile(weights, (1, ny))
    weights_norm = weights_2d / weights_2d.sum()

    def wmean(x):
        valid = np.isfinite(x)
        w = weights_norm[valid]
        return float((x[valid] * w).sum() / w.sum())

    PL_bar   = wmean(PL)
    fwet_bar = wmean(fwet)
    Pwet_bar = wmean(Pwet)
    a_bar    = max(Pwet_bar / PL_bar, 1e-6)

    # Analytical historical P99 threshold
    print(f"    [{model}] Computing analytical P99 threshold (area-weighted mean)...")
    if fwet_bar > 0.01:
        q_needed = 0.01 / fwet_bar
        P99_his = float(gamma_dist.isf(q_needed, a_bar, scale=PL_bar))
    else:
        P99_his = 0.0   # baseline-dry: 1st percentile in dry mass

    def area_frac(PL_, a_, fwet_):
        """Global exceedance probability for given scalar parameter values."""
        if P99_his > 0:
            sf = fwet_ * gamma_dist.sf(P99_his, a_, scale=PL_)
        else:
            sf = fwet_
        return float(np.nan_to_num(sf, nan=0.0))

    # Model 1: P_wet fixed, P_L scales, shape drops naturally
    print(f"    [{model}] Model 1 (P_wet fixed, shape drops naturally)...")
    area_m1 = []
    for dT in deltaT_list:
        PL_dT = PL_bar * (1.0 + sl_pl * dT)
        s_dT  = max(Pwet_bar / PL_dT, 1e-6) if PL_dT > 0 else 1e-6
        area_m1.append(area_frac(PL_dT, s_dT, fwet_bar))
    area_m1 = np.array(area_m1)

    # Models 2 & 3
    print(f"    [{model}] Models 2 & 3 ({len(deltaT_list)} deltaT steps)...")
    area_m2, area_m3 = [], []
    for dT in deltaT_list:
        PL_dT   = PL_bar   * (1.0 + sl_pl * dT)
        Pwet_dT = Pwet_bar * (1.0 + sl_pw * dT)
        fwet_dT = fwet_bar * (1.0 + sl_fw * dT)
        a_dT = max(Pwet_dT / PL_dT, 1e-6) if PL_dT > 0 else 1e-6
        area_m2.append(area_frac(PL_dT, a_dT, fwet_bar))   # f_wet fixed
        area_m3.append(area_frac(PL_dT, a_dT, fwet_dT))    # f_wet scales

    return area_m1, np.array(area_m2), np.array(area_m3)


# ══════════════════════════════════════════════════════════════════════════
# Main loop over models
# ══════════════════════════════════════════════════════════════════════════
def run_models():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 62)
    print("  compute_analytical_models_linear — Models 1, 2, 3")
    print("  Scaling: LINEAR  X(dT) = X_ref * (1 + rate * dT)")
    print("  Method:  analytical Bernoulli-Gamma exceedance (no MC sampling)")
    print("  Model 1: P_wet fixed, P_L scales, shape drops naturally")
    print(f"  run: {RUN_LABEL}  |  input: {GEN_DIR}")
    print(f"  Models: {len(cfg.MODELS)}  |  deltaT steps: {len(deltaT_list)}")
    print("=" * 62)

    for model in cfg.MODELS:
        out_f = os.path.join(OUT_DIR, f"{model}_analytical_models.nc")
        if os.path.exists(out_f):
            print(f"\n  [skip] {model}: output exists")
            continue

        print(f"\n  Processing: {model}")
        gen = os.path.join(GEN_DIR, model)

        PL,   lat, lon = load_ref_grid(os.path.join(gen, f"{model}_PL_ref_1980_2009.nc"),
                                       ["PL_mean", "PL"])
        TauP, _,   _   = load_ref_grid(os.path.join(gen, f"{model}_TauP_ref_1980_2009.nc"),
                                       ["TauP_mean", "TauP"])
        fwet, _,   _   = load_ref_grid(os.path.join(gen, f"{model}_fwet_ref_1980_2009.nc"),
                                       ["fwet_mean", "fwet"])
        Pwet, _,   _   = load_ref_grid(os.path.join(gen, f"{model}_Pwet_ref_1980_2009.nc"),
                                       ["Pwet_mean", "Pwet"])

        if any(v is None for v in [PL, TauP, fwet, Pwet]):
            print(f"  [warn] {model}: missing reference file(s) - skipping")
            continue

        rates = load_model_rates(model, GEN_DIR)
        sl_fw, sl_pw, sl_pl = rates["fw"], rates["pw"], rates["pl"]
        if any(r is None for r in [sl_fw, sl_pw, sl_pl]):
            print(f"  [warn] {model}: missing rate file(s) - skipping")
            continue

        print(f"    Rates:  sl_fw={sl_fw*100:+.3f}  "
              f"sl_pw={sl_pw*100:+.3f}  sl_pl={sl_pl*100:+.3f} %/°C")

        try:
            area_m1, area_m2, area_m3 = compute_analytical_one_model(
                model, PL, TauP, fwet, Pwet, lat, sl_fw, sl_pw, sl_pl, deltaT_list)
        except Exception as e:
            import traceback
            print(f"  [FAIL] {model}: {e}")
            traceback.print_exc()
            continue

        # Baseline (deltaT ~ 0) for relative change
        idx0 = np.argmin(np.abs(deltaT_list))
        dA_m1 = (area_m1 - area_m1[idx0]) / area_m1[idx0]
        dA_m2 = (area_m2 - area_m2[idx0]) / area_m2[idx0]
        dA_m3 = (area_m3 - area_m3[idx0]) / area_m3[idx0]

        ds_out = xr.Dataset(
            {
                "area_m1": (["deltaT"], area_m1),
                "area_m2": (["deltaT"], area_m2),
                "area_m3": (["deltaT"], area_m3),
                "dA_m1":   (["deltaT"], dA_m1),
                "dA_m2":   (["deltaT"], dA_m2),
                "dA_m3":   (["deltaT"], dA_m3),
            },
            coords={"deltaT": deltaT_list},
        )
        for v in ["area_m1", "area_m2", "area_m3"]:
            ds_out[v].attrs = {"long_name": f"Fraction exceeding P99 ({v}, analytical)",
                               "units": "fraction"}
        for v in ["dA_m1", "dA_m2", "dA_m3"]:
            ds_out[v].attrs = {"long_name": f"Relative change in extreme area ({v}, analytical)",
                               "units": "1 (fractional)"}
        ds_out["area_m1"].attrs["description"] = (
            "P_wet fixed at baseline; P_L scales at sl_pl; shape "
            "s(dT) = P_wet_ref / P_L(dT) drops naturally; f_wet fixed")
        ds_out.attrs = {
            "model"            : model,
            "sl_fw_pct"        : round(sl_fw * 100, 4),
            "sl_pw_pct"        : round(sl_pw * 100, 4),
            "sl_pl_pct"        : round(sl_pl * 100, 4),
            "description"      : "Analytical Bernoulli-Gamma exceedance, Models 1-2-3, LINEAR growth rate",
            "model1_definition": "P_wet fixed, P_L scales, shape drops naturally (f_wet fixed)",
            "scaling"          : "linear: X(dT) = X_ref * (1 + rate * dT)",
            "method"           : "closed-form gamma.sf exceedance (no Monte Carlo sampling)",
            "run"              : RUN_LABEL,
            "input_dir"        : GEN_DIR,
        }
        ds_out.to_netcdf(out_f)
        print(f"  [done] {model}: saved -> {out_f}")

    n_done = sum(os.path.exists(os.path.join(OUT_DIR, f"{m}_analytical_models.nc"))
                 for m in cfg.MODELS)
    print(f"\n{'=' * 62}")
    print(f"  Complete: {n_done} / {len(cfg.MODELS)} models")
    print(f"  Output: {OUT_DIR}")
    print("=" * 62)


# ══════════════════════════════════════════════════════════════════════════
# Figure: multi-model mean of homogeneous Models 1–3
# ══════════════════════════════════════════════════════════════════════════
def rate_pct_per_K(deltaT_, dA_pct):
    """OLS slope of dA (%) vs deltaT (°C) -> %/°C."""
    sl, *_ = linregress(deltaT_, dA_pct)
    return sl


def plot_homogeneous_comparison():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_analytical_models.nc")))
    if not files:
        raise FileNotFoundError(f"No analytical output files found in {OUT_DIR}")
    print(f"Found {len(files)} model output files")

    ds_list = [xr.open_dataset(f) for f in files]
    deltaT  = ds_list[0]["deltaT"].values
    dA_m1_all = np.stack([ds["dA_m1"].values for ds in ds_list], axis=0)
    dA_m2_all = np.stack([ds["dA_m2"].values for ds in ds_list], axis=0)
    dA_m3_all = np.stack([ds["dA_m3"].values for ds in ds_list], axis=0)
    for ds in ds_list:
        ds.close()

    dA_m1_mm = np.nanmean(dA_m1_all, axis=0) * 100
    dA_m2_mm = np.nanmean(dA_m2_all, axis=0) * 100
    dA_m3_mm = np.nanmean(dA_m3_all, axis=0) * 100

    r1 = rate_pct_per_K(deltaT, dA_m1_mm)
    r2 = rate_pct_per_K(deltaT, dA_m2_mm)
    r3 = rate_pct_per_K(deltaT, dA_m3_mm)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(deltaT, dA_m1_mm, "--", color="violet", lw=1.6,
            label=fr"Homogeneous $P_L$-only ($\bar P_{{wet}}$ fixed) ({r1:.2f} %/°C)")
    ax.plot(deltaT, dA_m2_mm, "--", color="darkorange", lw=1.6,
            label=fr"Homogeneous $P_L + \bar P_{{wet}}$ ({r2:.2f} %/°C)")
    ax.plot(deltaT, dA_m3_mm, "-", color="green", lw=1.8,
            label=fr"Homogeneous $P_L + \bar P_{{wet}} + f_{{wet}}$ ({r3:.2f} %/°C)")
    ax.axhline(0, color="0.6", lw=0.6, zorder=0)
    ax.axvline(0, color="0.6", lw=0.6, zorder=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xlabel(r"$\Delta T$ (°C)")
    ax.set_ylabel(r"$\delta A / A_0$ (%)")
    ax.set_xlim(deltaT.min(), deltaT.max())
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_title("(c)", loc="left", fontweight="bold")
    ax.set_title(f"run: {RUN_LABEL}", loc="right", fontsize=9)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "homogeneous_models_comparison.png")
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")


# ══════════════════════════════════════════════════════════════════════════
# Per-model check: empirical vs homogeneous extreme-area sensitivity
# ══════════════════════════════════════════════════════════════════════════
def empirical_rate(model):
    """OLS slope (with intercept) of empirical dA/A0 vs dT -> %/°C."""
    f = EMP_FILE.format(model=model)
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)
    dA = ds[EMP_VARS["dA"]].values.astype(float)
    dT = ds[EMP_VARS["dT"]].values.astype(float)
    ds.close()
    m = np.isfinite(dA) & np.isfinite(dT)
    return linregress(dT[m], dA[m]).slope * 100


def homogeneous_rate(model):
    """OLS slope of homogeneous Model 3 dA/A0 vs deltaT -> %/°C."""
    f = os.path.join(OUT_DIR, f"{model}_analytical_models.nc")
    if not os.path.exists(f):
        return None
    ds = xr.open_dataset(f)
    sl = linregress(ds["deltaT"].values, ds["dA_m3"].values * 100).slope
    ds.close()
    return sl


def compare_empirical_vs_homogeneous():
    rows = []
    for model in cfg.MODELS:
        g_hom = homogeneous_rate(model)
        g_emp = empirical_rate(model)
        if g_hom is None or g_emp is None:
            print(f"  [skip] {model}: missing homogeneous or empirical file")
            continue
        rows.append({"model": model, "gA_emp": g_emp, "gA_hom": g_hom})

    if len(rows) < 2:
        print(f"\n[stop] Only {len(rows)} model(s) with both files — "
              f"check EMP_FILE / EMP_VARS (currently {EMP_FILE}).")
        return

    df = pd.DataFrame(rows)
    df["diff"]  = df.gA_hom - df.gA_emp
    df["ratio"] = df.gA_hom / df.gA_emp

    r, p   = pearsonr(df.gA_emp, df.gA_hom)
    rmse   = np.sqrt(np.mean(df["diff"] ** 2))
    bias   = df["diff"].mean()
    sd_emp = df.gA_emp.std(ddof=1)
    within = np.mean(np.abs(df.ratio - 1) <= 0.10) * 100

    print(df.sort_values("gA_emp").round(2).to_string(index=False))
    print(f"\nModels          : {len(df)}")
    print(f"MMM empirical   : {df.gA_emp.mean():.2f} %/°C")
    print(f"MMM homogeneous : {df.gA_hom.mean():.2f} %/°C")
    print(f"Pearson r       : {r:.2f} (p = {p:.1e})")
    print(f"Bias (hom-emp)  : {bias:+.2f} %/°C")
    print(f"RMSE            : {rmse:.2f} %/°C  (inter-model SD of empirical = {sd_emp:.2f})")
    print(f"Ratio hom/emp   : median {df.ratio.median():.2f}, "
          f"range {df.ratio.min():.2f}–{df.ratio.max():.2f}")
    print(f"Models within ±10% : {within:.0f}%")

    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(df.gA_emp.min(), df.gA_hom.min()) - 1
    hi = max(df.gA_emp.max(), df.gA_hom.max()) + 1
    ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                    color="0.88", label="±10%")
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
    ax.scatter(df.gA_emp, df.gA_hom, s=30, c="steelblue", zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row.model, (row.gA_emp, row.gA_hom), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    ax.scatter(df.gA_emp.mean(), df.gA_hom.mean(), marker="*", s=180,
               c="crimson", zorder=4, label="Multi-model mean")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Empirical $\gamma_A$ (% °C$^{-1}$)")
    ax.set_ylabel(r"Homogeneous $\gamma_A$ (% °C$^{-1}$)")
    ax.set_title(f"r = {r:.2f}, RMSE = {rmse:.2f} % °C$^{{-1}}$", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "per_model_homogeneous_vs_empirical.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Run                 : {RUN_LABEL} (Method 2)")
    print(f"Reading from        : {GEN_DIR}")
    print(f"Writing to          : {OUT_DIR}\n")

    print_multimodel_rates()
    run_models()
    plot_homogeneous_comparison()
    compare_empirical_vs_homogeneous()
