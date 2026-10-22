#!/usr/bin/env python
"""
censored_vs_uncensored_comparison_method2.py

Method 2 version — inputs from general_method2/ (censored), no_threshold_method2/
(uncensored) and multimodel_mean_method2/. FG curves use the homogeneous reduction.

Sections
  1. Paths, model list, axes
  2. Check no-threshold files
  3. Utilities
  4. CMIP6 empirical extreme area
  5. Per-model homogeneous FG (censored + uncensored)
  6. Elasticity tables (uncensored Method A, censored, uncensored Method B)
  7. Multimodel statistics + 4-curve comparison plot
  8. Elasticities vs percentile (per model, MM stats, plot)
  9. Analytic alpha2/alpha3 vs percentile (shaded)
 10. Significance tests (paired, slope) + difference plot
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import mpmath as mp
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special   import gammaincc, gammainccinv, gamma as Gamma_func, digamma
from scipy.integrate import quad
from scipy.stats     import linregress, ttest_rel, wilcoxon
from scipy.stats     import gamma as gamma_dist, t as t_dist

plt.rcParams["figure.dpi"]       = 120
plt.rcParams["text.usetex"]      = False
plt.rcParams["font.size"]        = 25
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"]      = "STIXGeneral"

# ══════════════════════════════════════════════════════════════════════════
# 1. Paths, model list, axes
# ══════════════════════════════════════════════════════════════════════════
BASE_DIR   = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP"
MM_DIR     = os.path.join(BASE_DIR, "multimodel_mean_method2")
GEN_DIR    = os.path.join(BASE_DIR, "general_method2")        # censored, P0 = 0.1
NTH_DIR    = os.path.join(BASE_DIR, "no_threshold_method2")   # uncensored
EMP_DIR    = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2"
FIG_DIR    = "/nfs_home/dmuriar/PRECIP_AREA/ANALYTICAL/UNCENSORED"
FIG_SUFFIX = "_method2"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_START = 1980
BASELINE_END   = 2009

deltaT_plot = np.arange(-1.0, 7.10, 0.1)
COMMON_X    = np.linspace(-1.0, 7.0, 300)

sys.path.insert(0, BASE_DIR)
import config as cfg
EXCLUDE = ["CNRM_CM61"]     # duplicate of CNRM_CM6

def f_emp(m): return os.path.join(EMP_DIR, f"tas_vs_extreme_precip_{m}.nc")
def f_gen(m): return os.path.join(GEN_DIR, m, f"{m}_fwet_annual.nc")
def f_nth(m): return os.path.join(NTH_DIR, m, f"{m}_fwet_annual_no_thresh.nc")

REQUIRED = [("CMIP6 empirical", f_emp), ("censored (general_method2)", f_gen),
            ("uncensored (no_threshold_method2)", f_nth)]
MODELS = [m for m in cfg.MODELS
          if m not in EXCLUDE and all(os.path.exists(p(m)) for _, p in REQUIRED)]

print(f"Models: {len(MODELS)}  (excluded: {', '.join(EXCLUDE) or 'none'}; "
      f"Method 2 inputs from {GEN_DIR})")
for name, p in REQUIRED:
    miss = [m for m in cfg.MODELS if m not in EXCLUDE and not os.path.exists(p(m))]
    if miss:
        print(f"  missing {name}: {', '.join(miss)}  -> left out")

# ══════════════════════════════════════════════════════════════════════════
# 2. Check no-threshold files
# ══════════════════════════════════════════════════════════════════════════
need = ["PL_31yr_no_thresh", "Pwet_annual_no_thresh", "fwet_annual_no_thresh",
        "PL_ref_1980_2009_no_thresh", "Pwet_ref_1980_2009_no_thresh",
        "fwet_ref_1980_2009_no_thresh"]
for suffix in need:
    missing = [m for m in MODELS
               if not os.path.exists(os.path.join(NTH_DIR, m, f"{m}_{suffix}.nc"))]
    if missing:
        print(f"⚠  {suffix}: {len(missing)} model(s) missing: {missing}")
    else:
        print(f"✓  {suffix}: all {len(MODELS)} models")

# ══════════════════════════════════════════════════════════════════════════
# 3. Utilities
# ══════════════════════════════════════════════════════════════════════════
def load_grid(path, var_candidates):
    """Load a 2D gridded field from NetCDF, returning (data, lat, lon)."""
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


def load_grid_unc(nth_path, gen_path, var_candidates):
    """Load uncensored gridded field, falling back to censored if absent."""
    data, lat, lon = load_grid(nth_path, var_candidates)
    if data is None:
        data, lat, lon = load_grid(gen_path, var_candidates)
    return data, lat, lon


def wmean(field, lat):
    """cos(lat)-weighted global mean over finite cells."""
    w  = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, field.shape[1]))
    ok = np.isfinite(field)
    return float((field[ok] * w[ok]).sum() / w[ok].sum())


def _ols(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return linregress(x[ok], y[ok]).slope


def _pl_rate(pl_file, gen):
    """P_L rate: OLS WITH intercept vs dT_31yr (Method 2, fixed baseline)."""
    model   = os.path.basename(gen)
    ds_pl   = xr.open_dataset(pl_file)
    ds_taup = xr.open_dataset(os.path.join(gen, f"{model}_TauP_31yr.nc"))
    pl, dT  = xr.align(ds_pl["PL_rel"], ds_taup["dT_31yr"], join="inner")
    sl = _ols(dT.values, pl.values)
    ds_pl.close(); ds_taup.close()
    return sl


def get_rates_cen(model, gen_dir):
    """Censored (P0 = 0.1) per-°C rates, Method 2."""
    gen = os.path.join(gen_dir, model)
    with xr.open_dataset(os.path.join(gen, f"{model}_fwet_annual.nc")) as ds:
        sl_fw = _ols(ds["dT_annual"].values, ds["fwet_rel_global"].values)
    with xr.open_dataset(os.path.join(gen, f"{model}_Pwet_annual.nc")) as ds:
        sl_pw = _ols(ds["dT_annual"].values, ds["Pwet_rel_global"].values)
    sl_pl = _pl_rate(os.path.join(gen, f"{model}_PL_31yr.nc"), gen)
    return sl_fw, sl_pw, sl_pl


def get_rates_unc(model, gen_dir, nth_dir):
    """Uncensored (P0 → 0) per-°C rates, Method 2 no-threshold run."""
    nth = os.path.join(nth_dir, model)
    gen = os.path.join(gen_dir, model)
    with xr.open_dataset(os.path.join(nth, f"{model}_fwet_annual_no_thresh.nc")) as ds:
        sl_fw = _ols(ds["dT_annual"].values, ds["fwet_rel_global"].values)
    with xr.open_dataset(os.path.join(nth, f"{model}_Pwet_annual_no_thresh.nc")) as ds:
        sl_pw = _ols(ds["dT_annual"].values, ds["Pwet_rel_global"].values)
    sl_pl = _pl_rate(os.path.join(nth, f"{model}_PL_31yr_no_thresh.nc"), gen)
    return sl_fw, sl_pw, sl_pl


def homog_dA(PL0, Pw0, fw0, rates, exceed=0.01):
    """Homogeneous (single representative Gamma) δA/A0 in % on deltaT_plot."""
    sl_fw, sl_pw, sl_pl = rates
    s0 = max(Pw0 / PL0, 1e-6)
    q0 = np.clip(exceed / fw0, 1e-12, 1 - 1e-12)
    u0 = float(gammainccinv(s0, q0))
    A  = []
    for dT in deltaT_plot:
        PL = PL0 * (1 + sl_pl * dT)
        Pw = Pw0 * (1 + sl_pw * dT)
        fw = fw0 * (1 + sl_fw * dT)
        s  = max(Pw / PL, 1e-6)
        A.append(fw * gammaincc(s, u0 * PL0 / PL))
    A  = np.array(A)
    i0 = int(np.argmin(np.abs(deltaT_plot)))
    return (A - A[i0]) / A[i0] * 100


def fg_mm_stats(curves, dT_full, dT_slope_end=7.00):
    mean = np.nanmean(curves, axis=0)
    p05  = np.nanpercentile(curves,  5, axis=0)
    p95  = np.nanpercentile(curves, 95, axis=0)
    fit_mask = dT_full <= dT_slope_end + 1e-9
    sl, ic, r, _, _ = linregress(dT_full[fit_mask], mean[fit_mask])
    return mean, p05, p95, sl, ic, r


def wm_field(arr, lat):
    """cos(lat)-weighted mean using nansum (same as the notebook's wm())."""
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, arr.shape[1]))
    valid = np.isfinite(arr)
    return float(np.nansum(arr * w * valid) / np.nansum(w * valid))


def pwet_unc_scalar(model):
    """Global mean of the uncensored P̄_wet baseline map (no fallback)."""
    fp = os.path.join(NTH_DIR, model, f"{model}_Pwet_ref_1980_2009_no_thresh.nc")
    field, lat, _ = load_grid(fp, ["Pwet_mean", "Pwet"])
    return np.nan if field is None else wm_field(field, lat)


def ref_params_unc(model):
    """(PL_ref, fwet_ref, Pwet_ref) uncensored, or None if missing/non-positive."""
    gen = os.path.join(GEN_DIR, model)
    nth = os.path.join(NTH_DIR, model)
    PL_u, lat, _ = load_grid_unc(
        os.path.join(nth, f"{model}_PL_ref_1980_2009_no_thresh.nc"),
        os.path.join(gen, f"{model}_PL_ref_1980_2009.nc"), ["PL_mean", "PL"])
    fwet_u, _, _ = load_grid_unc(
        os.path.join(nth, f"{model}_fwet_ref_1980_2009_no_thresh.nc"),
        os.path.join(gen, f"{model}_fwet_ref_1980_2009.nc"), ["fwet_mean", "fwet"])
    Pwet_ref = pwet_unc_scalar(model)
    if PL_u is None or fwet_u is None or not np.isfinite(Pwet_ref):
        print(f"  ⚠  {model}: missing reference — skipping")
        return None
    PL_ref, fwet_ref = wm_field(PL_u, lat), wm_field(fwet_u, lat)
    if PL_ref <= 0 or Pwet_ref <= 0 or fwet_ref <= 0:
        print(f"  ⚠  {model}: non-positive reference value — skipping")
        return None
    return PL_ref, fwet_ref, Pwet_ref


def u_from_quantile(s, q):
    return float(gammainccinv(np.array([s]), np.array([q]))[0])


def elasticities(s_ref, u_ref):
    """Exact α1, α2, α3 for a Gamma(s) exceedance above u."""
    G_su    = Gamma_func(s_ref) * float(gammaincc(np.array([s_ref]), np.array([u_ref]))[0])
    T_su, _ = quad(lambda t: t**(s_ref - 1) * np.log(t) * np.exp(-t),
                   u_ref, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10)
    psi_s   = float(digamma(s_ref))
    alpha1  = 1.0
    alpha2  = s_ref * (T_su / G_su - psi_s)
    alpha3  = s_ref * (psi_s - T_su / G_su) + u_ref**s_ref * np.exp(-u_ref) / G_su
    return alpha1, alpha2, alpha3, u_ref**s_ref * np.exp(-u_ref) / G_su


def print_latex_table(df, mm, caption, label):
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\begin{tabular}{lccc}")
    print("\\hline\\hline")
    print("Model & $\\alpha_1$ & $\\alpha_2$ & $\\alpha_3$ \\\\")
    print("\\hline")
    for idx, row in df.iterrows():
        print(f"{idx} & {row['α₁']:.2f} & {row['α₂']:.2f} & {row['α₃']:.2f} \\\\")
    print("\\hline")
    print(f"\\textbf{{MM Mean}} & \\textbf{{{mm['α₁']:.2f}}} & "
          f"\\textbf{{{mm['α₂']:.2f}}} & \\textbf{{{mm['α₃']:.2f}}} \\\\")
    print("\\hline\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


print("Utilities ready.")

# ══════════════════════════════════════════════════════════════════════════
# 4. CMIP6 empirical extreme area
# ══════════════════════════════════════════════════════════════════════════
print(f"CMIP6 empirical files: {len(MODELS)} models from {EMP_DIR}")

fits_ea, A0_list = [], []
X_ea = np.linspace(-1.0, 6.90, 300)

for m in MODELS:
    ds   = xr.open_dataset(f_emp(m))
    tas  = ds["tas_anomaly"].values
    frac = ds["fraction_extreme_area"].values
    year = ds["time"].values                 # real years
    ds.close()
    bm   = (year >= BASELINE_START) & (year <= BASELINE_END)
    if bm.sum() == 0:
        continue
    A0   = np.nanmean(frac[bm]); A0_list.append(A0)
    rel  = (frac - A0) / A0
    mask = np.isfinite(tas) & np.isfinite(rel)
    sl, ic, *_ = linregress(tas[mask], rel[mask])
    fits_ea.append(sl * X_ea + ic)

fits_ea     = np.array(fits_ea)
ea_mean     = np.nanmean(fits_ea, axis=0)
ea_p05      = np.nanpercentile(fits_ea,  5, axis=0)
ea_p95      = np.nanpercentile(fits_ea, 95, axis=0)
A0_model_mm = np.nanmean(A0_list)

sl_cmip6, ic_cmip6, r_cmip6, _, _ = linregress(X_ea, ea_mean * 100)
print(f"A0 multimodel mean = {A0_model_mm:.6f}")
print(f"CMIP6 slope        = {sl_cmip6:.3f} %/K  (R = {r_cmip6:.3f})")

# ══════════════════════════════════════════════════════════════════════════
# 5. Per-model homogeneous FG (censored + uncensored)
# ══════════════════════════════════════════════════════════════════════════
fg_curves_cen, fg_curves_unc = [], []
fg_rates_log, shift_log = {}, {}

for model in MODELS:
    gen = os.path.join(GEN_DIR, model)
    nth = os.path.join(NTH_DIR, model)

    PL_c,   lat, _ = load_grid(os.path.join(gen, f"{model}_PL_ref_1980_2009.nc"),   ["PL_mean", "PL"])
    fwet_c, _,   _ = load_grid(os.path.join(gen, f"{model}_fwet_ref_1980_2009.nc"), ["fwet_mean", "fwet"])
    Pwet_c, _,   _ = load_grid(os.path.join(gen, f"{model}_Pwet_ref_1980_2009.nc"), ["Pwet_mean", "Pwet"])

    PL_u,   lat_u, _ = load_grid(os.path.join(nth, f"{model}_PL_ref_1980_2009_no_thresh.nc"),   ["PL_mean", "PL"])
    fwet_u, _,     _ = load_grid(os.path.join(nth, f"{model}_fwet_ref_1980_2009_no_thresh.nc"), ["fwet_mean", "fwet"])
    Pwet_u, _,     _ = load_grid(os.path.join(nth, f"{model}_Pwet_ref_1980_2009_no_thresh.nc"), ["Pwet_mean", "Pwet"])

    if any(v is None for v in [PL_c, fwet_c, Pwet_c, PL_u, fwet_u, Pwet_u]):
        print(f"  ⚠  {model}: missing reference file — skipping")
        continue

    try:
        r_c = get_rates_cen(model, GEN_DIR)
        r_u = get_rates_unc(model, GEN_DIR, NTH_DIR)
    except Exception as e:
        print(f"  ⚠  {model}: rate loading failed ({e}) — skipping")
        continue

    fg_rates_log[model] = dict(
        sl_fw_c=r_c[0]*100, sl_pw_c=r_c[1]*100, sl_pl_c=r_c[2]*100,
        sl_fw_u=r_u[0]*100, sl_pw_u=r_u[1]*100, sl_pl_u=r_u[2]*100)

    PL0_c, fw0_c, Pw0_c = wmean(PL_c, lat), wmean(fwet_c, lat), wmean(Pwet_c, lat)
    PL0_u, fw0_u, Pw0_u = wmean(PL_u, lat_u), wmean(fwet_u, lat_u), wmean(Pwet_u, lat_u)

    dA_c = homog_dA(PL0_c, Pw0_c, fw0_c, r_c, exceed=0.01)
    dA_u = homog_dA(PL0_u, Pw0_u, fw0_u, r_u, exceed=0.01)

    sl_c = linregress(deltaT_plot, dA_c).slope
    sl_u = linregress(deltaT_plot, dA_u).slope
    shift_log[model] = dict(offset=0.0, sl_raw_cen=sl_c, sl_raw_unc=sl_u, sl_sft_unc=sl_u,
                            s_c=Pw0_c / PL0_c, s_u=Pw0_u / PL0_u, fw_c=fw0_c, fw_u=fw0_u)

    print(f"  ✓  {model:<18} s_c={Pw0_c/PL0_c:.3f}  s_u={Pw0_u/PL0_u:.3f}  "
          f"slope cen={sl_c:.2f}  unc={sl_u:.2f} %/°C")

    fg_curves_cen.append(dA_c)
    fg_curves_unc.append(dA_u)

fg_curves_cen = np.array(fg_curves_cen)
fg_curves_unc = np.array(fg_curves_unc)

sl_c_all = np.array([v["sl_raw_cen"] for v in shift_log.values()])
sl_u_all = np.array([v["sl_raw_unc"] for v in shift_log.values()])
print(f"\n{'='*60}")
print(f"  Homogeneous FG, {len(sl_c_all)} models (slopes in %/°C)")
print(f"  censored   (P0 = 0.1): mean {sl_c_all.mean():.2f}  [{sl_c_all.min():.2f}, {sl_c_all.max():.2f}]")
print(f"  uncensored (P0 → 0)  : mean {sl_u_all.mean():.2f}  [{sl_u_all.min():.2f}, {sl_u_all.max():.2f}]")
print(f"  uncensored − censored: {np.mean(sl_u_all - sl_c_all):+.2f}")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════════════════
# 6a. Elasticity table — uncensored, Method A (u = Pr/PL), B as fallback
# ══════════════════════════════════════════════════════════════════════════
alpha_rows = []
for model in MODELS:
    gen = os.path.join(GEN_DIR, model)
    ref = ref_params_unc(model)
    if ref is None:
        continue
    PL_ref, fwet_ref, Pwet_ref = ref

    # P99 gridded field for Method A
    P99_ref_scalar = np.nan
    fp_exceed = os.path.join(gen, f"{model}_exceedance_gridded.nc")
    if os.path.exists(fp_exceed):
        field, lat_ex, _ = load_grid(fp_exceed, ["P99_ref", "P99", "pr_p99"])
        if field is not None:
            P99_ref_scalar = wm_field(field, lat_ex)

    s_ref = Pwet_ref / PL_ref
    if np.isfinite(P99_ref_scalar) and P99_ref_scalar > 0:
        u_ref, method = P99_ref_scalar / PL_ref, "A (Pr/PL)"
    else:
        q_ref = np.clip(0.01 / fwet_ref, 1e-10, 1 - 1e-10)
        u_ref, method = u_from_quantile(s_ref, q_ref), "B (gammainccinv fallback)"
        print(f"  ⚠  {model}: P99 file missing — using Method B for u")

    try:
        a1, a2, a3, rhs = elasticities(s_ref, u_ref)
        lhs = a2 + a3
        ok  = abs(lhs - rhs) < 1e-6
        alpha_rows.append({"Model": model, "s_ref": round(s_ref, 4), "u_ref": round(u_ref, 4),
                           "α₁": round(a1, 4), "α₂": round(a2, 4), "α₃": round(a3, 4),
                           "α₂+α₃": round(lhs, 4), "check": "✓" if ok else "✗",
                           "u_method": method})
        print(f"  ✓  {model:<22}  s={s_ref:.4f}  u={u_ref:.4f} [{method}]  "
              f"α₂={a2:.4f}  α₃={a3:.4f}  α₂+α₃={lhs:.4f}  check={'✓' if ok else '✗'}")
    except Exception as e:
        print(f"  ⚠  {model}: alpha computation failed ({e}) — skipping")

df = pd.DataFrame(alpha_rows).set_index("Model")
mm_row = pd.DataFrame([{
    "s_ref": round(df["s_ref"].mean(), 4), "u_ref": round(df["u_ref"].mean(), 4),
    "α₁": round(df["α₁"].mean(), 4), "α₂": round(df["α₂"].mean(), 4),
    "α₃": round(df["α₃"].mean(), 4), "α₂+α₃": round(df["α₂+α₃"].mean(), 4),
    "check": "", "u_method": ""}], index=["MM Mean"])
df_full = pd.concat([df, mm_row])

print(f"\n{'='*75}")
print("  Elasticity coefficients per model  (uncensored, Method A: u=Pr/PL)")
print(f"{'='*75}")
print(f"  {'Model':<22}  {'s_ref':>6}  {'u_ref':>6}  "
      f"{'α₁':>6}  {'α₂':>7}  {'α₃':>7}  {'α₂+α₃':>7}  ✓")
print(f"  {'─'*68}")
for idx, row in df_full.iterrows():
    if idx == "MM Mean":
        print(f"  {'─'*68}")
    print(f"  {idx:<22}  {row['s_ref']:>6.4f}  {row['u_ref']:>6.4f}  "
          f"{row['α₁']:>6.4f}  {row['α₂']:>7.4f}  {row['α₃']:>7.4f}  "
          f"{row['α₂+α₃']:>7.4f}  {row['check']}")
print(f"{'='*75}\n")

print("\n% ── LaTeX table (uncensored, Method A) ──")
print_latex_table(df, df_full.loc["MM Mean"],
                  "Elasticity coefficients $\\alpha_1$, $\\alpha_2$, $\\alpha_3$ "
                  "per model (uncensored, $P_0 \\to 0$, $u_0 = P_r/P_L$)",
                  "tab:alphas_unc")

# ══════════════════════════════════════════════════════════════════════════
# 6b. Elasticity table — censored
# ══════════════════════════════════════════════════════════════════════════
alpha_rows_cen = []
for model in MODELS:
    gen = os.path.join(GEN_DIR, model)
    PL_c,   lat, _ = load_grid(os.path.join(gen, f"{model}_PL_ref_1980_2009.nc"),   ["PL_mean", "PL"])
    fwet_c, _,   _ = load_grid(os.path.join(gen, f"{model}_fwet_ref_1980_2009.nc"), ["fwet_mean", "fwet"])
    Pwet_c, _,   _ = load_grid(os.path.join(gen, f"{model}_Pwet_ref_1980_2009.nc"), ["Pwet_mean", "Pwet"])
    if any(v is None for v in [PL_c, fwet_c, Pwet_c]):
        print(f"  ⚠  {model}: missing censored reference — skipping")
        continue

    PL_ref, fwet_ref, Pwet_ref = wm_field(PL_c, lat), wm_field(fwet_c, lat), wm_field(Pwet_c, lat)
    if PL_ref <= 0 or Pwet_ref <= 0 or fwet_ref <= 0:
        print(f"  ⚠  {model}: non-positive censored reference — skipping")
        continue

    s_ref = Pwet_ref / PL_ref
    q_ref = np.clip(0.01 / fwet_ref, 1e-10, 1 - 1e-10)
    try:
        u_ref = u_from_quantile(s_ref, q_ref)
    except Exception as e:
        print(f"  ⚠  {model}: gammainccinv failed ({e}) — skipping")
        continue

    try:
        a1, a2, a3, _ = elasticities(s_ref, u_ref)
        alpha_rows_cen.append({"Model": model, "α₁": round(a1, 4),
                               "α₂": round(a2, 4), "α₃": round(a3, 4)})
        print(f"  ✓  {model:<22}  α₁={a1:.4f}  α₂={a2:.4f}  α₃={a3:.4f}")
    except Exception as e:
        print(f"  ⚠  {model}: alpha computation failed ({e}) — skipping")

df_cen = pd.DataFrame(alpha_rows_cen).set_index("Model")
mm_row_cen = pd.DataFrame([{c: round(df_cen[c].mean(), 4) for c in ["α₁", "α₂", "α₃"]}],
                          index=["MM Mean"])
df_cen_full = pd.concat([df_cen, mm_row_cen])

print(f"\n{'='*52}")
print("  Elasticity coefficients per model  (censored, P₀=0.1 mm/day)")
print(f"{'='*52}")
print(f"  {'Model':<22}  {'α₁':>7}  {'α₂':>7}  {'α₃':>7}")
print(f"  {'─'*42}")
for idx, row in df_cen_full.iterrows():
    if idx == "MM Mean":
        print(f"  {'─'*42}")
    print(f"  {idx:<22}  {row['α₁']:>7.4f}  {row['α₂']:>7.4f}  {row['α₃']:>7.4f}")
print(f"{'='*52}\n")

print("\n% ── LaTeX table (censored) ──")
print_latex_table(df_cen, df_cen_full.loc["MM Mean"],
                  "Elasticity coefficients $\\alpha_1$, $\\alpha_2$, $\\alpha_3$ "
                  "per model (censored, $P_0 = 0.1$ mm day$^{-1}$)",
                  "tab:alphas_cen")

# ══════════════════════════════════════════════════════════════════════════
# 7. Multimodel statistics + four-curve comparison plot
# ══════════════════════════════════════════════════════════════════════════
fg_cen_mean, fg_cen_p05, fg_cen_p95, sl_fg_cen, ic_fg_cen, r_fg_cen = \
    fg_mm_stats(fg_curves_cen, deltaT_plot, dT_slope_end=7.00)
fg_unc_mean, fg_unc_p05, fg_unc_p95, sl_fg_unc, ic_fg_unc, r_fg_unc = \
    fg_mm_stats(fg_curves_unc, deltaT_plot, dT_slope_end=7.00)

rates_cen = np.array([[v["sl_fw_c"], v["sl_pw_c"], v["sl_pl_c"]] for v in fg_rates_log.values()])
rates_unc = np.array([[v["sl_fw_u"], v["sl_pw_u"], v["sl_pl_u"]] for v in fg_rates_log.values()])

print(f"\n{'='*60}")
print(f"{'Curve':<38}  {'Slope (%/K)':>10}    R")
print(f"  {'-'*52}")
print(f"  {'CMIP6':<36}  {sl_cmip6:>10.3f}   {r_cmip6:.3f}")
print(f"  {'-'*52}")
print(f"  {'FG censored   (P0=0.1)':<36}  {sl_fg_cen:>10.3f}   {r_fg_cen:.3f}")
print(f"  {'FG uncensored (P0→0)':<36}  {sl_fg_unc:>10.3f}   {r_fg_unc:.3f}")
print(f"  {'-'*52}")
print(f"  Censoring effect  (unc − cen) : {sl_fg_unc - sl_fg_cen:+.3f} %/K")
print(f"  Gap vs CMIP6  (cen − CMIP6)  : {sl_fg_cen - sl_cmip6:+.3f} %/K")
print(f"  Gap vs CMIP6  (unc − CMIP6)  : {sl_fg_unc - sl_cmip6:+.3f} %/K")
print("\n  MM mean rates (censored) :")
print(f"    sl_fw={np.mean(rates_cen[:,0]):+.3f}  sl_pw={np.mean(rates_cen[:,1]):+.3f}  "
      f"sl_pl={np.mean(rates_cen[:,2]):+.3f} %/K")
print("  MM mean rates (uncensored):")
print(f"    sl_fw={np.mean(rates_unc[:,0]):+.3f}  sl_pw={np.mean(rates_unc[:,1]):+.3f}  "
      f"sl_pl={np.mean(rates_unc[:,2]):+.3f} %/K")

fig, ax = plt.subplots(figsize=(13, 8))
ax.fill_between(X_ea, ea_p05*100, ea_p95*100, alpha=0.12, color="#1f78b4",
                label="5–95% CMIP6 spread")
ax.plot(X_ea, ea_mean*100, color="#1f78b4", lw=2.5, ls="--", zorder=5,
        label=f"CMIP6 MM  ({sl_cmip6:.2f} %/°C,  R={r_cmip6:.2f})")
ax.fill_between(deltaT_plot, fg_cen_p05, fg_cen_p95, alpha=0.12, color="#e31a1c")
ax.plot(deltaT_plot, fg_cen_mean, color="#e31a1c", lw=2.2, ls="-", zorder=6,
        label=f"Finite-Gamma censored  $P_0$=0.1  ({sl_fg_cen:.2f} %/°C,  R={r_fg_cen:.2f})")
ax.fill_between(deltaT_plot, fg_unc_p05, fg_unc_p95, alpha=0.12, color="#33a02c")
ax.plot(deltaT_plot, fg_unc_mean, color="#33a02c", lw=2.5, ls="-.", zorder=7,
        label=f"Finite-Gamma uncensored  $P_0\\to0$  ({sl_fg_unc:.2f} %/°C,  R={r_fg_unc:.2f})")
ax.axhline(0, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.axvline(0, color="gray", lw=0.7, ls=":", alpha=0.6)
ax.set_xlabel(r"$\Delta T$  ($^{\circ}$C)", fontsize=25)
ax.set_ylabel(r"$\delta A / A_0$  (%)", fontsize=25)
ax.set_title("CMIP6 Multimodel Mean  |  Censored vs Uncensored Finite-Gamma\n"
             r"Effect of $P_0 = 0.1$ mm day$^{-1}$ wet-day threshold",
             fontsize=25, fontweight="bold")
ax.legend(fontsize=20, loc="upper left", framealpha=0.93)
ax.grid(True, ls="--", alpha=0.35, lw=0.7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"MM_censored_vs_uncensored{FIG_SUFFIX}.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → MM_censored_vs_uncensored{FIG_SUFFIX}.png")

# ══════════════════════════════════════════════════════════════════════════
# 6c. Elasticity table — uncensored, Method B (u from gammainccinv)
# ══════════════════════════════════════════════════════════════════════════
alpha_rows_B = []
for model in MODELS:
    ref = ref_params_unc(model)
    if ref is None:
        continue
    PL_ref, fwet_ref, Pwet_ref = ref
    s_ref = Pwet_ref / PL_ref
    q_ref = np.clip(0.01 / fwet_ref, 1e-10, 1 - 1e-10)
    try:
        u_ref = u_from_quantile(s_ref, q_ref)
    except Exception as e:
        print(f"  ⚠  {model}: gammainccinv failed ({e}) — skipping")
        continue
    try:
        a1, a2, a3, _ = elasticities(s_ref, u_ref)
        alpha_rows_B.append({"Model": model, "s_ref": round(s_ref, 4), "u_ref": round(u_ref, 4),
                             "α₁": round(a1, 4), "α₂": round(a2, 4), "α₃": round(a3, 4)})
        print(f"  ✓  {model:<22}  s={s_ref:.4f}  u={u_ref:.4f}  "
              f"α₁={a1:.4f}  α₂={a2:.4f}  α₃={a3:.4f}")
    except Exception as e:
        print(f"  ⚠  {model}: alpha computation failed ({e}) — skipping")

df_B = pd.DataFrame(alpha_rows_B).set_index("Model")
mm_row_B = pd.DataFrame([{c: round(df_B[c].mean(), 4)
                          for c in ["s_ref", "u_ref", "α₁", "α₂", "α₃"]}], index=["MM Mean"])
df_B_full = pd.concat([df_B, mm_row_B])

print(f"\n{'='*75}")
print("  Elasticity coefficients per model  (uncensored, P₀→0)")
print(f"{'='*75}")
print(f"  {'Model':<22}  {'s_ref':>7}  {'u_ref':>7}  {'α₁':>7}  {'α₂':>9}  {'α₃':>9}")
print(f"  {'─'*65}")
for idx, row in df_B_full.iterrows():
    if idx == "MM Mean":
        print(f"  {'─'*65}")
    print(f"  {idx:<22}  {row['s_ref']:>7.4f}  {row['u_ref']:>7.4f}  "
          f"{row['α₁']:>7.4f}  {row['α₂']:>9.4f}  {row['α₃']:>9.4f}")
print(f"{'='*75}\n")
df_B_full.round(4).to_csv(os.path.join(FIG_DIR, f"alpha_table_uncensored{FIG_SUFFIX}.csv"))

# ══════════════════════════════════════════════════════════════════════════
# 8. Elasticities across multiple percentiles (uncensored)
#    u_ref(p): fwet_ref * gammaincc(s_ref, u_ref) = 1 - p/100
# ══════════════════════════════════════════════════════════════════════════
TARGET_PERCENTILES = [90, 92, 95, 97, 98, 99, 99.5, 99.8]
alpha_multi_rows = []

for model in MODELS:
    ref = ref_params_unc(model)
    if ref is None:
        continue
    PL_ref, fwet_ref, Pwet_ref = ref
    s_ref = Pwet_ref / PL_ref

    for p in TARGET_PERCENTILES:
        q_ref = np.clip((1.0 - p / 100.0) / fwet_ref, 1e-12, 1 - 1e-12)
        try:
            u_ref = u_from_quantile(s_ref, q_ref)
        except Exception as e:
            print(f"  ⚠  {model} P{p}: gammainccinv failed ({e}) — skipping")
            continue
        try:
            a1, a2, a3, _ = elasticities(s_ref, u_ref)
            alpha_multi_rows.append({"Model": model, "Percentile": p,
                                     "s_ref": s_ref, "u_ref": u_ref,
                                     "α₁": a1, "α₂": a2, "α₃": a3})
        except Exception as e:
            print(f"  ⚠  {model} P{p}: alpha computation failed ({e}) — skipping")

    print(f"  ✓  {model}  (s_ref={s_ref:.4f})")

df_multi = pd.DataFrame(alpha_multi_rows)
print(f"\nComputed elasticities for {df_multi['Model'].nunique()} models "
      f"across {len(TARGET_PERCENTILES)} percentiles.")

p05 = lambda x: np.nanpercentile(x, 5)
p95 = lambda x: np.nanpercentile(x, 95)
p05.__name__ = "p05"
p95.__name__ = "p95"

mm_stats = (df_multi.groupby("Percentile")[["α₁", "α₂", "α₃"]]
            .agg(["mean", "std", "min", "max", p05, p95]))
mm_stats.columns = ["_".join(c) for c in mm_stats.columns]
mm_stats = mm_stats.reset_index().sort_values("Percentile")
mm_stats["α₂+α₃_mean"] = mm_stats["α₂_mean"] + mm_stats["α₃_mean"]

print(f"\n{'='*78}")
print("  Multimodel-mean elasticity coefficients vs. percentile threshold")
print(f"{'='*78}")
print(f"  {'Percentile':>10}  {'α₁ (mean)':>10}  {'α₂ (mean)':>10}  {'α₃ (mean)':>10}  {'α₂+α₃':>8}")
print(f"  {'-'*58}")
for _, row in mm_stats.iterrows():
    print(f"  P{row['Percentile']:>8.1f}  {row['α₁_mean']:>10.4f}  {row['α₂_mean']:>10.4f}  "
          f"{row['α₃_mean']:>10.4f}  {row['α₂+α₃_mean']:>8.4f}")
print(f"{'='*78}")
mm_stats.to_csv(os.path.join(FIG_DIR, f"MM_alpha_vs_percentile{FIG_SUFFIX}.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
pct = mm_stats["Percentile"].values
ax.plot(pct, mm_stats["α₁_mean"], "^--", color="#2ca02c", lw=2.0, ms=7,
        label=r"$\alpha_1$ — MM mean")
ax.plot(pct, mm_stats["α₂_mean"], "o-", color="#1f77b4", lw=2.5, ms=8,
        label=r"$\alpha_2$ — MM mean")
ax.fill_between(pct, mm_stats["α₂_p05"], mm_stats["α₂_p95"],
                color="#1f77b4", alpha=0.15, label="5–95% model spread")
ax.plot(pct, mm_stats["α₃_mean"], "s-", color="#d62728", lw=2.5, ms=8,
        label=r"$\alpha_3$ — MM mean")
ax.fill_between(pct, mm_stats["α₃_p05"], mm_stats["α₃_p95"], color="#d62728", alpha=0.15)
ax.set_xticks(pct)
ax.set_xticklabels([f"P{x:g}" for x in pct], rotation=90, fontsize=15)
ax.tick_params(axis="y", labelsize=20)
ax.set_xlabel("Percentile threshold")
ax.set_ylabel(r"Elasticity  $\partial \ln A / \partial \ln(\cdot)$")
ax.set_title(r"CMIP6 multimodel-mean of $\alpha_1$, $\alpha_2$ and $\alpha_3$", fontsize=14)
ax.legend(fontsize=11, loc="upper left", frameon=False)
ax.grid(True, ls=":", alpha=0.4)
ax.set_xlim(min(pct) - 1, max(pct) + 0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"MM_alpha123_vs_percentile{FIG_SUFFIX}.png"),
            dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, f"MM_alpha123_vs_percentile{FIG_SUFFIX}.pdf"),
            bbox_inches="tight")
plt.close(fig)
print(f"Saved → MM_alpha123_vs_percentile{FIG_SUFFIX}.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════
# 9. Analytic rarity-dependence of α2, α3 (mpmath), shaded by model s_ref
# ══════════════════════════════════════════════════════════════════════════
mp.mp.dps = 30

def upper_gamma(s, u):
    return mp.gammainc(s, u, mp.inf)

def T_su_mp(s, u):
    return mp.diff(lambda ss: upper_gamma(ss, u), s)

def alpha2_alpha3(s, u):
    G = upper_gamma(s, u)
    T = T_su_mp(s, u)
    bracket = T / G - mp.digamma(s)
    a2 = s * bracket
    a3 = -s * bracket + (u**s * mp.e**(-u)) / G
    return float(a2), float(a3)

def alpha_curve(s_val, percentiles):
    s_mp   = mp.mpf(s_val)
    u_vals = gamma_dist.ppf(percentiles / 100.0, a=float(s_val), scale=1.0)
    a2_list, a3_list = [], []
    for u in u_vals:
        a2, a3 = alpha2_alpha3(s_mp, mp.mpf(u))
        a2_list.append(a2); a3_list.append(a3)
    return np.array(a2_list), np.array(a3_list)

# Mean and 5–95% spread of s_ref from the same source (df_multi)
s_ref_per_model = df_multi.drop_duplicates("Model")["s_ref"].values
s_ref_mm = np.nanmean(s_ref_per_model)
s_p05    = np.nanpercentile(s_ref_per_model, 5)
s_p95    = np.nanpercentile(s_ref_per_model, 95)
print(f"s_ref (mean) = {s_ref_mm:.4f}   [P05={s_p05:.4f}, P95={s_p95:.4f}]  "
      f"n_models={len(s_ref_per_model)}")

ref_percentiles = np.array([90, 92, 94, 96, 98, 99, 99.5, 99.8])
ref_a2, ref_a3  = alpha_curve(s_ref_mm, ref_percentiles)

X_MIN = 89
percentiles = np.concatenate([np.linspace(X_MIN, 99, 60),
                              100 - np.geomspace(1.0, 0.02, 60)])
percentiles = np.unique(np.round(percentiles, 4))

alpha2_vals, alpha3_vals = alpha_curve(s_ref_mm, percentiles)
alpha2_p05,  alpha3_p05  = alpha_curve(s_p05, percentiles)
alpha2_p95,  alpha3_p95  = alpha_curve(s_p95, percentiles)
alpha2_lo, alpha2_hi = np.minimum(alpha2_p05, alpha2_p95), np.maximum(alpha2_p05, alpha2_p95)
alpha3_lo, alpha3_hi = np.minimum(alpha3_p05, alpha3_p95), np.maximum(alpha3_p05, alpha3_p95)

plt.rcParams.update({"font.size": 14, "font.family": "serif", "mathtext.fontset": "cm"})

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.plot(percentiles, alpha2_vals, color="#1f77b4", lw=2.5,
        label=r"$\alpha_2$ (thermodynamic, $P_L$)")
ax.fill_between(percentiles, alpha2_lo, alpha2_hi, color="#1f77b4", alpha=0.15,
                label="5–95% spread (model $s_{\\mathrm{ref}}$)")
ax.plot(percentiles, alpha3_vals, color="#d62728", lw=2.5,
        label=r"$\alpha_3$ (dynamical, shape)")
ax.fill_between(percentiles, alpha3_lo, alpha3_hi, color="#d62728", alpha=0.15)
ax.scatter(ref_percentiles, ref_a2, color="#1f77b4", s=55, zorder=5, edgecolor="k", lw=0.6)
ax.scatter(ref_percentiles, ref_a3, color="#d62728", s=55, zorder=5, edgecolor="k", lw=0.6)
ax.set_xticks(ref_percentiles)
ax.set_xticklabels([f"P{p:g}" for p in ref_percentiles], rotation=90, fontsize=14)
ax.set_xlabel("Percentile threshold defining the extreme event")
ax.set_ylabel(r"Elasticity  $\partial \ln A / \partial \ln(\cdot)$")
ax.set_title(r"Analytic rarity-dependence of $\alpha_2$ and $\alpha_3$" + "\n", fontsize=13)
ax.legend(fontsize=11, loc="upper left", frameon=False)
ax.grid(True, ls=":", alpha=0.4)
ax.set_xlim(X_MIN, 100)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "alpha2_alpha3_vs_percentile_shaded.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "alpha2_alpha3_vs_percentile_shaded.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved → alpha2_alpha3_vs_percentile_shaded.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════
# 10. Significance: censored vs uncensored FG curves (paired by model)
# ══════════════════════════════════════════════════════════════════════════
WARMING_LEVELS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
N_BOOT         = 10000
DROP_DUPLICATE = False          # CNRM_CM61 already excluded from MODELS
rng = np.random.default_rng(42)

model_names = np.array(list(shift_log.keys()))
keep = np.ones(len(model_names), bool)
if DROP_DUPLICATE:
    keep &= model_names != "CNRM_CM61"
cen  = fg_curves_cen[keep]
unc  = fg_curves_unc[keep]
n_fg = keep.sum()

def boot_ci(x, n=N_BOOT):
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])

rows = []
for wl in WARMING_LEVELS:
    i = int(np.argmin(np.abs(deltaT_plot - wl)))
    c, u = cen[:, i], unc[:, i]
    d = u - c
    _, p_t = ttest_rel(u, c)
    _, p_w = wilcoxon(d)
    lo, hi = boot_ci(d)
    rows.append({"ΔT (°C)": wl, "cen mean (%)": c.mean(), "unc mean (%)": u.mean(),
                 "Δ unc−cen (%)": d.mean(), "95% CI lo": lo, "95% CI hi": hi,
                 "Cohen dz": d.mean() / d.std(ddof=1),
                 "models unc<cen": f"{(d < 0).sum()}/{n_fg}",
                 "p paired-t": p_t, "p Wilcoxon": p_w})

df_sig = pd.DataFrame(rows).set_index("ΔT (°C)")
pd.set_option("display.float_format", lambda x: f"{x:.3g}")
pd.set_option("display.width", 200)
print(f"n models = {n_fg}")
print(df_sig.to_string())

sc, su = sl_c_all[keep], sl_u_all[keep]
dsl = su - sc
_, p_t = ttest_rel(su, sc)
_, p_w = wilcoxon(dsl)
lo, hi = boot_ci(dsl)
print(f"\nSlope  cen = {sc.mean():.3f}  unc = {su.mean():.3f} %/°C")
print(f"Slope  unc − cen = {dsl.mean():+.3f} %/°C  [95% CI {lo:+.3f}, {hi:+.3f}]")
print(f"  paired t p = {p_t:.2e}   Wilcoxon p = {p_w:.2e}   "
      f"models with unc<cen: {(dsl < 0).sum()}/{n_fg}")

# Mean paired difference vs ΔT with 95% bootstrap CI
D = unc - cen
idx = rng.integers(0, n_fg, size=(N_BOOT, n_fg))
boot_means = D[idx].mean(axis=1)
ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(11, 6))
ax.fill_between(deltaT_plot, ci_lo, ci_hi, color="#33a02c", alpha=0.25,
                label="95% bootstrap CI (paired)")
ax.plot(deltaT_plot, D.mean(axis=0), color="#33a02c", lw=2.5,
        label=r"Mean $\delta A/A_0$ (uncensored − censored)")
for row in D:
    ax.plot(deltaT_plot, row, color="gray", lw=0.6, alpha=0.4)
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel(r"$\Delta T$  ($^{\circ}$C)", fontsize=22)
ax.set_ylabel(r"$\Delta(\delta A/A_0)$  (%)", fontsize=22)
ax.set_title("Censoring effect: paired difference across models", fontsize=22)
ax.legend(fontsize=16, loc="lower left", frameon=False)
ax.grid(True, ls="--", alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"MM_censoring_difference_CI{FIG_SUFFIX}.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → MM_censoring_difference_CI{FIG_SUFFIX}.png")

# t-based 95% CI of the mean paired slope difference
ds_all = sl_u_all - sl_c_all
n      = len(ds_all)
se     = ds_all.std(ddof=1) / np.sqrt(n)
tcrit  = t_dist.ppf(0.975, df=n - 1)
print(f"Δslope = {ds_all.mean():+.3f} %/°C  "
      f"[95% CI {ds_all.mean() - tcrit*se:+.3f}, {ds_all.mean() + tcrit*se:+.3f}]  "
      f"p = {ttest_rel(sl_u_all, sl_c_all).pvalue:.2e}  n = {n}")
