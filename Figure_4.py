#!/usr/bin/env python
"""
Figure 4 — decomposition of the extreme-area sensitivity and threshold dependence

  a — waterfall: α1·δf_wet, α2·δP̄_wet, α3·δP_L contributions, linear total, FG (homogeneous, P99)
  b — sensitivity vs percentile threshold: CMIP6 empirical (dot + 5–95%) and FG

Method 2 run (pipeline_method2.py): global mean first, full P_L variance,
fixed 1980–2009 P_L baseline; outputs in *_method2 folders.
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy.special   import gammaincc, gammainccinv, gamma as Gamma_func, digamma
from scipy.integrate import quad
from scipy.stats     import linregress
from scipy.stats     import gamma as gamma_dist

# ══════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════
RUN_LABEL = "method2"
BASE_DIR  = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP"
SUFFIX    = "_method2"

MM_DIR   = os.path.join(BASE_DIR, "multimodel_mean" + SUFFIX)
GEN_DIR  = os.path.join(BASE_DIR, "general" + SUFFIX)
FIG_DIR  = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/MULTIMODEL_MEAN"
FIG_NAME = f"Fig4_combined_decomposition_threshold_riskratio_{RUN_LABEL}"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_START = 1980
BASELINE_END   = 2009

# Same ΔT grid and fit range as the analytical notebook (−1 to 7 °C)
deltaT_plot = np.arange(-1.0, 7.10, 0.1)
I0          = int(np.argmin(np.abs(deltaT_plot)))     # ΔT ≈ 0 reference

EMP_MT_DIR = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2/multithresh"
P99_DIR    = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2"
CSV_OUT    = os.path.join(BASE_DIR, "fig4b_inputs.csv")

sys.path.insert(0, BASE_DIR)
import config as cfg
EXCLUDE = ["CNRM_CM61"]     # duplicate of CNRM_CM6

def f_mt(m):  return os.path.join(EMP_MT_DIR, f"tas_vs_extreme_precip_multithresh_{m}.nc")
def f_gen(m): return os.path.join(GEN_DIR, m, f"{m}_fwet_annual.nc")

REQUIRED = [("Method 2 pipeline outputs", f_gen), ("CMIP6 empirical multi-threshold", f_mt)]
MODELS = [m for m in cfg.MODELS
          if m not in EXCLUDE and all(os.path.exists(p(m)) for _, p in REQUIRED)]

print(f"Run                 : {RUN_LABEL} (Method 2)")
print(f"Reading from        : {GEN_DIR}")
print(f"Empirical (panel b) : {EMP_MT_DIR}")
print(f"Models (panels a, b): {len(MODELS)}  (excluded: {', '.join(EXCLUDE) or 'none'})")
for name, p in REQUIRED:
    miss = [m for m in cfg.MODELS if m not in EXCLUDE and not os.path.exists(p(m))]
    if miss:
        print(f"  missing {name}: {', '.join(miss)}  -> left out of panels a and b")

# ── Nature print-spec settings ───────────────────────────────────────────────
MM_TO_IN    = 1 / 25.4
FIG_SIZE_IN = (180 * MM_TO_IN, 170 * MM_TO_IN)
DPI = 300
FS  = 8

plt.rcParams.update({
    "figure.dpi":        DPI,
    "savefig.dpi":       DPI,
    "font.size":         FS,
    "axes.titlesize":    FS,
    "axes.labelsize":    FS,
    "xtick.labelsize":   FS - 1,
    "ytick.labelsize":   FS - 1,
    "legend.fontsize":   FS - 1,
    "text.usetex":       False,
    "mathtext.fontset":  "dejavusans",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth":   1.0,
})


def panel_label(ax, letter, pad=4):
    """Bold lower-case panel label, top-left — Nature convention."""
    ax.set_title(rf"$\mathbf{{{letter}}}$", fontsize=FS + 1, loc="left", pad=pad)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
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


def get_rates_cen(model, gen_dir):
    """Censored per-°C rates (global mean first): fwet, Pwet annual OLS; PL 31-yr OLS with intercept."""
    gen = os.path.join(gen_dir, model)

    ds   = xr.open_dataset(os.path.join(gen, f"{model}_fwet_annual.nc"))
    mask = np.isfinite(ds["dT_annual"].values) & np.isfinite(ds["fwet_rel_global"].values)
    sl_fw, *_ = linregress(ds["dT_annual"].values[mask], ds["fwet_rel_global"].values[mask])
    ds.close()

    ds   = xr.open_dataset(os.path.join(gen, f"{model}_Pwet_annual.nc"))
    mask = np.isfinite(ds["dT_annual"].values) & np.isfinite(ds["Pwet_rel_global"].values)
    sl_pw, *_ = linregress(ds["dT_annual"].values[mask], ds["Pwet_rel_global"].values[mask])
    ds.close()

    ds_pl   = xr.open_dataset(os.path.join(gen, f"{model}_PL_31yr.nc"))
    ds_taup = xr.open_dataset(os.path.join(gen, f"{model}_TauP_31yr.nc"))
    pl, dT  = xr.align(ds_pl["PL_rel"], ds_taup["dT_31yr"], join="inner")
    mask    = np.isfinite(pl.values) & np.isfinite(dT.values)
    sl_pl, *_ = linregress(dT.values[mask], pl.values[mask])
    ds_pl.close(); ds_taup.close()

    return sl_fw, sl_pw, sl_pl


def homog_ref(model):
    """Global-mean (cos-lat weighted) baseline parameters of one model."""
    gen = os.path.join(GEN_DIR, model)
    fields = {}
    for key, fname, cands in [("PL",   "PL_ref_1980_2009",   ["PL_mean", "PL"]),
                              ("fwet", "fwet_ref_1980_2009", ["fwet_mean", "fwet"]),
                              ("Pwet", "Pwet_ref_1980_2009", ["Pwet_mean", "Pwet"])]:
        arr, lat, lon = load_grid(os.path.join(gen, f"{model}_{fname}.nc"), cands)
        if arr is None:
            return None
        fields[key] = arr
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, fields["PL"].shape[1]))

    def wm(x):
        ok = np.isfinite(x)
        return float((x[ok] * w[ok]).sum() / w[ok].sum())

    return {k: wm(v) for k, v in fields.items()}


def homog_dA(ref, rates, exceed=0.01, scale=("pl", "pw", "fw")):
    """δA/A0 (%) on deltaT_plot for the representative distribution (default M3)."""
    PL0, Pw0, fw0 = ref["PL"], ref["Pwet"], ref["fwet"]
    a0  = max(Pw0 / PL0, 1e-6)
    thr = float(gamma_dist.isf(exceed / fw0, a0, scale=PL0)) if fw0 > exceed else 0.0

    A = []
    for dT in deltaT_plot:
        PL = PL0 * (1 + rates["pl"] * dT) if "pl" in scale else PL0
        Pw = Pw0 * (1 + rates["pw"] * dT) if "pw" in scale else Pw0
        fw = fw0 * (1 + rates["fw"] * dT) if "fw" in scale else fw0
        a  = max(Pw / PL, 1e-6) if PL > 0 else 1e-6
        A.append(fw * gamma_dist.sf(thr, a, scale=PL) if thr > 0 else fw)
    A = np.nan_to_num(np.array(A, dtype=float))
    return (A - A[I0]) / A[I0] * 100


def slope(y):
    """OLS slope (%/°C) of a δA/A0 curve over the full ΔT grid."""
    return linregress(deltaT_plot, y).slope


# ══════════════════════════════════════════════════════════════════════════
# Check: multi-model mean scaling rates
# ══════════════════════════════════════════════════════════════════════════
_rates = []
for _m in MODELS:
    try:
        _rates.append(get_rates_cen(_m, GEN_DIR))
    except Exception as e:
        print(f"  skip {_m}: {e}")
_rates = np.array(_rates) * 100
print(f"Multi-model mean slopes ({len(_rates)} models, run: {RUN_LABEL})")
for _name, _col in zip(["f_wet", "P_wet", "P_L"], _rates.T):
    print(f"  {_name:<6}: {np.mean(_col):+.2f} %/°C")

# ══════════════════════════════════════════════════════════════════════════
# Homogeneous FG curves at P99 (panel a "FG" bar)
# ══════════════════════════════════════════════════════════════════════════
fg_curves_cen, fg_rates_log, shift_log, homog_refs = [], {}, {}, {}

for model in MODELS:
    ref = homog_ref(model)
    if ref is None:
        continue
    try:
        sl_fw_c, sl_pw_c, sl_pl_c = get_rates_cen(model, GEN_DIR)
    except Exception:
        continue
    rates = {"pl": sl_pl_c, "pw": sl_pw_c, "fw": sl_fw_c}

    fg_rates_log[model] = dict(sl_fw_c=sl_fw_c * 100, sl_pw_c=sl_pw_c * 100, sl_pl_c=sl_pl_c * 100)
    homog_refs[model] = ref

    curve = homog_dA(ref, rates, exceed=0.01)
    shift_log[model] = dict(sl_raw_cen=slope(curve))
    fg_curves_cen.append(curve)

fg_curves_cen = np.array(fg_curves_cen)
fg_slopes_per_model = np.array([shift_log[m]["sl_raw_cen"] for m in shift_log])
print(f"Homogeneous FG at P99: {len(fg_slopes_per_model)} models, "
      f"multi-model mean slope = {np.mean(fg_slopes_per_model):.2f} %/°C")

# ══════════════════════════════════════════════════════════════════════════
# Censored elasticities α1, α2, α3
# ══════════════════════════════════════════════════════════════════════════
alpha_rows_cen = []
for model in MODELS:
    gen = os.path.join(GEN_DIR, model)
    PL_c,   lat, _ = load_grid(os.path.join(gen, f"{model}_PL_ref_1980_2009.nc"),   ["PL_mean", "PL"])
    fwet_c, _,   _ = load_grid(os.path.join(gen, f"{model}_fwet_ref_1980_2009.nc"), ["fwet_mean", "fwet"])
    Pwet_c, _,   _ = load_grid(os.path.join(gen, f"{model}_Pwet_ref_1980_2009.nc"), ["Pwet_mean", "Pwet"])
    if any(v is None for v in [PL_c, fwet_c, Pwet_c]):
        continue

    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, PL_c.shape[1]))

    def wm(arr):
        valid = np.isfinite(arr)
        return float(np.nansum(arr * w * valid) / np.nansum(w * valid))

    PL_ref, fwet_ref, Pwet_ref = wm(PL_c), wm(fwet_c), wm(Pwet_c)
    if PL_ref <= 0 or Pwet_ref <= 0 or fwet_ref <= 0:
        continue

    s_ref = Pwet_ref / PL_ref
    q_ref = np.clip(0.01 / fwet_ref, 1e-10, 1 - 1e-10)
    try:
        u_ref = float(gammainccinv(np.array([s_ref]), np.array([q_ref]))[0])
    except Exception:
        continue

    try:
        G_su    = Gamma_func(s_ref) * float(gammaincc(np.array([s_ref]), np.array([u_ref]))[0])
        T_su, _ = quad(lambda t: t**(s_ref - 1) * np.log(t) * np.exp(-t),
                       u_ref, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10)
        psi_s   = float(digamma(s_ref))
        alpha1  = 1.0
        alpha2  = s_ref * (T_su / G_su - psi_s)
        alpha3  = s_ref * (psi_s - T_su / G_su) + u_ref**s_ref * np.exp(-u_ref) / G_su
        alpha_rows_cen.append({"Model": model, "α₁": round(alpha1, 4),
                               "α₂": round(alpha2, 4), "α₃": round(alpha3, 4)})
    except Exception:
        pass

df_cen = pd.DataFrame(alpha_rows_cen).set_index("Model")
mm_row_cen = pd.DataFrame([{c: round(df_cen[c].mean(), 4) for c in ["α₁", "α₂", "α₃"]}],
                          index=["MM Mean"])
df_cen_full = pd.concat([df_cen, mm_row_cen])

# ══════════════════════════════════════════════════════════════════════════
# Contribution terms (panel a)
# ══════════════════════════════════════════════════════════════════════════
C_FW, C_PW, C_PL, C_TOT = ("α₁·δf_wet/f_wet (%/K)", "α₂·δP̄_wet/P̄_wet (%/K)",
                           "α₃·δP_L/P_L (%/K)", "Linear total (%/K)")
contrib_rows = []
for model in MODELS:
    if model not in df_cen_full.index or model not in fg_rates_log:
        continue
    row, rates = df_cen_full.loc[model], fg_rates_log[model]
    term_fwet = float(row["α₁"]) * rates["sl_fw_c"]
    term_pwet = float(row["α₂"]) * rates["sl_pw_c"]
    term_pl   = float(row["α₃"]) * rates["sl_pl_c"]
    contrib_rows.append({"Model": model, C_FW: term_fwet, C_PW: term_pwet,
                         C_PL: term_pl, C_TOT: term_fwet + term_pwet + term_pl})

df_contrib = pd.DataFrame(contrib_rows).set_index("Model")
mm_contrib = pd.DataFrame([{c: df_contrib[c].mean() for c in [C_FW, C_PW, C_PL, C_TOT]}], index=["MM Mean"])
sd_contrib = pd.DataFrame([{c: df_contrib[c].std()  for c in [C_FW, C_PW, C_PL, C_TOT]}], index=["MM Std"])
df_contrib_full = pd.concat([df_contrib, mm_contrib, sd_contrib])

sl_fg_cen_check = float(np.mean(fg_slopes_per_model))
print(f"Panel a FG (homogeneous, P99) = {sl_fg_cen_check:.2f} %/°C")

# ══════════════════════════════════════════════════════════════════════════
# FG sensitivity at each percentile threshold (panel b)
# ══════════════════════════════════════════════════════════════════════════
SLUGS       = ["p95", "p98", "p99", "p995", "p998"]
LABELS      = ["P95", "P98", "P99", "P99.5", "P99.8"]
COLORS      = ["#4575b4", "#74add1", "#f46d43", "#88d727", "#00a592"]
PERCENTILES = [95.0, 98.0, 99.0, 99.5, 99.8]


def _rates_frac(model):
    r = fg_rates_log[model]
    return {"pl": r["sl_pl_c"] / 100, "pw": r["sl_pw_c"] / 100, "fw": r["sl_fw_c"] / 100}


fg_multithresh_rows = []
for model, ref in homog_refs.items():
    rates = _rates_frac(model)
    for p, label in zip(PERCENTILES, LABELS):
        curve = homog_dA(ref, rates, exceed=1.0 - p / 100.0)
        fg_multithresh_rows.append({"Model": model, "Label": label, "fg_slope": slope(curve)})

df_fg_multi = pd.DataFrame(fg_multithresh_rows)
fg_stats = (df_fg_multi.groupby("Label")["fg_slope"]
            .agg(mean="mean",
                 p05=lambda x: np.nanpercentile(x, 5),
                 p95=lambda x: np.nanpercentile(x, 95))
            .reindex(LABELS))
fg_mm, fg_p05, fg_p95 = fg_stats["mean"].tolist(), fg_stats["p05"].tolist(), fg_stats["p95"].tolist()

print(f"{'Threshold':<10}  {'FG mean (%/°C)':>16}  {'[P05, P95]':>18}")
for label, mm, p05, p95 in zip(LABELS, fg_mm, fg_p05, fg_p95):
    print(f"  {label:<8}  {mm:>+16.2f}  [{p05:+.2f}, {p95:+.2f}]")

# Fine percentile grid, P95 -> P99.8
PERCENTILES_FINE = list(np.round(np.arange(95.0, 99.5 + 1e-9, 0.5), 2))
if PERCENTILES_FINE[-1] != 99.8:
    PERCENTILES_FINE.append(99.8)

fg_fine_rows = []
for model, ref in homog_refs.items():
    rates = _rates_frac(model)
    for p in PERCENTILES_FINE:
        curve = homog_dA(ref, rates, exceed=1.0 - p / 100.0)
        fg_fine_rows.append({"Model": model, "Percentile": p, "fg_slope": slope(curve)})

df_fg_fine = pd.DataFrame(fg_fine_rows)
fg_fine_stats = (df_fg_fine.groupby("Percentile")["fg_slope"].mean()
                 .reset_index().sort_values("Percentile"))
fg_fine_pcts = fg_fine_stats["Percentile"].tolist()
fg_fine_mm   = fg_fine_stats["fg_slope"].tolist()
x_fg_fine    = np.interp(fg_fine_pcts, PERCENTILES, [0, 1, 2, 3, 4])

print(f"{'Percentile':<10}  {'FG mean (%/°C)':>14}")
for p, v in zip(fg_fine_pcts, fg_fine_mm):
    print(f"  P{p:<8g}  {v:>+14.2f}")

# ══════════════════════════════════════════════════════════════════════════
# CMIP6 empirical sensitivity at each threshold (panel b)
# ══════════════════════════════════════════════════════════════════════════
COMMON_X = np.linspace(-1.0, 7.0, 300)
all_dT = []
all_dA = {slug: [] for slug in SLUGS}
for m in MODELS:
    ds = xr.open_dataset(f_mt(m))
    all_dT.append(ds["tas_anomaly"].values)
    for slug in SLUGS:
        all_dA[slug].append(ds[f"dA_rel_{slug}"].values)
    ds.close()


def mm_stats_emp(dT_list, dA_list, common_x=COMMON_X):
    fits, slopes = [], []
    for dT, dA in zip(dT_list, dA_list):
        valid = np.isfinite(dT) & np.isfinite(dA)
        if valid.sum() < 5:
            continue
        sl, ic, *_ = linregress(dT[valid], dA[valid])
        fits.append(sl * common_x + ic)
        slopes.append(sl)
    mean = np.nanmean(np.array(fits), axis=0)
    sl_mm, *_ = linregress(common_x, mean)
    return sl_mm, np.array(slopes)


emp_stats = {}
for slug in SLUGS:
    sl_mm, slopes_ = mm_stats_emp(all_dT, all_dA[slug])
    emp_stats[slug] = dict(sl=sl_mm, slopes=slopes_)

emp_mm  = [emp_stats[s]["sl"] for s in SLUGS]
emp_p05 = [np.nanpercentile(emp_stats[s]["slopes"],  5) for s in SLUGS]
emp_p95 = [np.nanpercentile(emp_stats[s]["slopes"], 95) for s in SLUGS]

print(f"CMIP6 empirical (panel b): {len(all_dT)} models")
print(f"{'Threshold':<10}  {'Empirical mean (%/°C)':>22}  {'[P05, P95]':>18}")
for label, mm, p05, p95 in zip(LABELS, emp_mm, emp_p05, emp_p95):
    print(f"  {label:<8}  {mm:>+22.2f}  [{p05:+.2f}, {p95:+.2f}]")

# Consistency check: multi-threshold P99 vs main P99 empirical files
max_diff = 0.0
for m in MODELS:
    a = xr.open_dataset(os.path.join(P99_DIR, f"tas_vs_extreme_precip_{m}.nc"))
    b = xr.open_dataset(f_mt(m))
    fa, fb = xr.align(a["fraction_extreme_area"], b["fraction_p99"], join="inner")
    d = float(np.nanmax(np.abs(fa.values - fb.values)))
    max_diff = max(max_diff, d)
    if d > 1e-10:
        print(f"  ⚠  {m}: max difference = {d:.2e}")
    a.close(); b.close()
print(f"P99 check over {len(MODELS)} models: max |difference| = {max_diff:.2e} "
      f"({'identical' if max_diff < 1e-10 else 'DIFFERENT — check inputs'})")

# ══════════════════════════════════════════════════════════════════════════
# Figure: panels a and b side by side
# ══════════════════════════════════════════════════════════════════════════
FIG_SIZE_IN_2R = (183 * MM_TO_IN, FIG_SIZE_IN[1])
fig = plt.figure(figsize=FIG_SIZE_IN_2R)
gs_main = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.95], hspace=0.45, figure=fig)
gs_top  = gs_main[0].subgridspec(1, 2, wspace=0.35)
ax_a = fig.add_subplot(gs_top[0, 0])
ax_b = fig.add_subplot(gs_top[0, 1])

# ── Panel a — waterfall decomposition ───────────────────────────────────────
ax = ax_a
sd_fg_cen = np.std(fg_slopes_per_model)

mm_c_fw, mm_c_pw, mm_c_pl, mm_tot = (mm_contrib.loc["MM Mean", c] for c in [C_FW, C_PW, C_PL, C_TOT])
sd_c_fw, sd_c_pw, sd_c_pl, sd_tot = (sd_contrib.loc["MM Std",  c] for c in [C_FW, C_PW, C_PL, C_TOT])

labels = [
    r"$\alpha_1\!\cdot\!\delta f_\mathrm{wet}/f_\mathrm{wet}$",
    r"$\alpha_2\!\cdot\!\delta \bar{P}_\mathrm{wet}/\bar{P}_\mathrm{wet}$",
    r"$\alpha_3\!\cdot\!\delta P_L/P_L$",
    "Linear total",
    "FG",
]
bar_vals = [mm_c_fw, mm_c_pw, mm_c_pl, mm_tot, sl_fg_cen_check]
errors   = [sd_c_fw, sd_c_pw, sd_c_pl, sd_tot, sd_fg_cen]
colors   = ["#8C4CAF", "#FF9800", "#2196F3", "#65B027", "#BC35E5"]
bottoms  = [0, mm_c_fw, mm_c_fw + mm_c_pw, 0, 0]
x = np.arange(len(labels))

for i, (val, bot, err, col) in enumerate(zip(bar_vals, bottoms, errors, colors)):
    ax.bar(x[i], val, bottom=bot, color=col, alpha=0.85, width=0.62,
           edgecolor="black", linewidth=0.5, zorder=3)
    top    = bot + val
    offset = err + 0.2 if err > 0 else 0.2
    if err > 0:
        ax.errorbar(x[i], top, yerr=err, fmt="none", color="black",
                    capsize=2, capthick=0.6, lw=0.6, zorder=5)
    ax.text(x[i], top + offset, f"{val:.1f}", ha="center", va="bottom",
            fontsize=FS - 1, fontweight="bold", color=col, zorder=10)

cumsum = 0
for i in range(2):
    cumsum += bar_vals[i]
    ax.plot([x[i] + 0.31, x[i + 1] - 0.31], [cumsum, cumsum], color="gray", ls="--", lw=0.5, zorder=2)

ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.axhline(7, color="blue", ls="-.", lw=0.3, alpha=0.7, label="CC (7%/°C)", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FS - 1, rotation=30, ha="right")
ax.set_ylabel(r"$\delta A/A_0$ (%/°C)", fontsize=FS)
ymax = max(b + v for b, v in zip(bottoms, bar_vals)) + max(errors) + 2.0
ax.set_ylim(-2.5, ymax)
ax.yaxis.set_major_locator(mticker.MultipleLocator(3))
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
ax.tick_params(axis="both", labelsize=FS - 1)
ax.grid(True, axis="y", ls="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=FS - 1)
panel_label(ax, "a")

# ── Panel b — threshold-sensitivity dot-and-whisker + FG overlay ────────────
ax = ax_b
x = np.arange(len(LABELS))
ax.vlines(x, emp_p05, emp_p95, color="black", linewidth=0.6, zorder=3)
ax.scatter(x, emp_mm, s=26, color=COLORS, edgecolor="black", linewidth=0.5,
           zorder=4, label="CMIP6 empirical")
ax.plot(x_fg_fine, fg_fine_mm, color="#BC35E5", linewidth=0.9, zorder=2, label="FG curve")
ax.scatter(x, fg_mm, marker="D", facecolor="white", edgecolor="#BC35E5",
           s=26, linewidth=1.0, zorder=5, label="FG")

label_offset = 1.5
for xi, hi_pos, fg in zip(x, emp_p95, fg_mm):
    ax.text(xi, hi_pos + label_offset, f"{fg:.1f}", ha="center", va="bottom",
            fontsize=FS - 1, fontweight="bold", color="#BC35E5")

ax.axhline(7.0, color="blue", ls="-.", lw=0.3, label="CC(7%/°C)", zorder=1)
ax.axhline(0,   color="black", lw=0.5, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.set_xlim(-0.5, len(LABELS) - 0.5)
ax.set_xlabel("Percentile threshold", fontsize=FS)
ax.set_ylabel(r"$\delta A/A_0$ (%/°C)", fontsize=FS)
ax.legend(loc="upper left", fontsize=FS - 2, frameon=False)
ax.grid(True, axis="y", ls="--", alpha=0.2)
ax.set_axisbelow(True)
ax.tick_params(axis="both", which="major", labelsize=FS - 1)
ax.set_ylim(top=40)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
panel_label(ax, "b")

plt.savefig(os.path.join(FIG_DIR, f"{FIG_NAME}.png"), dpi=DPI, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, f"{FIG_NAME}.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved → {os.path.join(FIG_DIR, FIG_NAME)}.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════
# Panel b inputs to CSV
# ══════════════════════════════════════════════════════════════════════════
_d = pd.DataFrame({"kind": "discrete", "label": LABELS, "pct": PERCENTILES,
                   "fg": fg_mm, "emp_mm": emp_mm, "emp_p05": emp_p05, "emp_p95": emp_p95})
_f = pd.DataFrame({"kind": "fine", "label": "", "pct": fg_fine_pcts, "fg": fg_fine_mm})
pd.concat([_d, _f]).to_csv(CSV_OUT, index=False)
print(f"Saved → {CSV_OUT}")
