#!/usr/bin/env python
"""
Figure 3 — Finite-Gamma decomposition and CMIP6 scaling of the extreme-precipitation area

  a — Wet-day precipitation PDF (GFDL-ESM4): historical vs. two future scenarios
  b — Relative change in P_L, P̄_wet, f_wet vs ΔT (CMIP6 multimodel mean, 5–95%)
  c — δA/A0 vs ΔT: CMIP6 MM vs homogeneous M1–M3 and SpEx (M5)

Method 2 run (pipeline_method2.py): global mean first, full P_L variance,
fixed 1980–2009 P_L baseline; outputs in *_method2 folders.
"""

import os
import sys
import subprocess

import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ── Data paths ───────────────────────────────────────────────────────────────
GFDL_DATA_DIR = "/nfs_scratch/dmuriar/CIMP6/GFDL_ESM/Precip"
GFDL_TAS_FILE = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2/tas_vs_extreme_precip_GFDL.nc"

RUN_LABEL = "method2"
SUFFIX    = "_method2"
BASE_DIR  = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP"

PL_TAUP_MM_DIR      = os.path.join(BASE_DIR, "multimodel_mean" + SUFFIX)
PL_TAUP_GEN_DIR     = os.path.join(BASE_DIR, "general" + SUFFIX)
ANALYTICAL_DIR_M123 = os.path.join(BASE_DIR, "analytical_models_linear" + SUFFIX)
ANALYTICAL_DIR_M5   = os.path.join(BASE_DIR, "analytical_models" + SUFFIX)
EMP_DIR             = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2"

# Re-run multimodel_mean.py on the *_method2 outputs before plotting.
# If False, it still runs automatically when the multimodel files are missing.
RUN_MULTIMODEL_MEAN = False

# ── Model inventory: the SAME models in panels b and c ──────────────────────
sys.path.insert(0, BASE_DIR)
import config as cfg
EXCLUDE = ["CNRM_CM61"]     # duplicate of CNRM_CM6

def f_emp(m): return os.path.join(EMP_DIR, f"tas_vs_extreme_precip_{m}.nc")
def f_hom(m): return os.path.join(ANALYTICAL_DIR_M123, f"{m}_analytical_models.nc")
def f_m5(m):  return os.path.join(ANALYTICAL_DIR_M5, f"{m}_analytical_model5_linear.nc")

REQUIRED = [("CMIP6 empirical", f_emp), ("homogeneous M1–M3", f_hom), ("SpEx M5", f_m5)]
MODELS = [m for m in cfg.MODELS
          if m not in EXCLUDE and all(os.path.exists(p(m)) for _, p in REQUIRED)]

FIG_NAME = f"Fig3_gamma_decomposition_and_scaling_{RUN_LABEL}"

print(f"Run                 : {RUN_LABEL} (Method 2)")
print(f"Panel b input       : {PL_TAUP_MM_DIR}")
print(f"Panel c M1–M3 input : {ANALYTICAL_DIR_M123}")
print(f"Panel c M5 input    : {ANALYTICAL_DIR_M5}")
print(f"Panel c CMIP6 input : {EMP_DIR}")
print(f"Models (panels b, c): {len(MODELS)}  (excluded: {', '.join(EXCLUDE) or 'none'})")
for name, p in REQUIRED:
    miss = [m for m in cfg.MODELS if m not in EXCLUDE and not os.path.exists(p(m))]
    if miss:
        print(f"  missing {name}: {', '.join(miss)}  -> left out of panels b and c")

# ── Analysis settings ────────────────────────────────────────────────────────
WET_DAY_THRESH = 0.1
BASELINE_START, BASELINE_END = 1980, 2009
HIST_START,     HIST_END     = 1980, 2009
FUTURE_START,   FUTURE_END   = 2075, 2099

# ── Nature print-spec figure sizing ──────────────────────────────────────────
MM_TO_IN    = 1 / 25.4
FIG_SIZE_IN = (180 * MM_TO_IN, 160 * MM_TO_IN)
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
    "legend.fontsize":   FS - 2,
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
# Panel a data: Gamma-PDF schematic (GFDL-ESM4)
# ══════════════════════════════════════════════════════════════════════════
FILES_BASELINE_P99 = [
    f"{GFDL_DATA_DIR}/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19700101-19891231.nc",
    f"{GFDL_DATA_DIR}/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19900101-20091231.nc",
]
FILES_HIST_PERIOD = [
    f"{GFDL_DATA_DIR}/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_19900101-20091231.nc",
    f"{GFDL_DATA_DIR}/pr_day_GFDL-ESM4_historical_r1i1p1f1_gr1_20100101-20141231.nc",
]


def compute_grid_stats(file_list, time_slice, wet_thresh=WET_DAY_THRESH):
    """Per-grid-cell Pwet, PL = Var/mean, tauP = 1 - Pwet/PL, wet-day P99, n_wet."""
    ds = xr.open_mfdataset(file_list, combine="by_coords", use_cftime=True,
                           chunks={"time": 1000, "lat": -1, "lon": -1})
    pr     = ds["pr"].sel(time=time_slice)
    pr_mm  = (pr * 86400.0).astype("float32")
    pr_wet = pr_mm.where(pr_mm >= wet_thresh)

    n_wet_grid = pr_wet.count(dim="time")
    Pwet_grid  = pr_wet.mean(dim="time", skipna=True)
    var_grid   = pr_wet.var(dim="time", skipna=True, ddof=0)
    PL_grid    = var_grid / Pwet_grid
    tauP_grid  = 1.0 - (Pwet_grid / PL_grid)

    P99_grid = pr_wet.quantile(0.99, dim="time", skipna=True)
    if "quantile" in P99_grid.dims:
        P99_grid = P99_grid.drop_vars("quantile").squeeze("quantile", drop=True)

    out = xr.Dataset({"Pwet": Pwet_grid, "PL": PL_grid, "tauP": tauP_grid,
                      "P99": P99_grid, "n_wet": n_wet_grid}).compute()
    ds.close()
    return out["Pwet"], out["PL"], out["tauP"], out["P99"], out["n_wet"]


def area_weighted_scalar(field, min_n_wet=None, n_wet_grid=None):
    """cos(lat)-weighted scalar mean, ignoring NaN cells and cells with few wet days."""
    lat_weights = np.clip(np.cos(np.deg2rad(field.lat.values)), 0, None)
    w_grid = xr.DataArray(np.broadcast_to(lat_weights[:, None], field.shape),
                          dims=field.dims, coords=field.coords)
    valid = np.isfinite(field)
    if min_n_wet is not None and n_wet_grid is not None:
        valid = valid & (n_wet_grid >= min_n_wet)
    num = (field.where(valid) * w_grid.where(valid)).sum(skipna=True)
    den = w_grid.where(valid).sum(skipna=True)
    return float(num / den)


# Baseline (1980-2009)
Pwet_grid_base, PL_grid_base, tauP_grid_base, P99_grid_base, n_wet_grid_base = compute_grid_stats(
    FILES_BASELINE_P99, time_slice=slice(f"{BASELINE_START}-01-01", f"{BASELINE_END}-12-31"))
MIN_N_WET = 30

Pwet_ref   = area_weighted_scalar(Pwet_grid_base, MIN_N_WET, n_wet_grid_base)
PL_ref     = area_weighted_scalar(PL_grid_base,   MIN_N_WET, n_wet_grid_base)
P99_global = area_weighted_scalar(P99_grid_base,  MIN_N_WET, n_wet_grid_base)
tau_P_base = 1 - (Pwet_ref / PL_ref)
k_base     = 1 - tau_P_base

# Historical (same window here — matches source notebook)
Pwet_grid_hist, PL_grid_hist, tauP_grid_hist, P99_grid_hist, n_wet_grid_hist = compute_grid_stats(
    FILES_HIST_PERIOD, time_slice=slice(f"{HIST_START}-01-01", f"{HIST_END}-12-31"))
MIN_N_WET_HIST = 30

Pwet_hist  = area_weighted_scalar(Pwet_grid_hist, MIN_N_WET_HIST, n_wet_grid_hist)
PL_hist    = area_weighted_scalar(PL_grid_hist,   MIN_N_WET_HIST, n_wet_grid_hist)
tau_P_hist = 1 - (Pwet_hist / PL_hist)
k_hist     = 1 - tau_P_hist
gamma_hist = stats.gamma(a=k_hist, loc=0, scale=PL_hist)

# ΔT (2075-2099 vs 1980-2009) from GFDL tas_anomaly
ds_tas = xr.open_dataset(GFDL_TAS_FILE)
tas_anomaly = ds_tas["tas_anomaly"].values.flatten()
years       = ds_tas["time"].values
ds_tas.close()
baseline_mask = (years >= BASELINE_START) & (years <= BASELINE_END)
future_mask   = (years >= FUTURE_START)   & (years <= FUTURE_END)
DELTA_T = np.nanmean(tas_anomaly[future_mask]) - np.nanmean(tas_anomaly[baseline_mask])

sl_pl = 0.077    # P_L increases 7.7% per °C
sl_pw = 0.0256   # P_wet increases 2.56% per °C

# Future 1: P_L changes, P_wet fixed
PL_fut1    = PL_ref * (1 + sl_pl * DELTA_T)
Pwet_fut1  = Pwet_ref
k_fut1     = Pwet_fut1 / PL_fut1
gamma_fut1 = stats.gamma(a=k_fut1, loc=0, scale=PL_fut1)

# Future 2: both P_L and P_wet change
PL_fut2    = PL_ref   * (1 + sl_pl * DELTA_T)
Pwet_fut2  = Pwet_ref * (1 + sl_pw * DELTA_T)
k_fut2     = Pwet_fut2 / PL_fut2
gamma_fut2 = stats.gamma(a=k_fut2, loc=0, scale=PL_fut2)


def fmt_pct(value, decimals=0):
    """Format a percentage, avoiding the 'negative zero' artifact."""
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.{decimals}f}"


mean_hist = k_hist * PL_hist
mean_fut1 = k_fut1 * PL_fut1
mean_fut2 = k_fut2 * PL_fut2

A_hist = gamma_hist.sf(P99_global)
A_fut1 = gamma_fut1.sf(P99_global)
A_fut2 = gamma_fut2.sf(P99_global)

frac_change_mean_fut1 = (mean_fut1 - mean_hist) / mean_hist * 100
frac_change_mean_fut2 = (mean_fut2 - mean_hist) / mean_hist * 100
frac_change_area_fut1 = (A_fut1 - A_hist) / A_hist * 100
frac_change_area_fut2 = (A_fut2 - A_hist) / A_hist * 100

print(f"DELTA_T = {DELTA_T:.3f} °C")
print(f"Historical exceedance frac: {A_hist:.4f}")
print(f"Future 1 exceedance frac:   {A_fut1:.4f}")
print(f"Future 2 exceedance frac:   {A_fut2:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# Multimodel means on the *_method2 outputs (runs multimodel_mean.py)
# ══════════════════════════════════════════════════════════════════════════
MM_FILES = ["multimodel_PL_rel.nc", "multimodel_Pwet_rel_global.nc", "multimodel_fwet_rel_global.nc"]
mm_missing = [f for f in MM_FILES if not os.path.exists(os.path.join(PL_TAUP_MM_DIR, f))]

if RUN_MULTIMODEL_MEAN or mm_missing:
    code = r'''
import runpy
import config as cfg
sfx = "_method2"
cfg.GENERAL_DIR    = cfg.GENERAL_DIR.rstrip("/")    + sfx
cfg.MULTIMODEL_DIR = cfg.MULTIMODEL_DIR.rstrip("/") + sfx
print("reading", cfg.GENERAL_DIR)
print("writing", cfg.MULTIMODEL_DIR)
runpy.run_path("multimodel_mean.py", run_name="__main__")
'''
    env = {k: v for k, v in os.environ.items() if k != "BASELINE_METHOD"}  # keep default (fixed)
    res = subprocess.run([sys.executable, "-c", code], cwd=BASE_DIR, env=env,
                         capture_output=True, text=True)
    print(res.stdout[-3000:])
    print(res.stderr[-3000:])

# ══════════════════════════════════════════════════════════════════════════
# Panel b data: P_L, P̄_wet, f_wet vs ΔT
# ══════════════════════════════════════════════════════════════════════════
COMBINED_VARIABLES = [
    dict(label="P_L",  mm_file="multimodel_PL_rel.nc",
         var_key="PL_rel",          dt_file="{model}_tas_dT_31yr.nc",
         dt_key="dT_31yr",  scale=100, color="#1f4e79",
         legend_label=r"$\Delta P_L / P_L$"),
    dict(label="Pwet", mm_file="multimodel_Pwet_rel_global.nc",
         var_key="Pwet_rel_global", dt_file="{model}_Pwet_annual.nc",
         dt_key="dT_annual", scale=100, color="#c0392b",
         legend_label=r"$\delta\bar{P}_{wet}/\bar{P}_{wet}$"),
    dict(label="fwet", mm_file="multimodel_fwet_rel_global.nc",
         var_key="fwet_rel_global", dt_file="{model}_fwet_annual.nc",
         dt_key="dT_annual", scale=100, color="#2e7d32",
         legend_label=r"$\delta f_{wet}/f_{wet}$"),
]


def _load_model_series(vinfo):
    """Return {model: (dT, y)} for one variable, from the multimodel file."""
    vkey = vinfo["var_key"]
    model_data = {}
    ds_mm  = xr.open_dataset(os.path.join(PL_TAUP_MM_DIR, vinfo["mm_file"]))
    models = [k.replace(f"{vkey}_", "") for k in ds_mm.data_vars
              if k.startswith(f"{vkey}_")
              and k not in (f"{vkey}_mean", f"{vkey}_std", f"{vkey}_min", f"{vkey}_max")]
    for model in models:
        if model not in MODELS:          # same model set as panel c
            continue
        dt_path = os.path.join(PL_TAUP_GEN_DIR, model, vinfo["dt_file"].format(model=model))
        if not os.path.exists(dt_path) or f"{vkey}_{model}" not in ds_mm:
            continue
        var_da = ds_mm[f"{vkey}_{model}"]
        dt_da  = xr.open_dataset(dt_path)[vinfo["dt_key"]]
        var_da, dt_da = xr.align(var_da, dt_da, join="inner")
        mask = np.isfinite(var_da.values) & np.isfinite(dt_da.values)
        if mask.sum() < 4:
            continue
        model_data[model] = (dt_da.values[mask], var_da.values[mask] * vinfo["scale"])
    ds_mm.close()
    return model_data


def compute_multimodel_fit(vinfo, common_x=None):
    model_data = _load_model_series(vinfo)
    if not model_data:
        print(f"No data for {vinfo['label']} — skipping.")
        return None
    if common_x is None:
        all_dt   = np.concatenate([dt for dt, _ in model_data.values()])
        common_x = np.linspace(np.nanmin(all_dt), np.nanmax(all_dt), 200)

    model_fits_linear = []
    for dt_vals, y_vals in model_data.values():
        sl, ic, *_ = linregress(dt_vals, y_vals)
        model_fits_linear.append(sl * common_x + ic)
    model_fits_linear = np.array(model_fits_linear)

    linear_mean = np.nanmean(model_fits_linear, axis=0)
    linear_p05  = np.nanpercentile(model_fits_linear, 5,  axis=0)
    linear_p95  = np.nanpercentile(model_fits_linear, 95, axis=0)
    slope_mm, *_ = linregress(common_x, linear_mean)
    return dict(common_x=common_x, mean=linear_mean, p05=linear_p05,
                p95=linear_p95, slope=slope_mm, n_models=len(model_data))


panel_b_fits = {}
for vinfo in COMBINED_VARIABLES:
    fit = compute_multimodel_fit(vinfo)
    if fit is not None:
        panel_b_fits[vinfo["label"]] = (vinfo, fit)
        print(f"  {vinfo['label']:5s}  slope = {fit['slope']:.2f} %/°C  (n={fit['n_models']} models)")

# ══════════════════════════════════════════════════════════════════════════
# Panel c data: CMIP6 actual vs analytical decomposition models
# ══════════════════════════════════════════════════════════════════════════
def mm_stats_range(curves, dT_raw, common_x, dT_min=None, dT_max=None):
    if dT_min is None: dT_min = dT_raw[0]
    if dT_max is None: dT_max = dT_raw[-1]
    fits = []
    for c in curves:
        fit_mask = (dT_raw >= dT_min) & (dT_raw <= dT_max) & np.isfinite(c)
        if fit_mask.sum() < 5:
            continue
        sl, ic, *_ = linregress(dT_raw[fit_mask], c[fit_mask])
        fits.append(sl * common_x + ic)
    fits = np.array(fits)
    mean = np.nanmean(fits, axis=0)
    p05  = np.nanpercentile(fits,  5, axis=0)
    p95  = np.nanpercentile(fits, 95, axis=0)
    sl, ic, r, _, _ = linregress(common_x, mean)
    return mean, p05, p95, sl, ic, r


# COMMON_X from the actual CMIP6 tas_anomaly range
all_tas = []
for m in MODELS:
    ds  = xr.open_dataset(f_emp(m))
    tas = ds["tas_anomaly"].values.flatten()
    all_tas.append(tas[np.isfinite(tas)])
    ds.close()
all_tas    = np.concatenate(all_tas)
xmin, xmax = np.nanmin(all_tas), np.nanmax(all_tas)
COMMON_X   = np.linspace(xmin, xmax, 300)
print(f"CMIP6 tas range: {xmin:.2f} to {xmax:.2f} °C")

# Analytical Models 1–3
m1, m2, m3 = [], [], []
dT_m123 = None
for m in MODELS:
    ds = xr.open_dataset(f_hom(m))
    if dT_m123 is None:
        dT_m123 = ds["deltaT"].values
    m1.append(ds["dA_m1"].values * 100)
    m2.append(ds["dA_m2"].values * 100)
    m3.append(ds["dA_m3"].values * 100)
    ds.close()
print(f"Analytical M1/M2/M3 loaded: {len(m1)} models")

# Analytical Model 5 (SpEx)
m5 = []
dT_m5 = None
for m in MODELS:
    ds = xr.open_dataset(f_m5(m))
    if dT_m5 is None:
        dT_m5 = ds["deltaT"].values
    m5.append(ds["dA_m5"].values * 100)
    ds.close()
print(f"Analytical Model 5 (SpEx) loaded: {len(m5)} models")

# CMIP6 extreme area
fits_ea = []
for m in MODELS:
    ds   = xr.open_dataset(f_emp(m))
    tas  = ds["tas_anomaly"].values
    frac = ds["fraction_extreme_area"].values
    year = ds["time"].values
    ds.close()
    bm   = (year >= BASELINE_START) & (year <= BASELINE_END)
    A0   = np.nanmean(frac[bm])
    rel  = (frac - A0) / A0
    mask = np.isfinite(tas) & np.isfinite(rel)
    sl, ic, *_ = linregress(tas[mask], rel[mask])
    fits_ea.append(sl * COMMON_X + ic)
print(f"CMIP6 empirical loaded: {len(fits_ea)} models")

fits_ea = np.array(fits_ea)
ea_mean = np.nanmean(fits_ea, axis=0)
ea_p05  = np.nanpercentile(fits_ea,  5, axis=0)
ea_p95  = np.nanpercentile(fits_ea, 95, axis=0)
sl_cmip6, ic_cmip6, r_cmip6, _, _ = linregress(COMMON_X, ea_mean * 100)

m1_mean, m1_p05, m1_p95, sl_m1, ic_m1, r_m1 = mm_stats_range(np.array(m1), dT_m123, COMMON_X)
m2_mean, m2_p05, m2_p95, sl_m2, ic_m2, r_m2 = mm_stats_range(np.array(m2), dT_m123, COMMON_X)
m3_mean, m3_p05, m3_p95, sl_m3, ic_m3, r_m3 = mm_stats_range(np.array(m3), dT_m123, COMMON_X)
m5_mean, m5_p05, m5_p95, sl_m5, ic_m5, r_m5 = mm_stats_range(np.array(m5), dT_m5,   COMMON_X)

print(f"\nCMIP6 actual slope        = {sl_cmip6:.3f} %/°C")
print(f"M1 (P_L only, Pwet fixed) = {sl_m1:.3f} %/°C")
print(f"M2 (P_L + Pwet)           = {sl_m2:.3f} %/°C")
print(f"M3 (P_L + Pwet + fwet)    = {sl_m3:.3f} %/°C")
print(f"M5 (SpEx, gridded rates)  = {sl_m5:.3f} %/°C")


# ══════════════════════════════════════════════════════════════════════════
# Shared panel drawing
# ══════════════════════════════════════════════════════════════════════════
def draw_panel_a(ax_a, box_xy, legend_anchor):
    x_max = max(gamma_hist.ppf(0.9999), gamma_fut1.ppf(0.9999), gamma_fut2.ppf(0.9999))
    x = np.linspace(WET_DAY_THRESH, x_max, 3000)
    pdf_hist, pdf_fut1, pdf_fut2 = gamma_hist.pdf(x), gamma_fut1.pdf(x), gamma_fut2.pdf(x)

    ax_a.plot(x, pdf_hist, color="#1f4e79", lw=1.0, label="Historical (1980-2009)")
    ax_a.plot(x, pdf_fut1, color="#e08214", lw=1.0, ls="--",
              label=r"Future 1: $P_L$ changes, $\bar{P}_{wet}$ fixed")
    ax_a.plot(x, pdf_fut2, color="#c0392b", lw=1.0, ls="-.",
              label=r"Future 2: $P_L,\, \bar{P}_{wet}$ change")

    mask = x >= P99_global
    ax_a.fill_between(x[mask], 0, pdf_hist[mask], color="#1f4e79", alpha=0.20, lw=0,
                      label=f"Historical tail frac = {A_hist:.3f}")
    ax_a.fill_between(x[mask], pdf_hist[mask], np.maximum(pdf_fut1[mask], pdf_hist[mask]),
                      color="#e08214", alpha=0.22, lw=0,
                      label=f"Future 1 added tail frac = {A_fut1 - A_hist:.3f}")
    ax_a.fill_between(x[mask], pdf_hist[mask], np.maximum(pdf_fut2[mask], pdf_hist[mask]),
                      color="#c0392b", alpha=0.22, lw=0,
                      label=f"Future 2 added tail frac = {A_fut2 - A_hist:.3f}")

    ax_a.axvline(P99_global, color="0.2", lw=0.6, ls=":")
    ax_a.text(P99_global + x_max * 0.02, ax_a.get_ylim()[1], r"$P_{99}$", va="bottom", ha="left")

    ax_a.text(box_xy[0], box_xy[1],
              r"$\Delta\bar{P}_{wet}$: " +
              rf"{fmt_pct(frac_change_mean_fut1)}$\%$ / {fmt_pct(frac_change_mean_fut2)}$\%$" + "\n" +
              r"$\Delta$tail frac: " +
              rf"{fmt_pct(frac_change_area_fut1)}$\%$ / {fmt_pct(frac_change_area_fut2)}$\%$",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=FS - 1,
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.4", lw=0.6))

    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"Wet-day precipitation (mm day$^{-1}$)")
    ax_a.set_ylabel(r"Probability density (mm day$^{-1}$)$^{-1}$")
    ax_a.set_xlim(0, x[-1])
    ax_a.set_ylim(bottom=max(pdf_hist.min(), pdf_fut1.min(), pdf_fut2.min(), 1e-7))
    ax_a.legend(loc="center right", bbox_to_anchor=legend_anchor, frameon=False, fontsize=FS - 2)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    panel_label(ax_a, "a")


def draw_panel_b_curves(ax_b):
    for vinfo, fit in panel_b_fits.values():
        ax_b.fill_between(fit["common_x"], fit["p05"], fit["p95"], alpha=0.15, color=vinfo["color"])
        ax_b.plot(fit["common_x"], fit["mean"], color=vinfo["color"], lw=1.0,
                  label=vinfo["legend_label"] +
                  rf" (slope = {fit['slope']:.2f}$\%$/$^{{\circ}}$C)")


def draw_panel_c_curves(ax_c):
    ax_c.fill_between(COMMON_X, ea_p05 * 100, ea_p95 * 100, alpha=0.12, color="blue",
                      label=r"5–95$\%$ spread")
    ax_c.plot(COMMON_X, ea_mean * 100, color="blue", lw=1.2, ls="-", zorder=8,
              label=rf"CMIP6 MM  ({sl_cmip6:.2f} $\%$/$^{{\circ}}$C)")
    ax_c.fill_between(COMMON_X, m1_p05, m1_p95, alpha=0.10, color="#e43a0b")
    ax_c.plot(COMMON_X, m1_mean, color="#e81b0c", lw=1.0, ls="--", zorder=5,
              label=rf"Homogeneous $P_L$-only ($\bar{{P}}_{{wet}}$ fixed) ({sl_m1:.2f} $\%$/$^{{\circ}}$C)")
    ax_c.fill_between(COMMON_X, m2_p05, m2_p95, alpha=0.08, color="#0a73ca")
    ax_c.plot(COMMON_X, m2_mean, color="#074096", lw=1.0, ls="--", zorder=5, alpha=0.9,
              label=rf"Homogeneous $P_L + \bar{{P}}_{{wet}}$  ({sl_m2:.2f} $\%$/$^{{\circ}}$C)")
    ax_c.fill_between(COMMON_X, m3_p05, m3_p95, alpha=0.10, color="#33a02c")
    ax_c.plot(COMMON_X, m3_mean, color="#33a02c", lw=1.0, ls="-", zorder=6,
              label=rf"Homogeneous $P_L + \bar{{P}}_{{wet}} + f_{{wet}}$  ({sl_m3:.2f} $\%$/$^{{\circ}}$C)")
    ax_c.fill_between(COMMON_X, m5_p05, m5_p95, alpha=0.10, color="#b15928")
    ax_c.plot(COMMON_X, m5_mean, color="red", lw=1.0, ls="-.", zorder=7,
              label=rf"SpEx Gamma MM mean  ({sl_m5:.2f} $\%$/$^{{\circ}}$C)")
    ax_c.set_xlabel(r"$\Delta T$  ($^{\circ}$C)")
    ax_c.set_ylabel(r"$\delta A / A_0$  ($\%$)")
    ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"{v:.0f}$\%$"))
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    panel_label(ax_c, "c")


# ══════════════════════════════════════════════════════════════════════════
# Layout 1: a, b on top row + c centred in the bottom row
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(180 * MM_TO_IN, 150 * MM_TO_IN))
outer = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4, figure=fig)
gs_ab = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.4)
gs_c  = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[0.15, 0.7, 0.15])

ax_a = fig.add_subplot(gs_ab[0, 0])
draw_panel_a(ax_a, box_xy=(1.0, 0.97), legend_anchor=(1.1, 0.6))

ax_b = fig.add_subplot(gs_ab[0, 1])
draw_panel_b_curves(ax_b)
ax_b.axhline(0, color="dimgray", lw=0.2, ls="--")
ax_b.set_xlim(-1.4, 7.4)
ax_b.set_xticks(np.arange(-1, 8, 1))
ax_b.set_xlabel(r"$\Delta T$ ($^{\circ}$C)")
ax_b.set_ylabel(r"Relative change ($\%$)")
ax_b.legend(loc="upper left", frameon=False)
ax_b.grid(True, ls="--", alpha=0.1)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
panel_label(ax_b, "b")

ax_c = fig.add_subplot(gs_c[0, 1])
draw_panel_c_curves(ax_c)
ax_c.axhline(0, color="gray", lw=0.4, ls=":", alpha=0.1)
ax_c.axvline(0, color="gray", lw=0.4, ls=":", alpha=0.1)
ax_c.legend(loc="upper left", framealpha=0.3)
ax_c.grid(True, ls="--", alpha=0.35, lw=0.2)

plt.savefig(f"{FIG_NAME}_layout1.png", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{FIG_NAME}_layout1.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIG_NAME}_layout1.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════
# Layout 2: a on top + b, c below
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=FIG_SIZE_IN)
outer = gridspec.GridSpec(2, 1, height_ratios=[0.85, 1], hspace=0.35, figure=fig)

ax_a = fig.add_subplot(outer[0])
draw_panel_a(ax_a, box_xy=(0.97, 0.95), legend_anchor=(0.97, 0.5))

gs_bc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.35)

ax_b = fig.add_subplot(gs_bc[0, 0])
draw_panel_b_curves(ax_b)
ax_b.axhline(0, color="dimgray", lw=0.4, ls="--")
ax_b.set_xlabel(r"$\Delta T$ ($^{\circ}$C)")
ax_b.set_ylabel(r"Relative change ($\%$)")
ax_b.legend(loc="upper left", frameon=False)
ax_b.grid(True, ls="--", alpha=0.3)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
panel_label(ax_b, "b")

ax_c = fig.add_subplot(gs_bc[0, 1])
draw_panel_c_curves(ax_c)
ax_c.axhline(0, color="gray", lw=0.4, ls=":", alpha=0.6)
ax_c.axvline(0, color="gray", lw=0.4, ls=":", alpha=0.6)
ax_c.legend(loc="upper left", framealpha=0.6)
ax_c.grid(True, ls="--", alpha=0.35, lw=0.5)

plt.savefig(f"{FIG_NAME}.png", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{FIG_NAME}.pdf", bbox_inches="tight")
plt.close(fig)
print(f"Saved → {FIG_NAME}.png/.pdf")
