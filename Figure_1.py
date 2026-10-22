#!/usr/bin/env python
"""
PRECIP_AREA — Combined Figure 1: metric definition + CMIP6–ERA5 validation

  a — definition of the extreme-precipitation-area metric A(t) (4-step schematic, ERA5)
  b — time evolution of δA/A0 (CMIP6 multimodel mean + range, vs. ERA5)
  c — δA/A0 scaling against ΔT (CMIP6 vs. ERA5)

a1–a4 use the full-resolution ERA5 daily precipitation and its 1980–2009 local P99.
b, c use the per-model CMIP6 fraction_extreme_area / tas_anomaly files from
compute_empirical_area.py (Method 2 inventory and ΔT) and the ERA5 area-fraction /
T2m series, all relative to the 1980–2009 baseline.
"""

import os
import sys

import numpy as np
import xarray as xr
from scipy.stats import linregress

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Data paths ───────────────────────────────────────────────────────────────
ERA5_DIR  = "/nfs_home/dmuriar/PRECIP_AREA/ERA5"
CMIP6_DIR = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2"
sys.path.insert(0, "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP")
import config as cfg
EXCLUDE = ["CNRM_CM61"]                           # duplicate of CNRM_CM6

# ── Analysis settings ────────────────────────────────────────────────────────
BASELINE_START = 1980
BASELINE_END   = 2009
SELECTED_DATE  = "2017-08-10"

# ── Nature print-spec figure sizing ──────────────────────────────────────────
MM_TO_IN    = 1 / 25.4
FIG_SIZE_IN = (180 * MM_TO_IN, 150 * MM_TO_IN)
DPI = 300
FS  = 9

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
# Panel a data: metric-definition schematic
# ══════════════════════════════════════════════════════════════════════════
# a1: historical (1980-2009) local P99 threshold
threshold = xr.open_dataarray(os.path.join(ERA5_DIR, "era5_precip_99percentile_1980-2009.nc"))
if "longitude" in threshold.dims and "latitude" in threshold.dims:
    threshold = threshold.transpose("latitude", "longitude")
threshold_mm = threshold * 1000
vmax_thresh  = float(threshold_mm.quantile(0.98))

# Full precip record (meters)
precip_full = xr.open_dataset(
    os.path.join(ERA5_DIR, "era5_precip_1950_2019_daily_combined.nc"))["precip"]
precip_full = precip_full.transpose("time", "latitude", "longitude")

lat_rad      = np.deg2rad(precip_full["latitude"])
cos_lat_da   = xr.DataArray(np.cos(lat_rad), coords=[precip_full["latitude"]], dims=["latitude"])
denom_static = (xr.ones_like(threshold).sum(dim="longitude") * cos_lat_da).sum()

# Baseline mean area fraction, A_baseline
precip_baseline = precip_full.sel(time=slice("1980-01-01", "2009-12-31"))
baseline_exceed = (precip_baseline > threshold).astype(int)
baseline_numerator = (baseline_exceed.sum(dim="longitude") * cos_lat_da).sum(dim="latitude")
baseline_daily_fraction = baseline_numerator / denom_static
A_baseline = float(baseline_daily_fraction.mean())

# a2/a3: one example day
example_day    = precip_full.sel(time=SELECTED_DATE).squeeze()
example_day_mm = example_day * 1000
vmax_day       = float(example_day_mm.quantile(0.98))
exceed_mask    = (example_day > threshold).astype(int)

# a4: A(t) time series (JJA 2017) with the example day marked
season_window    = precip_full.sel(time=slice("2017-06-01", "2017-08-31"))
season_exceed    = (season_window > threshold).astype(int)
season_numerator = (season_exceed.sum(dim="longitude") * cos_lat_da).sum(dim="latitude")
season_daily_fraction_pct = (season_numerator / denom_static).values * 100
season_dates = season_window["time"].values
selected_idx = list(season_window["time"].dt.strftime("%Y-%m-%d").values).index(SELECTED_DATE)

# ══════════════════════════════════════════════════════════════════════════
# Panels b, c data: CMIP6 + ERA5 validation
# ══════════════════════════════════════════════════════════════════════════
def emp_file(model):
    return os.path.join(CMIP6_DIR, f"tas_vs_extreme_precip_{model}.nc")


def load_empirical(model):
    """Annual extreme-area fraction and ΔT, indexed by the REAL year."""
    ds   = xr.open_dataset(emp_file(model))
    frac = ds["fraction_extreme_area"].rename({"time": "year"}).load()
    tas  = ds["tas_anomaly"].rename({"time": "year"}).load()
    ds.close()
    return frac, tas


MODELS  = [m for m in cfg.MODELS if m not in EXCLUDE and os.path.exists(emp_file(m))]
MISSING = [m for m in cfg.MODELS if m not in EXCLUDE and not os.path.exists(emp_file(m))]
print(f"CMIP6 models: {len(MODELS)}  (excluded: {', '.join(EXCLUDE) or 'none'}; "
      f"missing: {', '.join(MISSING) or 'none'})")

# ── Panel b: time evolution ─────────────────────────────────────────────────
datasets = []
for m in MODELS:
    frac, _ = load_empirical(m)
    da = frac.rename({"year": "time"})
    baseline = da.sel(time=slice(BASELINE_START, BASELINE_END)).mean()
    datasets.append((da - baseline) / baseline)

combined = xr.concat(datasets, dim="model", join="inner")   # years common to all models
mean_rel = combined.mean(dim="model") * 100
p05_rel  = combined.quantile(0.05, dim="model") * 100
p95_rel  = combined.quantile(0.95, dim="model") * 100
END_YEAR = int(mean_rel["time"].values[-1])
print(f"Panel b common period: {int(mean_rel['time'].values[0])}–{END_YEAR}")

era5_area = xr.open_dataarray(os.path.join(ERA5_DIR, "fraction_extreme_area_daily_1950_2019.nc"))
if not np.issubdtype(era5_area["time"].dtype, np.datetime64):
    era5_area = xr.decode_cf(era5_area.to_dataset(name="var"))["var"]

era5_annual = era5_area.resample(time="1YE").mean()
era5_annual["time"] = era5_annual["time"].dt.year
era5_baseline = era5_annual.sel(time=slice(BASELINE_START, BASELINE_END)).mean()
era5_rel = (era5_annual - era5_baseline) / era5_baseline * 100

# ── Panel c: scaling against warming, per-model linear fits ─────────────────
all_tas = []
for m in MODELS:
    _, tas = load_empirical(m)
    all_tas.append(tas.values[np.isfinite(tas.values)])
all_tas  = np.concatenate(all_tas)
COMMON_X = np.linspace(np.nanmin(all_tas), np.nanmax(all_tas), 200)

model_fits = []
for m in MODELS:
    frac, tas = load_empirical(m)
    year = frac["year"].values
    frac = frac.values
    tas  = tas.values

    baseline_mask = (year >= BASELINE_START) & (year <= BASELINE_END)
    if baseline_mask.sum() == 0:
        continue
    baseline_frac = np.nanmean(frac[baseline_mask])
    rel_change = (frac - baseline_frac) / baseline_frac

    mask = np.isfinite(tas) & np.isfinite(rel_change)
    slope, intercept, *_ = linregress(tas[mask], rel_change[mask])
    model_fits.append(slope * COMMON_X + intercept)

model_fits = np.array(model_fits)
fit_mean = np.nanmean(model_fits, axis=0)
fit_p05  = np.nanpercentile(model_fits, 5,  axis=0)
fit_p95  = np.nanpercentile(model_fits, 95, axis=0)
slope_cmip_pct, *_ = linregress(COMMON_X, fit_mean)
slope_cmip_pct *= 100

t2m_anomaly = xr.open_dataarray(os.path.join(ERA5_DIR, "t2m_global_diff_1950_2019.nc"))
x_era5 = t2m_anomaly.values.flatten()
y_era5_full = era5_rel.values
repeat_factor = int(len(x_era5) / len(y_era5_full))
y_era5 = np.repeat(y_era5_full, repeat_factor) if repeat_factor > 1 else y_era5_full

mask_era5 = np.isfinite(x_era5) & np.isfinite(y_era5)
coeffs_era5 = np.polyfit(x_era5[mask_era5], y_era5[mask_era5], deg=1)
slope_era5, intercept_era5 = coeffs_era5
trend_era5  = np.poly1d(coeffs_era5)
x_line_era5 = np.linspace(x_era5[mask_era5].min(), x_era5[mask_era5].max(), 100)
y_line_era5 = trend_era5(x_line_era5)

print(f"CMIP6 MM mean slope = {slope_cmip_pct:.2f} %/°C  (n={len(model_fits)} models)")
print(f"ERA5 slope           = {slope_era5:.2f} %/°C")

# ══════════════════════════════════════════════════════════════════════════
# Combined plot: a (schematic, 2x2) on top + b, c below
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(180 * MM_TO_IN, 200 * MM_TO_IN))
outer = gridspec.GridSpec(2, 1, height_ratios=[1.3, 1], hspace=0.23, figure=fig)
proj = ccrs.Robinson(central_longitude=180)

# ── Panel a ──────────────────────────────────────────────────────────────────
gs_a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0.55, wspace=0.28)

ax_a1 = fig.add_subplot(gs_a[0, 0], projection=proj)
ax_a1.set_global()
ax_a1.add_feature(cfeature.COASTLINE, linewidth=0.3)
m_a1 = ax_a1.pcolormesh(threshold_mm["longitude"], threshold_mm["latitude"], threshold_mm,
                        transform=ccrs.PlateCarree(), cmap="Greens", vmin=0, vmax=vmax_thresh)
plt.colorbar(m_a1, ax=ax_a1, orientation="horizontal", pad=0.06, shrink=0.85, aspect=18,
             label=r"$P^{hist}_{99}(x)$ (mm day$^{-1}$)")
ax_a1.set_title("1. Historical local\nthreshold", fontsize=FS - 1)
panel_label(ax_a1, "a")

ax_a2 = fig.add_subplot(gs_a[0, 1], projection=proj)
ax_a2.set_global()
ax_a2.add_feature(cfeature.COASTLINE, linewidth=0.3)
m_a2 = ax_a2.pcolormesh(example_day_mm["longitude"], example_day_mm["latitude"], example_day_mm,
                        transform=ccrs.PlateCarree(), cmap="Blues", vmin=0, vmax=vmax_day)
plt.colorbar(m_a2, ax=ax_a2, orientation="horizontal", pad=0.06, shrink=0.85, aspect=18,
             label=r"$P(x,t)$ (mm day$^{-1}$)")
ax_a2.set_title(f"2. Daily field\n{SELECTED_DATE}", fontsize=FS - 1)

ax_a3 = fig.add_subplot(gs_a[1, 0], projection=proj)
ax_a3.set_global()
ax_a3.add_feature(cfeature.COASTLINE, linewidth=0.3)
cmap_mask = ListedColormap(["lightgray", "yellow"])
ax_a3.pcolormesh(exceed_mask["longitude"], exceed_mask["latitude"], exceed_mask,
                 transform=ccrs.PlateCarree(), cmap=cmap_mask, vmin=0, vmax=1)
ax_a3.set_title(r"3. Exceedance mask $I(x,t)$", fontsize=FS - 1)
ax_a3.text(0.5, -0.22, r"$I(x,t) = \mathbf{1}\left\{P(x,t) > P_{99}^{hist}(x)\right\}$",
           transform=ax_a3.transAxes, fontsize=FS - 1, ha="center", va="top")

ax_a4 = fig.add_subplot(gs_a[1, 1])
ax_a4.plot(season_dates, season_daily_fraction_pct, color="gray", lw=0.8)
ax_a4.scatter(season_dates[selected_idx], season_daily_fraction_pct[selected_idx],
              color="red", s=18, zorder=5, label=f"$A(t)$, {SELECTED_DATE}")
ax_a4.axhline(A_baseline * 100, color="black", linestyle="--", lw=0.2, label=r"$A_{baseline}$")
ax_a4.set_ylabel(r"$A(t)$ (%)")
ax_a4.set_title("4. Area aggregation", fontsize=FS - 1)
ax_a4.legend(loc="upper left", fontsize=FS - 2)
ax_a4.tick_params(axis="x", rotation=45, labelsize=FS - 2)

# connecting arrows: a1 -> a2 -> a3 -> a4
arrow_style = dict(arrowstyle="-|>", mutation_scale=8, color="black", linewidth=0.7, zorder=10)
fig.add_artist(ConnectionPatch(xyA=(1.0, 0.5), coordsA=ax_a1.transAxes,
                               xyB=(0.0, 0.5), coordsB=ax_a2.transAxes, **arrow_style))
fig.add_artist(ConnectionPatch(xyA=(0.0, 0.0), coordsA=ax_a2.transAxes,
                               xyB=(1.0, 1.0), coordsB=ax_a3.transAxes, **arrow_style))
fig.add_artist(ConnectionPatch(xyA=(1.0, 0.8), coordsA=ax_a3.transAxes,
                               xyB=(0.0, 0.8), coordsB=ax_a4.transAxes, **arrow_style))

# ── Panels b, c ──────────────────────────────────────────────────────────────
gs_bc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.35)

ax_b = fig.add_subplot(gs_bc[0, 0])
ax_b.fill_between(mean_rel["time"], p05_rel, p95_rel, color="blue", alpha=0.15,
                  label="CMIP6 5–95% range")
ax_b.plot(mean_rel["time"], mean_rel, color="blue", lw=1.0, label="CMIP6 Multimodel Mean")
ax_b.plot(era5_rel["time"], era5_rel, color="red", lw=0.8, label="ERA5")
ax_b.axhline(0, color="dimgray", lw=0.4, ls="--")
ax_b.set_xlabel("Year")
ax_b.set_ylabel(r"$\delta A \,/\, A_{0}\ (\%)$")
ax_b.legend(loc="upper left", framealpha=0.5)
ax_b.grid(ls="--", alpha=0.2)
ax_b.set_xlim(1945, 2100)
ax_b.xaxis.set_major_locator(plt.MultipleLocator(25))
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
panel_label(ax_b, "b")

ax_c = fig.add_subplot(gs_bc[0, 1])
ax_c.fill_between(COMMON_X, fit_p05 * 100, fit_p95 * 100, color="blue", alpha=0.15,
                  label="CMIP6 5–95% range")
ax_c.plot(COMMON_X, fit_mean * 100, color="blue", lw=1.0,
          label=f"CMIP6 MM mean  ({slope_cmip_pct:.2f} %/°C)")
ax_c.plot(x_line_era5, y_line_era5, color="red", lw=1.0, zorder=5,
          label=f"ERA5  ({slope_era5:.2f} %/°C)")
CC_RATE = 7.0
ax_c.axhline(CC_RATE, color="#6A6AE0", lw=0.5, ls="--", zorder=1, label=f"CC  ({CC_RATE:.0f} %)")
ax_c.axhline(0, color="dimgray", lw=0.2, ls="--")
ax_c.axvline(0, color="dimgray", lw=0.2, ls="--")
ax_c.set_xlabel(r"$\Delta T$  (°C)")
ax_c.set_ylabel(r"$\delta A \,/\, A_{0}\ (\%)$")
ax_c.legend(loc="upper left", framealpha=0.5)
ax_c.grid(ls="--", alpha=0.2)
ax_c.xaxis.set_major_locator(plt.MultipleLocator(1))
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
panel_label(ax_c, "c")

plt.savefig("Fig1_combined_definition_and_validation.png", dpi=DPI, bbox_inches="tight")
plt.savefig("Fig1_combined_definition_and_validation.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved → Fig1_combined_definition_and_validation.png/.pdf")

# ══════════════════════════════════════════════════════════════════════════
# CMIP6 5th / 95th percentiles
# ══════════════════════════════════════════════════════════════════════════
model_slopes = (model_fits[:, -1] - model_fits[:, 0]) / (COMMON_X[-1] - COMMON_X[0]) * 100
s05, s95 = np.percentile(model_slopes, [5, 95])

b05   = float(p05_rel.sel(time=END_YEAR))
b95   = float(p95_rel.sel(time=END_YEAR))
bmean = float(mean_rel.sel(time=END_YEAR))

print(f"Panel b — CMIP6 δA/A0 in {END_YEAR}: mean = {bmean:.1f} %, "
      f"5th = {b05:.1f} %, 95th = {b95:.1f} %")
print(f"Panel c — CMIP6 slope: mean = {slope_cmip_pct:.2f} %/°C, "
      f"5th = {s05:.2f} %/°C, 95th = {s95:.2f} %/°C  (n = {len(model_slopes)} models)")
print(f"ERA5 slope = {slope_era5:.2f} %/°C")
