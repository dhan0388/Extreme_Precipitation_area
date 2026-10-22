#!/usr/bin/env python
"""
Map_of_rates.py — CMIP6 multimodel maps of per-cell warming rates

  1. Load per-model gridded rates (sl_pl, sl_pw, sl_fw, sl_tp) onto a common 1°×1° grid
  2. Mask unreliable TauP cells (multimodel-mean baseline TauP)
  3. Figures:
       MM_gridded_rates_maps_2x2.png     — P_L, P_wet, f_wet, TauP (MM mean + 80% stippling)
       MM_gridded_rates_maps.png         — P_L, P_wet, f_wet (1×3, panels a–c)
       MM_gridded_rates_maps_mean_std.png — MM mean and MM std (3×2)
"""

import os

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

plt.rcParams["figure.dpi"]       = 120
plt.rcParams["text.usetex"]      = False
plt.rcParams["font.size"]        = 25
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"]      = "STIXGeneral"

# ══════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════
GEN_DIR     = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP/general"
MM_DIR      = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP/multimodel_mean"
MC_DIR      = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP/mc_models"
EXTREME_DIR = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/MULTIMODEL_MEAN"
FIG_DIR     = "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/NUMERICAL_MODEL/COMPLEX_MODEL"
os.makedirs(FIG_DIR, exist_ok=True)

deltaT_list    = np.arange(-0.6, 7.1, 0.1)
BASELINE_START = 1980
BASELINE_END   = 2009
START_YEAR     = 1950

# TauP masking parameters
TAUP_MIN    = 0.15   # per-model: mask cells where baseline TauP < this
SLOPE_CLIP  = 50.0   # clip |sl_tp| > this (%/K) as physically unreasonable
TAUP_MIN_MM = 0.05   # multimodel-mean baseline TauP threshold (section 2)

# Common 1°×1° grid
common_lat = np.arange(-89.5, 90.0, 1.0)
common_lon = np.arange(0.5, 360.0, 1.0)


def open_latlon(path):
    """Open a NetCDF file and rename latitude/longitude to lat/lon if needed."""
    ds    = xr.open_dataset(path)
    lat_n = "lat" if "lat" in ds.coords else "latitude"
    lon_n = "lon" if "lon" in ds.coords else "longitude"
    if lat_n != "lat":
        ds = ds.rename({lat_n: "lat", lon_n: "lon"})
    return ds


# ══════════════════════════════════════════════════════════════════════════
# 1. Load per-model gridded rates and build multimodel mean maps
# ══════════════════════════════════════════════════════════════════════════
ds_tmp = xr.open_dataset(os.path.join(MM_DIR, "multimodel_fwet_rel_global.nc"))
MODELS = [k.replace("fwet_rel_global_", "") for k in ds_tmp.data_vars
          if k.startswith("fwet_rel_global_")
          and k not in ("fwet_rel_global_mean", "fwet_rel_global_std",
                        "fwet_rel_global_min", "fwet_rel_global_max")]
ds_tmp.close()
print(f"Models: {len(MODELS)}")

sl_pl_stack, sl_pw_stack, sl_fw_stack, sl_tp_stack = [], [], [], []
models_loaded = []

for model in MODELS:
    f = os.path.join(GEN_DIR, model, f"{model}_gridded_rates.nc")
    if not os.path.exists(f):
        print(f"  ⚠  {model}: _gridded_rates.nc not found — skipping")
        continue

    ds = open_latlon(f)
    if "sl_tp" not in ds.data_vars:
        print(f"  ⚠  {model}: sl_tp missing — run compute_gridded_rates_taup.py first")
        ds.close()
        continue

    ds_i = ds.interp(lat=common_lat, lon=common_lon, method="linear")

    # Per-model TauP baseline mask
    ds_ref    = open_latlon(os.path.join(GEN_DIR, model, f"{model}_TauP_ref_1980_2009.nc"))
    ref_i     = ds_ref.interp(lat=common_lat, lon=common_lon, method="linear")
    taup_base = ref_i["TauP_mean"].values.astype(float)
    ds_ref.close()

    reliable_m = taup_base >= TAUP_MIN

    # Convert to %/K, apply mask, clip outliers
    sl_tp_pct = ds_i["sl_tp"].values * 100
    sl_tp_pct = np.where(reliable_m, sl_tp_pct, np.nan)
    sl_tp_pct = np.where(np.abs(sl_tp_pct) <= SLOPE_CLIP, sl_tp_pct, np.nan)

    sl_pl_stack.append(ds_i["sl_pl"].values)    # fraction/K
    sl_pw_stack.append(ds_i["sl_pw"].values)    # fraction/K
    sl_fw_stack.append(ds_i["sl_fw"].values)    # fraction/K
    sl_tp_stack.append(sl_tp_pct)               # %/K, masked
    models_loaded.append(model)
    ds.close()

sl_pl_stack = np.array(sl_pl_stack)   # (n_models, 180, 360) — fraction/K
sl_pw_stack = np.array(sl_pw_stack)
sl_fw_stack = np.array(sl_fw_stack)
sl_tp_stack = np.array(sl_tp_stack)   # %/K already

lat_ref = common_lat
lon_ref = common_lon

# Multimodel mean and std (%/K)
sl_pl_mm  = np.nanmean(sl_pl_stack, axis=0) * 100
sl_pw_mm  = np.nanmean(sl_pw_stack, axis=0) * 100
sl_fw_mm  = np.nanmean(sl_fw_stack, axis=0) * 100
sl_tp_mm  = np.nanmean(sl_tp_stack, axis=0)

sl_pl_std = np.nanstd(sl_pl_stack, axis=0) * 100
sl_pw_std = np.nanstd(sl_pw_stack, axis=0) * 100
sl_fw_std = np.nanstd(sl_fw_stack, axis=0) * 100
sl_tp_std = np.nanstd(sl_tp_stack, axis=0)

print(f"\nLoaded {len(models_loaded)} / {len(MODELS)} models onto {sl_pl_mm.shape} grid")
for name, arr in [("sl_pl", sl_pl_mm), ("sl_pw", sl_pw_mm), ("sl_fw", sl_fw_mm), ("sl_tp", sl_tp_mm)]:
    print(f"  {name}: {np.nanmean(arr):+.3f} %/K  [{np.nanmin(arr):.1f}, {np.nanmax(arr):.1f}]")
print(f"\n  TauP masking: TAUP_MIN={TAUP_MIN}  SLOPE_CLIP=±{SLOPE_CLIP} %/K")

# ══════════════════════════════════════════════════════════════════════════
# 2. Mask unreliable TauP cells (multimodel-mean baseline TauP)
# ══════════════════════════════════════════════════════════════════════════
taup_base_stack = []
for model in models_loaded:
    f = os.path.join(GEN_DIR, model, f"{model}_TauP_ref_1980_2009.nc")
    if not os.path.exists(f):
        continue
    ds   = open_latlon(f)
    ds_i = ds.interp(lat=common_lat, lon=common_lon, method="linear")
    taup_base_stack.append(ds_i["TauP_mean"].values.astype(float))
    ds.close()

taup_base_mm = np.nanmean(np.array(taup_base_stack), axis=0)

reliable = taup_base_mm >= TAUP_MIN_MM
sl_tp_mm_masked = np.where(reliable, sl_tp_mm, np.nan)
sl_tp_stack_masked = np.where(reliable[np.newaxis, :, :], sl_tp_stack * 100, np.nan)

print(f"TauP_base range: [{np.nanmin(taup_base_mm):.3f}, {np.nanmax(taup_base_mm):.3f}]")
print(f"Reliable cells:  {reliable.sum()} / {reliable.size}  ({100 * reliable.mean():.1f}%)")
print(f"sl_tp masked range: [{np.nanmin(sl_tp_mm_masked):.1f}, "
      f"{np.nanmax(sl_tp_mm_masked):.1f}] %/K")

# ══════════════════════════════════════════════════════════════════════════
# Shared plotting helpers
# ══════════════════════════════════════════════════════════════════════════
LON2D, LAT2D = np.meshgrid(lon_ref, lat_ref)
data = ccrs.PlateCarree()

# Yellow (negative) → white (zero) → green (positive)
cmap_YlGn_div = LinearSegmentedColormap.from_list(
    "YlGn_div",
    [(0.00, "#FFA500"),
     (0.25, "#FFFF66"),
     (0.50, "#FFFFFF"),
     (0.75, "#66CC66"),
     (1.00, "#004d00")],
    N=256)


def sign_agreement_mask(stack, threshold=0.80):
    """True where ≥ threshold of models agree on the sign."""
    n_total  = np.sum(np.isfinite(stack), axis=0).astype(float)
    n_pos    = np.sum(stack > 0, axis=0).astype(float)
    frac_pos = np.where(n_total > 0, n_pos / n_total, np.nan)
    return (frac_pos >= threshold) | (frac_pos <= (1 - threshold))


step    = 4
lat_idx = np.arange(0, len(lat_ref), step)
lon_idx = np.arange(0, len(lon_ref), step)
LON_sub = LON2D[np.ix_(lat_idx, lon_idx)]
LAT_sub = LAT2D[np.ix_(lat_idx, lon_idx)]


def plot_mean_with_stippling(ax, mean_map, stack, label, cb_fs, tick_fs):
    """MM mean map (symmetric yellow–green scale) + 80% sign-agreement stippling."""
    finite = mean_map[np.isfinite(mean_map)]
    vmax   = np.nanpercentile(np.abs(finite), 98)
    norm   = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.pcolormesh(LON2D, LAT2D, mean_map, transform=data,
                       cmap=cmap_YlGn_div, norm=norm, shading="auto", rasterized=True)

    agree_sub = sign_agreement_mask(stack, threshold=0.80)[np.ix_(lat_idx, lon_idx)]
    ax.scatter(LON_sub[agree_sub], LAT_sub[agree_sub], transform=data,
               s=1.4, c="k", marker=".", linewidths=0, alpha=0.9, zorder=5)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.1)
    ax.set_global()

    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.88, aspect=35)
    cb.set_label(label, fontsize=cb_fs)
    cb.ax.tick_params(labelsize=tick_fs)


# ══════════════════════════════════════════════════════════════════════════
# 3a. 2×2 maps: P_L, P_wet, f_wet, TauP
# ══════════════════════════════════════════════════════════════════════════
proj = ccrs.Robinson(central_longitude=180)
variables = [
    (sl_pl_mm, sl_pl_stack * 100, r"$\delta P_L / P_L$  (%/K)"),
    (sl_pw_mm, sl_pw_stack * 100, r"$\delta \bar{P}_{wet} / \bar{P}_{wet}$  (%/K)"),
    (sl_fw_mm, sl_fw_stack * 100, r"$\delta f_{wet} / f_{wet}$  (%/K)"),
    (sl_tp_mm, sl_tp_stack,       r"$\delta \tau_P / \tau_P$  (%/K)"),
]

fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw={"projection": proj})
for ax, (mean_map, stack, label) in zip(axes.flatten(), variables):
    plot_mean_with_stippling(ax, mean_map, stack, label, cb_fs=25, tick_fs=20)
    ax.set_title(label, fontsize=25, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG_DIR, "MM_gridded_rates_maps_2x2.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════
# 3b. 1×3 maps: P_L, P_wet, f_wet (panels a–c)
# ══════════════════════════════════════════════════════════════════════════
variables = [
    (sl_pl_mm, sl_pl_stack * 100, r"$\delta P_L / P_L$  (%/°C)"),
    (sl_pw_mm, sl_pw_stack * 100, r"$\delta \bar{P}_{wet} / \bar{P}_{wet}$  (%/°C)"),
    (sl_fw_mm, sl_fw_stack * 100, r"$\delta f_{wet} / f_{wet}$  (%/°C)"),
]
panel_labels = ["a", "b", "c"]

fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": proj})
for i, (ax, (mean_map, stack, label)) in enumerate(zip(axes.flatten(), variables)):
    plot_mean_with_stippling(ax, mean_map, stack, label, cb_fs=28, tick_fs=28)
    ax.set_title(panel_labels[i], fontsize=35, fontweight="bold", loc="left")

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(FIG_DIR, "MM_gridded_rates_maps.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ══════════════════════════════════════════════════════════════════════════
# 3c. 3×2 maps: MM mean and MM std
# ══════════════════════════════════════════════════════════════════════════
proj = ccrs.Robinson()
variables = [
    (sl_pl_mm, sl_pl_std, r"$\delta P_L / P_L$  (%/K)"),
    (sl_pw_mm, sl_pw_std, r"$\delta \bar{P}_{wet} / \bar{P}_{wet}$  (%/K)"),
    (sl_fw_mm, sl_fw_std, r"$\delta f_{wet} / f_{wet}$  (%/K)"),
]

fig, axes = plt.subplots(3, 2, figsize=(16, 11), subplot_kw={"projection": proj})
fig.suptitle("CMIP6 Multimodel Mean  |  Per-cell warming rates\n"
             "(annual OLS vs annual ΔT,  1950–2100)", fontsize=14, fontweight="bold")

for row, (mean_map, std_map, label) in enumerate(variables):
    for col, (data_map, title_sfx) in enumerate([(mean_map, "MM mean"), (std_map, "MM std")]):
        ax = axes[row, col]
        finite = data_map[np.isfinite(data_map)]
        if col == 0:
            vmax = np.nanpercentile(np.abs(finite), 98)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cmap = "RdBu_r"
        else:
            norm = plt.Normalize(vmin=0, vmax=np.nanpercentile(finite, 98))
            cmap = "YlOrRd"

        im = ax.pcolormesh(LON2D, LAT2D, data_map, transform=data, cmap=cmap, norm=norm,
                           shading="auto", rasterized=True)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.1)
        ax.set_global()

        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85, aspect=30)
        cb.set_label(f"{label}  ({title_sfx})", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(f"{label}  —  {title_sfx}", fontsize=10, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(FIG_DIR, "MM_gridded_rates_maps_mean_std.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
