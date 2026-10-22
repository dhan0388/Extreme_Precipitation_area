#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# compute_empirical_area.py
#
# Empirical CMIP6 extreme-precipitation area for every model in cfg.MODELS
# (same model inventory and input paths as pipeline_method2.py).
#
# Method — identical to GFDL_concatanet.ipynb (original tas_vs_extreme_precip):
#   • P99(i,j): 99th percentile of ALL days (dry days included), 1980–2009,
#     xarray quantile (linear interpolation)
#   • daily extreme area = Σ cos(lat)·[pr > P99] / Σ cos(lat)·[valid cell]
#     (valid cells = non-NaN and ≥ 0 on the first time step, fixed denominator)
#   • annual value = mean of the daily fraction over each calendar year
# ΔT: annual global-mean TAS anomaly from the Method 2 pipeline
#     (general_method2/<MODEL>/<MODEL>_tas_dT_annual.nc), so the empirical rate
#     uses exactly the same ΔT as the parameter rates.
#
# Output (per model, in OUT_DIR):
#   tas_vs_extreme_precip_<MODEL>.nc   time(year) : fraction_extreme_area, tas_anomaly
#   P99_ref_1980_2009_<MODEL>.nc       P99 map (lat, lon)
#
# Usage:  sbatch run_empirical_area.sh [MODEL ...]   (no args = all models)
#         Models whose output already exists are skipped.
# ══════════════════════════════════════════════════════════════════════════════
import os
import sys
import traceback
import numpy as np
import xarray as xr

sys.path.insert(0, "/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP")
import config as cfg
from utils import setup_cluster, get_lat_lon_names, load_pr

GEN_DIR = os.path.join("/nfs_home/dmuriar/PRECIP_AREA/NUMERICAL/PL_and_TauP",
                       "general_method2")
OUT_DIR = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2"

BASELINE_START = cfg.BASELINE_START      # 1980
BASELINE_END   = cfg.BASELINE_END        # 2009
QUANTILE       = 0.99


def out_file(model):
    return os.path.join(OUT_DIR, f"tas_vs_extreme_precip_{model}.nc")


def p99_file(model):
    return os.path.join(OUT_DIR, f"P99_ref_1980_2009_{model}.nc")


def run_model(model, pr_dir):
    print(f"\n{'═' * 62}\n  MODEL: {model}\n{'═' * 62}", flush=True)

    if os.path.exists(out_file(model)):
        print(f"  ⏭  {model}: output exists — skipping.", flush=True)
        return

    dT_file = os.path.join(GEN_DIR, model, f"{model}_tas_dT_annual.nc")
    if not os.path.exists(dT_file):
        print(f"  ❌  {model}: no ΔT file ({dT_file}) — run pipeline_method2 first.",
              flush=True)
        return

    # ── Load daily precipitation (mm/day, same loader as the pipeline) ────────
    ds = load_pr(pr_dir, cfg.YEAR_START, cfg.YEAR_END)
    lat, lon = get_lat_lon_names(ds)
    pr = ds["pr"].transpose("time", lat, lon)
    yrs = pr["time"].dt.year
    print(f"  pr: {int(yrs[0])}–{int(yrs[-1])} | "
          f"{pr.sizes[lat]} lat × {pr.sizes[lon]} lon", flush=True)

    # ── 1. P99 threshold, all days, 1980–2009 ────────────────────────────────
    base = pr.sel(time=(yrs >= BASELINE_START) & (yrs <= BASELINE_END))
    n_base_yrs = len(np.unique(base["time"].dt.year))
    if n_base_yrs < (BASELINE_END - BASELINE_START + 1):
        print(f"  ⚠  {model}: only {n_base_yrs} baseline years available", flush=True)
    base = base.chunk({"time": -1, lat: max(1, pr.sizes[lat] // 16), lon: -1})
    p99 = base.quantile(QUANTILE, dim="time").drop_vars("quantile").compute()
    print(f"  P99 range: {float(p99.min()):.2f} – {float(p99.max()):.2f} mm/day",
          flush=True)

    # ── 2. Daily area fraction above P99 (cos-lat weights, fixed denominator) ─
    w = np.cos(np.deg2rad(pr[lat]))
    valid0 = (pr.isel(time=0).notnull() & (pr.isel(time=0) >= 0)).astype(float)
    denom = float((valid0.sum(lon) * w).sum(lat).compute())

    exceed = (pr > p99).astype("float32")
    frac_daily = (exceed.sum(lon) * w).sum(lat) / denom

    # ── 3. Annual mean ───────────────────────────────────────────────────────
    frac_annual = frac_daily.groupby("time.year").mean("time").compute()

    # ── 4. ΔT from the Method 2 pipeline, aligned on year ────────────────────
    dT = xr.open_dataset(dT_file)["dT_annual"].load()
    fa, dT = xr.align(frac_annual, dT, join="inner")
    years = fa["year"].values
    bl = (years >= BASELINE_START) & (years <= BASELINE_END)
    print(f"  years: {years[0]}–{years[-1]} ({len(years)}) | "
          f"baseline A0 = {float(fa.values[bl].mean()):.5f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    ds_out = xr.Dataset(
        {"fraction_extreme_area": (["time"], fa.values.astype(float)),
         "tas_anomaly":           (["time"], dT.values.astype(float))},
        coords={"time": years},
    )
    ds_out["time"].attrs = {"long_name": "year"}
    ds_out["fraction_extreme_area"].attrs = {
        "long_name": "Annual mean fraction of area with daily pr > local P99",
        "units": "1",
        "threshold": f"all-day P{QUANTILE*100:g}, {BASELINE_START}-{BASELINE_END}",
        "weighting": "cos(lat); fixed denominator of valid cells",
    }
    ds_out["tas_anomaly"].attrs = {
        "long_name": "Global-mean annual TAS anomaly (Method 2 pipeline)",
        "units": "K", "baseline": f"{BASELINE_START}-{BASELINE_END}",
        "source_file": dT_file,
    }
    ds_out.attrs = {"model": model, "pr_dir": pr_dir,
                    "script": "compute_empirical_area.py"}
    tmp = out_file(model) + ".tmp"
    ds_out.to_netcdf(tmp)
    os.replace(tmp, out_file(model))       # only complete files carry the final name

    xr.Dataset({"P99": p99}).assign_attrs(
        model=model, units="mm/day",
        description=f"All-day P99 of daily pr, {BASELINE_START}-{BASELINE_END}",
    ).to_netcdf(p99_file(model))

    print(f"  ✓ {out_file(model)}", flush=True)


if __name__ == "__main__":
    selected = sys.argv[1:] or list(cfg.MODELS)
    unknown = [m for m in selected if m not in cfg.MODELS]
    if unknown:
        sys.exit(f"Not in cfg.MODELS: {unknown}")

    client, cluster = setup_cluster(cfg.N_WORKERS, cfg.THREADS_PER_WORKER,
                                    cfg.MEMORY_LIMIT)
    try:
        for m in selected:
            try:
                run_model(m, cfg.MODELS[m]["pr_dir"])
            except Exception as e:
                print(f"\n  ❌  {m} FAILED: {e}", flush=True)
                traceback.print_exc()
    finally:
        client.close()
        cluster.close()

    done = sum(os.path.exists(out_file(m)) for m in selected)
    print(f"\nComplete: {done} / {len(selected)}  →  {OUT_DIR}", flush=True)
