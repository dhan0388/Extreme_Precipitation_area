#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# compute_empirical_multithresh.py
#
# Multi-threshold version of compute_empirical_area.py (Figure 4b). Same model
# inventory (cfg.MODELS), same method, same ΔT — only the threshold changes:
#   • P_q(i,j): q-th percentile of ALL days (dry days included), 1980–2009,
#     q ∈ {95, 98, 99, 99.5, 99.8}, xarray quantile (linear interpolation)
#   • daily extreme area = Σ cos(lat)·[pr > P_q] / Σ cos(lat)·[valid cell]
#     (valid cells = non-NaN and ≥ 0 on the first time step, fixed denominator)
#   • annual value = mean of the daily fraction over each calendar year
#   • dA_rel_<slug> = (A - A0) / A0 × 100 (%), A0 = 1980–2009 mean
#     (percent, as in the earlier MultiThreshold files used by Figure 4)
# ΔT: annual global-mean TAS anomaly from the Method 2 pipeline
#     (general_method2/<MODEL>/<MODEL>_tas_dT_annual.nc).
# The P99 series is identical to compute_empirical_area.py (consistency check).
#
# Output (per model): OUT_DIR/tas_vs_extreme_precip_multithresh_<MODEL>.nc
#   time(year): fraction_<slug>, dA_rel_<slug>, tas_anomaly
#
# Usage:  sbatch run_empirical_multithresh.sh [MODEL ...]   (no args = all models)
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
OUT_DIR = "/nfs_home/dmuriar/PRECIP_AREA/CIMP6/EMPIRICAL_AREA_method2/multithresh"

BASELINE_START = cfg.BASELINE_START      # 1980
BASELINE_END   = cfg.BASELINE_END        # 2009
PERCENTILES    = [95.0, 98.0, 99.0, 99.5, 99.8]
SLUGS          = ["p95", "p98", "p99", "p995", "p998"]


def out_file(model):
    return os.path.join(OUT_DIR, f"tas_vs_extreme_precip_multithresh_{model}.nc")


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

    # ── 1. Percentile thresholds, all days, 1980–2009 ────────────────────────
    base = pr.sel(time=(yrs >= BASELINE_START) & (yrs <= BASELINE_END))
    n_base_yrs = len(np.unique(base["time"].dt.year))
    if n_base_yrs < (BASELINE_END - BASELINE_START + 1):
        print(f"  ⚠  {model}: only {n_base_yrs} baseline years available", flush=True)
    base = base.chunk({"time": -1, lat: max(1, pr.sizes[lat] // 16), lon: -1})
    thr_all = base.quantile([p / 100.0 for p in PERCENTILES], dim="time").compute()

    # ── 2. Fixed denominator (cos-lat weights, valid cells) ──────────────────
    w = np.cos(np.deg2rad(pr[lat]))
    valid0 = (pr.isel(time=0).notnull() & (pr.isel(time=0) >= 0)).astype(float)
    denom = float((valid0.sum(lon) * w).sum(lat).compute())

    dT = xr.open_dataset(dT_file)["dT_annual"].load()

    out_vars, years = {}, None
    for p, slug in zip(PERCENTILES, SLUGS):
        thr = thr_all.sel(quantile=p / 100.0).drop_vars("quantile")
        exceed = (pr > thr).astype("float32")
        frac_daily  = (exceed.sum(lon) * w).sum(lat) / denom
        frac_annual = frac_daily.groupby("time.year").mean("time").compute()

        fa, dTa = xr.align(frac_annual, dT, join="inner")
        yy = fa["year"].values
        bl = (yy >= BASELINE_START) & (yy <= BASELINE_END)
        A0 = float(fa.values[bl].mean())
        out_vars[f"fraction_{slug}"] = (["time"], fa.values.astype(float))
        out_vars[f"dA_rel_{slug}"]   = (["time"], (fa.values - A0) / A0 * 100)   # %
        years, dT_vals = yy, dTa.values
        print(f"  {slug:<5} A0 = {A0:.5f}  (target {1 - p/100:.4f})", flush=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_vars["tas_anomaly"] = (["time"], dT_vals.astype(float))
    ds_out = xr.Dataset(out_vars, coords={"time": years})
    ds_out["time"].attrs = {"long_name": "year"}
    ds_out["tas_anomaly"].attrs = {
        "long_name": "Global-mean annual TAS anomaly (Method 2 pipeline)",
        "units": "K", "baseline": f"{BASELINE_START}-{BASELINE_END}",
        "source_file": dT_file,
    }
    ds_out.attrs = {"model": model, "pr_dir": pr_dir,
                    "percentiles": str(PERCENTILES),
                    "threshold": f"all-day percentiles, {BASELINE_START}-{BASELINE_END}",
                    "weighting": "cos(lat); fixed denominator of valid cells",
                    "script": "compute_empirical_multithresh.py"}
    tmp = out_file(model) + ".tmp"
    ds_out.to_netcdf(tmp)
    os.replace(tmp, out_file(model))

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
