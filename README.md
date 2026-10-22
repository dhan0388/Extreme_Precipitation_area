# Global extreme precipitation area expands beyond Clausius–Clapeyron scaling

Code accompanying:

> Mudiar, D. & Martinez-Villalobos, C. Global extreme precipitation area expands beyond Clausius–Clapeyron scaling (submitted).

The scripts compute the global area experiencing locally extreme daily precipitation (above the historical 1980–2009 local percentile), its scaling with global warming in CMIP6 models and ERA5, and the Gamma-distribution framework (spatially explicit and homogeneous models, elasticity decomposition) used to explain it.

## Scripts

| File | Purpose |
|---|---|
| `compute_empirical_area.py` | Daily extreme-precipitation area A(t) from CMIP6 and ERA5 at the P99 threshold |
| `compute_empirical_multithresh.py` | Extreme-precipitation area for thresholds from P95 to P99.8 |
| `Construction_of_homogeneous_models.py` | Spatially explicit and homogeneous Gamma models, and elasticity coefficients |
| `censored_vs_uncensored_comparison.py` | Sensitivity of the Gamma fit to the wet-day threshold (censored vs uncensored) |
| `Figure_1.py` | Fig. 1: metric definition and CMIP6–ERA5 evolution of extreme area |
| `Figure_2.py` | Fig. 2: spatial scaling of distribution parameters and spatially explicit Gamma model |
| `Figure_3.py` | Fig. 3: homogeneous Gamma reduction |
| `Figure_4.py` | Fig. 4: elasticity decomposition and threshold dependence |

## Suggested order

1. `compute_empirical_area.py` and `compute_empirical_multithresh.py`
2. `Construction_of_homogeneous_models.py`
3. `censored_vs_uncensored_comparison.py`
4. `Figure_1.py` to `Figure_4.py`

## Data

The input data are not included in this repository. They are publicly available from:

- **CMIP6** daily precipitation and near-surface temperature (historical and SSP5-8.5): Earth System Grid Federation, https://esgf.llnl.gov
- **ERA5** reanalysis: Copernicus Climate Change Service Climate Data Store, https://doi.org/10.24381/cds.adbb2d47
- **GPCP, PERSIANN-CDR and REGEN-LONG** daily precipitation at 1° resolution: FROGs database (Roca et al., 2019)

All fields are regridded to a common 1° × 1° grid. Set the data paths at the top of each script before running.

## Requirements

Python 3, with:

- numpy
- scipy
- xarray
- pandas
- matplotlib
- cartopy

## Licence

MIT (see `LICENSE`).

## Contact

Dipjyoti Mudiar — d.mudiar@gmail.com
