# Task: build and persist neighbouring-MSOA (spatial lag) variables for the MSOA11 × year panel

## Context (read first)

- Project root: the `working/` repository. Read `README.md`, `DATA_PROCESSING_LOG.md` (especially sections J and the
  2023 new-build limitation), `config/config.yaml`, `outputs/eda/EDA_REPORT.md` §6 and `outputs/models/MODEL_LAB_REPORT.md` §5
  before writing code. Follow the existing conventions: numbered scripts in `src/`, a log in `outputs/logs/`, QA tables in
  `outputs/qa/`, tests in `tests/`, settings in `config/config.yaml`, sources in `config/source_manifest.csv`.
- Panel: `data/processed/final_msoa_year_dissertation_panel.parquet` (81,492 rows = 6,791 MSOA11 × 2012–2023, balanced).
  Baseline table: `data/processed/msoa_baseline_characteristics.parquet`. Key: `msoa11cd`, `year`.
- Boundaries: `data/raw/geography/msoa11_boundaries/MSOA11_BGC_England.geojson` (6,791 features, field `MSOA11CD`, WGS84).
- **Reuse, do not rebuild, the existing neighbour structure.** `src/eda/eda_common.py::queen_weights()` already builds queen
  contiguity on these boundaries and attaches islands to their nearest neighbour (1 island: Isles of Scilly; mean 5.68
  neighbours, min 1, median 5, max 22). `src/models/m05_spillovers.py` already builds a row-standardised mean of neighbours'
  `newbuilds_lag1_per_1000` **in memory only**. This task persists a fuller, documented set of neighbour variables so they can
  be used as controls in any later regression (including from R).
- Environment: Python, packages in `requirements.txt` (geopandas, libpysal, esda, pandas<3, duckdb, pyarrow, pytest).
  Do not upgrade pandas to 3.x (DuckDB 1.2 incompatibility; see log J.1).

## Deliverables

1. `src/21_build_spatial_lags.py` (or the next free number), runnable on its own, reading only processed data and boundaries.
2. `data/interim/spatial/msoa11_neighbours.parquet` — long neighbour table (definition below).
3. `data/interim/spatial/msoa_year_spatial_lags.parquet` — one row per `msoa11cd × year` with the new variables.
4. The new variables merged into the final panel (new columns only; no existing column changed), the Excel export
   (`src/20_export_excel.py`) and the `variable_dictionary` sheet updated with a row per new variable.
5. `tests/test_spatial_lags.py`.
6. QA tables in `outputs/qa/spatial/` and a new section **K. Spatial lag variables** appended to `DATA_PROCESSING_LOG.md`.

## Step 1 — Neighbour definitions

Build and save all three; the first is the main one.

| `nb_def` | Definition | Notes |
|---|---|---|
| `queen1` | First-order queen contiguity (via `queen_weights()`), islands attached to nearest neighbour | Main definition, consistent with EDA 06 and M05 |
| `queen2` | Second-order ring: neighbours of neighbours, **excluding** first-order neighbours and the MSOA itself | Robustness: wider market |
| `knn6` | 6 nearest neighbours by distance between centroids in EPSG:27700 | Robustness: same number of neighbours for every MSOA, unaffected by polygon shape and coastline clipping |

Use population-weighted centroids (ONS "MSOA (Dec 2011) Population Weighted Centroids") if they can be downloaded and
recorded in `source_manifest.csv`; otherwise use geometric centroids from the BGC polygons and state that in the log.

Neighbour table columns: `msoa11cd`, `neighbour_msoa11cd`, `nb_def`, `order` (1 or 2), `distance_km` (centroid to centroid),
`same_lad` (bool), `island_attached` (bool). Neighbours are **not** restricted to the same LAD or region: housing markets cross
administrative boundaries.

## Step 2 — Spatial lag variables (per MSOA-year, for each `nb_def`)

Always **exclude the MSOA itself**. Prefix every column with `w_<nb_def>_`, e.g. `w_queen1_nb_lag1_per_1000`.

### New-build supply (main controls)
- `nb_lag1_per_1000`: household-weighted rate = 1000 × Σ neighbours' `newbuild_total` in t-1 ÷ Σ neighbours'
  `baseline_households_2011`. **Not** the mean of neighbours' rates (that over-weights small MSOAs). Also save
  `nb_lag1_per_1000_meanrate` (row-standardised mean of rates) so results can be compared with M05.
- `nb_prev3yr_per_1000`: as above using t-3 to t-1.
- `nb_total_lag1`: Σ neighbours' `newbuild_total` in t-1 (count).
- Build the t-1 values from `newbuild_total` by year, not by re-using the own-MSOA lag columns, and check they match those
  columns for the own MSOA.

### Prices (secondary controls / robustness)
- `lp_median`: median across neighbours of `log_median_ppsqm` in year t.
- `lp_median_lag1`: the same in t-1.
- `lp_salesw`: sales-weighted mean of neighbours' `log_median_ppsqm` in t (weights = `sale_count`).
- `dlp_median_lag1`: median across neighbours of the annual change in `log_median_ppsqm` between t-2 and t-1.
- `dlp_median`: the same change between t-1 and t.

### Support and quality
- `n_neighbours`, `nb_households` (Σ neighbours' baseline households), `nb_sales` (Σ neighbours' `sale_count` in t),
  `share_neighbours_other_lad`.
- Flags: `flag_few_neighbours` (n < 2), `flag_thin_nb_sales` (`nb_sales` < 30), `flag_island_attached`,
  `flag_nb_2023_incomplete` (true whenever a new-build input year is 2023, because 2023 completions are under-counted).
- Missing inputs (e.g. no year t-1 for 2012): leave the lag as missing; do not fill with zeros.

## Step 3 — Checks (fail loudly)

Hard stops:
- `queen1` is symmetric; no self-neighbours; every MSOA has ≥ 1 neighbour after island attachment; 6,791 MSOAs in, 6,791 out.
- `queen2` contains no first-order neighbour and no self.
- Output is unique on `msoa11cd × year × nb_def` and has exactly 81,492 rows per `nb_def`.
- Spatial lags reproduce a hand calculation for at least 5 randomly chosen MSOA-years (seeded), for every variable family.
- The row-standardised mean of `newbuilds_lag1_per_1000` reproduces M05's `nb_mean_all` exactly.

QA tables (`outputs/qa/spatial/`): distribution of neighbour counts by definition; correlation of each own variable with its
spatial lag (pooled and within-MSOA, for comparison with EDA 06: pooled 0.22, within 0.10 for new-build intensity); the share
of MSOA-years with each flag; and a map of `n_neighbours` for `queen1`.

## Step 4 — Tests (`tests/test_spatial_lags.py`)

Use a toy 3×3 grid of square polygons, where the queen neighbours are known by hand. Test the neighbour sets for all three
definitions, self-exclusion, the household-weighted rate against a hand calculation, the median and sales-weighted price lags,
lag timing (t-1 uses year t-1), and that missing inputs give missing outputs, not zeros.

## Step 5 — Documentation

In `DATA_PROCESSING_LOG.md` §K and `variable_dictionary` record, for each variable: definition, neighbour definition, timing,
weighting, and the following note on how to use them.

## Econometric notes to copy into the log (do not change the variable construction because of them)

- **Preferred controls are neighbours' lagged new-build supply** (a "spatially lagged X"; Gibbons and Overman, 2012). It
  separates the effect of building in the MSOA from building next door. Own and neighbour construction are positively
  correlated (EDA 06), so omitting it attributes neighbouring effects to the own MSOA.
- **Neighbours' contemporaneous prices (`lp_median`, `lp_salesw`, `dlp_median`) are outcomes, not exogenous controls.** They
  respond to the same shocks as the own MSOA's price, and to the own MSOA's new building (spillovers). Including them as
  regressors runs into the reflection problem (Manski, 1993) and is a "bad control" (Angrist and Pischke, 2009, §3.2.3). They
  are built for description and for clearly labelled robustness checks only. If a price control is wanted, use the lagged
  versions (`lp_median_lag1`, `dlp_median_lag1`), and say why.
- With MSOA fixed effects, a time-invariant neighbour characteristic is absorbed; only the time-varying neighbour variables
  above add information. With LAD × year fixed effects (M6), neighbour variation that is common within a LAD-year is removed;
  report how much variation is left (as M05 does).

## Out of scope

Do not re-estimate the model lab, change existing variables, or alter the estimation sample. Report the run time, row counts,
check results and any decisions taken (e.g. which centroids were used) at the end.

### References
- Angrist, J. D. and Pischke, J.-S. (2009) *Mostly Harmless Econometrics*. Princeton University Press.
- Gibbons, S. and Overman, H. G. (2012) "Mostly pointless spatial econometrics?", *Journal of Regional Science*, 52(2), 172–191.
- Manski, C. F. (1993) "Identification of endogenous social effects: the reflection problem", *Review of Economic Studies*, 60(3), 531–542.
