# Spatial spillover exposure: methodology

Implemented in `src/13b_prepare_spatial_spillovers.py` (functions in `src/spillover_methods.py`, queen edge table in `src/eda/eda_common.py`), merged by
`src/14_merge_final_panel.py`, explored in `src/eda/eda_10_spillover_exposure.py` and used by `src/models/m05_spillovers.py`.
No new data were downloaded: every input is an existing intermediate or raw file listed in `config/source_manifest.csv`.

## 1. Economic motivation
New housing in one MSOA may be associated with prices in nearby MSOAs, because neighbouring areas can be linked through residential substitution, shared amenities,
congestion, transport accessibility and changes in neighbourhood composition. An own-MSOA construction variable therefore need not capture the full local housing-market
association of new development. The spatial exposure variables describe new supply in the surrounding areas so that its association with a focal MSOA's prices can be
examined, conditional on own construction and the fixed effects. They are descriptive exposure measures; they do not by themselves identify causal spillovers.

## 2. Data and reference years
| Input | File | Reference |
|---|---|---|
| MSOA11 boundaries (BGC, generalised clipped) | `data/raw/geography/msoa11_boundaries/MSOA11_BGC_England.geojson` | 2011 geography |
| MSOA11 population-weighted centroids | `data/raw/geography/msoa11_pwc/MSOA11_PWC_England_BNG.geojson` | 2011 geography, 2011 Census population, British National Grid (EPSG:27700; the file carries BNG coordinates without a CRS tag, so the CRS is declared, not converted) |
| New-build counts and lags (EPC new-dwelling records) | `data/interim/geography_crosswalks/msoa_year_newbuild_panel.parquet` | 2012-2023 plus lookback 2009-2011 already used by the upstream lags |
| Baseline households | same file (`baseline_households_2011`) | Census 2011 (fixed denominator) |
| Baseline income | `msoa11_baseline_income.parquet` (`baseline_income_bhc_2011_12`) | FYE 2012 |
| Prices | `msoa_year_price_panel_trimmed.parquet` | annual, nominal |
| LAD of each MSOA | `msoa11_geography_reference.parquet` (`lad23cd_analysis`) | fixed 2024 LAD vintage |

Notes on timing: IMD 2015 (the deprivation moderator) is built from data mostly dated 2012-2013 and published in 2015, so it is not strictly predetermined for every year of the panel;
income (FYE 2012) and the 2011 Census characteristics are. Distances use fixed 2011 centroids, so exposure definitions do not change over time.
Lookback quality: the 2011 lookback year matches MHCLG's official new-build completions (ratio 1.00), 2009 and 2010 are 6% and 11% below (`outputs/qa/spatial_spillovers/lookback_year_coverage.csv`).
This affects only the older lags (lag2 and lag3, and the three-year total) for 2012-2013.

## 3. Neighbour networks
Directed edges are stored with one row per focal-neighbour pair; the focal MSOA is never its own neighbour.

**A. Queen contiguity (primary).** MSOAs are neighbours if their boundaries share an edge or point (`msoa11_spatial_neighbors_queen.parquet/.csv`; columns `focal_msoa11cd`, `neighbour_msoa11cd`, `edge_type`, `same_lad`, `distance_m`).
38,594 directed edges, 5.68 neighbours on average (1-22). Genuine queen edges (38,592) are reciprocal. The Isles of Scilly has no contiguous neighbour and is attached to its nearest neighbour (a Cornwall MSOA),
exactly as in the EDA and model lab (`queen_attach_islands`); those two directed edges are labelled `island_fallback` and are not genuine contiguity. `same_lad` compares `lad23cd_analysis`;
8,444 directed edges cross LAD borders and 3,390 MSOAs have no cross-LAD neighbour. `distance_m` is the centroid-to-centroid distance.

**B. Centroid distance (robustness).** MSOAs whose population-weighted centroids are within 5 km or 10 km (`msoa11_spatial_neighbors_distance.parquet`, found with a KD-tree, not a full pairwise matrix).
This is centroid-based proximity, not the distance between the nearest points on MSOA boundaries. 136,950 directed pairs within 5 km (20.2 neighbours on average) and 451,886 within 10 km (66.5).
480 MSOAs have no centroid within 5 km and 41 have none within 10 km; their distance-based exposures are undefined (`NaN`, with flags `nbr_5km_none`, `nbr_10km_none`), never zero.
These MSOAs are rural and remote (mean density 144 persons/km2 against 3,322 overall), so dropping them changes the sample composition.

**C. Distance decay (robustness).** For pairs within 10 km, w_ij = exp(-d_ij / 3000) (d in metres). The 3 km parameter is a proposed modelling choice, not an estimated or literature-derived value, and is configurable
(`spatial_spillovers.distance_decay_m`). Unnormalised weights and row-normalised weights are both stored.

**Optional second-order queen neighbours** (neighbours of neighbours, excluding the focal MSOA and first-order neighbours) are implemented and tested but disabled by default (`compute_second_order: false`).

## 4. Exposure variables
Let N(i) be the neighbours of MSOA i, c_jt completions in j in year t, h_j baseline (2011) households, and c_j,t-k lagged completions (from the upstream lookback, so 2012 is defined).

* Pooled neighbouring rate (primary): `rate_it = 1000 * sum_{j in N(i)} c_jt / sum_{j in N(i)} h_j`.
  Lagged version `nbr_queen_nb_lag1_rate_per1000` uses c_{j,t-1} (primary regression variable); `nbr_queen_nb_prev3yr_rate_per1000` uses c_{j,t-1} + c_{j,t-2} + c_{j,t-3}
  (a cumulative three-year total, not an annual average); totals are `nbr_queen_nb_total`, `nbr_queen_nb_lag1_total`.
* Preliminary measure `nbr_queen_nb_mean_rate_lag1`: the arithmetic mean over j in N(i) of `newbuilds_lag1_per_1000`. It reproduces the row-standardised queen lag used by the earlier model-lab regressions to 4e-14.
  It differs from the pooled rate whenever neighbours differ in size, because it weights each neighbour equally.
* Cross-LAD: `nbr_queen_crosslad_nb_lag1_rate_per1000` uses only neighbours in a different LAD; `NaN` (flag `nbr_queen_crosslad_none`, count `nbr_queen_crosslad_n`) where there are none.
* Distance: `nbr_5km_nb_total`, `nbr_5km_nb_rate_per1000`, `nbr_5km_nb_lag1_rate_per1000`, `nbr_10km_nb_lag1_rate_per1000` (same pooled definition with a different neighbour set; `nbr_5km_n`, `nbr_10km_n` counts).
* Distance-weighted: `nbr_exp10km_nb_lag1_rate_per1000 = 1000 * sum_j w_ij c_{j,t-1} / sum_j w_ij h_j` (a weighted construction rate, not a count of homes within 10 km); the weight sum is `nbr_exp10km_weight_sum`.
* Neighbouring baseline income (time invariant, copied to every year): `nbr_queen_income_bhc2012_hhmean = sum_j h_j y_j / sum_j h_j` (household-weighted mean of the income LEVEL),
  `nbr_queen_log_income_hhmean = ln(that mean)` (the log of the weighted mean, not the weighted mean of logs) and `income_gap_own_minus_nbr_queen_log = ln(y_i) - ln(neighbour mean)`;
  positive = the focal MSOA is richer than its surroundings. 5 km versions are also stored (`nbr_5km_*`, `income_gap_own_minus_nbr_5km_log`).
* Neighbouring prices (descriptive): `nbr_queen_price_log_ppsqm_salewmean = sum_j s_jt ln(p_jt) / sum_j s_jt` over neighbours with a valid price and a positive sale count (a sale-count-weighted mean of MSOA log median prices,
  not a pooled transaction median), with `nbr_queen_price_n_valid`, `nbr_queen_price_total_sales` and a one-year lag. Prices are never summed and missing prices are never zero.
  Contemporaneous neighbouring prices are not used as a regressor.

## 5. Why the primary variable is the pooled rate
A raw sum of neighbouring new builds depends on how many neighbours an MSOA has and how large they are. The unweighted mean of neighbouring MSOA rates gives every neighbour equal weight regardless of its
number of households. The pooled rate measures additions relative to the combined baseline housing stock of the neighbouring area, so it is the proposed primary exposure; the unweighted mean is retained because it is the
measure used in the preliminary model-lab results. Both are in new builds per 1,000 households, the same units as own construction.

## 6. Spatial definition and scale
Queen contiguity is primary because it is transparent, administrative and matches the existing EDA. MSOAs differ greatly in area, so contiguity does not necessarily imply economic proximity;
5 km and 10 km radii and the 3 km decay are exploratory scales chosen to span adjacent-MSOA to city-wide neighbourhoods, not distances established by the literature. An MSOA-level exposure cannot identify
effects within a few hundred metres of individual construction sites.

## 7. Interpretation and identification
A spatial-lag-of-X specification is used (no spatial lag of the outcome):

`ln(price_it) = a_i + d_{l(i)t} + beta * own_lag1_it + theta * neighbour_lag1_it + e_it`

with a_i an MSOA effect and d_{l(i)t} a year effect (M3) or LAD-by-year effect (M6). theta is the association between neighbouring construction and the focal MSOA's price, conditional on own construction and the fixed effects.
For heterogeneity the neighbour term is interacted with predetermined focal baseline income (z-score); time-invariant levels are absorbed by the MSOA effects but their interactions with time-varying exposure are identified.
These are not causal estimates. Construction responds to local demand shocks, planning decisions and expectations; lagging exposure and LAD-by-year effects do not remove these concerns. With LAD-by-year effects,
the part of an exposure common to a LAD in a year is absorbed, and neighbour pools that overlap across MSOAs in the same LAD induce dependence between observations, so spatially robust inference
(for example Conley errors) is a sensible sensitivity check beyond LAD clustering.

## 8. Socioeconomic heterogeneity
Three distinct quantities are available: the baseline income of the focal MSOA experiencing the price change; the household-weighted baseline income of the surrounding MSOAs receiving the new housing; and their difference.
Start with focal-income interactions (consistent with the research question); treat source-neighbour income and the relative-income gap as supplementary analyses, assessing interpretability, collinearity and multiple testing
before adding interactions. Interactions are computed at regression time, not stored.

## 9. Quality assurance (all run in `13b`, saved in `outputs/qa/spatial_spillovers/`)
No self or duplicated edges; genuine queen edges reciprocal (island fallback reported separately); distances non-negative and within thresholds; neighbour counts equal to the EDA's;
exact reproduction of the preliminary neighbour-mean measure (4e-14); 280 manual recomputations on 40 random MSOA-years agree with the stored values (pooled rates, lags, three-year total, mean, price and income);
lagged exposures equal the preceding year's neighbouring construction; focal construction excluded; 2012 lagged exposures defined through the lookback; no negative or infinite rates; static variables constant within MSOA;
panel rows and keys unchanged (81,492; no duplicates) and no existing column changed (`panel_integrity_vs_previous_build.csv`). Unit tests with synthetic graphs: `tests/test_spatial_spillovers.py`.

## 10. Limitations
* Completions are measured from EPC new-dwelling records. Many recent EPCs lack a UPRN (11% in 2022, 52% in 2023), so exposures that use 2022-2023 construction are understated; nothing is imputed.
* Distance exposures are undefined for 480 (5 km) and 41 (10 km) mostly rural MSOAs.
* MAUP and edge effects: MSOA boundaries are administrative; coastal and national-border MSOAs have fewer neighbours.
* The island fallback is an artificial connection (2 edges).
* Neighbour household denominators are 2011 values; housing stock growth is not reflected.
* IMD 2015 is not strictly predetermined for the whole panel.

## 11. References
* Halleck Vega, S. and Elhorst, J. P. (2015). "The SLX Model." *Journal of Regional Science*, 55(3), 339-363. https://doi.org/10.1111/jors.12188 - motivates relating the outcome to own-area and weighted neighbouring explanatory variables (SLX), the design used here.
* Gonzalez-Pampillon, N. (2022). "Spillover effects from new housing supply." *Regional Science and Urban Economics*, 92, 103759. https://doi.org/10.1016/j.regsciurbeco.2021.103759 - motivates exposure to surrounding residential developments and distance decay; the paper uses project-level investment and a policy-based identification strategy, which this MSOA-level design does not replicate.
* Asquith, B. J., Mast, E. and Reed, D. (2023). "Local Effects of Large New Apartment Buildings in Low-Income Areas." *Review of Economics and Statistics*, 105(2), 359-375. https://doi.org/10.1162/rest_a_01055 - motivates highly local effects of new supply; its microgeographic design is much finer than an MSOA adjacency network.
* Gibbons, S. and Overman, H. G. (2012). "Mostly pointless spatial econometrics?" *Journal of Regional Science*, 52(2), 172-191. https://doi.org/10.1111/j.1467-9787.2012.00760.x - cautions that spatial lag and error models are weakly identified; supports using SLX with exogenous exposure variables and not putting neighbouring outcomes on the right-hand side.

The four references and their journal details were checked in Crossref; what each paper actually does should be confirmed from the papers themselves. The decisions about radii, decay parameter and variable construction are specific to this dissertation and are not taken from the literature.
