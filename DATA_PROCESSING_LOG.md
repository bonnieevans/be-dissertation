# Data processing log: England MSOA11 × year panel

What was done to the data, in order, with sources. All numbers are from the
actual pipeline runs (logs in `outputs/logs/`, QA tables in `outputs/qa/`).
Raw files are never modified; everything derived lives in `data/interim/` and
`data/processed/`. Every external file has a URL, download date, size and
SHA-256 checksum in `config/source_manifest.csv`. All thresholds and
year/category rules live in `config/config.yaml`.

**Revision history.**
- 2026-09: first full version (MSOA11 panel, sections A-H).
- 2026-10-02: **IMD 2015 revised** (sections A.6, E.2, F.3, G, H.4 and the new
  section I). The overall IMD includes a housing-affordability domain, so it
  was replaced as the primary deprivation moderator; rank-averaging and the
  old bottom-20 share were removed from estimation. Task specification:
  `docs/imd_revision_brief.md`; results: `outputs/qa/imd_revision/IMD_REVISION_REPORT.md`.

## A. Sources

1. **UCL ReShare 857911, bundle 1** (open access):
   `epc_2024.csv`, the domestic EPC register with address fields removed by
   UCL. 26,979,573 rows, 87 columns, to 31 Oct 2024. Used to identify new-build
   dwellings.
2. **UCL ReShare 857911, bundle 2**: `tranall_link_26122024.csv`, the linked
   HMLR Price Paid–EPC transactions. 22,449,587 rows, 106 columns, 1995 to
   31 Oct 2024. Also contains the original Land Registry PPD (not used for
   outcomes) and `NSPL21_NOV_2024` (postcode lookup, 2021 geography).
3. **ONS NSUL, Feb 2024**: UPRN → postcode, MSOA21, current LAD/region,
   coordinates. (Scotland files skipped.)
4. **ONS NSPL "2011 Census" edition, Feb 2024**: postcode → MSOA11 / LSOA11.
5. **ONS Small Area Income Estimates, FYE 2012 (2011/12)**: net weekly
   equivalised household income before housing costs, MSOA11, with confidence
   limits.
6. **English Indices of Deprivation 2015** (gov.uk): **File 7** (LSOA11 domain
   scores, ranks, deciles, overall IMD, mid-2012 total and working-age
   population; 32,844 LSOAs); **File 9** (official exponentially transformed
   domain scores, used only as an independent check); and the **Technical
   Report** (method: Section 3.6-3.7, Appendix F).
7. **ONS LSOA11 → MSOA11 → LAD11 lookup** (ArcGIS FeatureServer, paginated;
   34,753 distinct LSOA11s).
8. **Census 2011 via Nomis API** (geography type 297 = MSOA11): QS405EW
   (tenure), QS102EW (population/area), QS501EW (qualifications), QS601EW
   (economic activity). **KS102EW** (age structure) was supplied as a Nomis
   export (`data/raw/census2011/agestructure_ks102ew.xlsx`; 16 age bands plus total).
   **Accessibility (added 2026-10-08):** DfT *Journey Time Statistics: access to
   services 2014* LSOA11 tables JTS0501-0508; MHCLG *English Town Centres 2004*
   polygons (WFS); ONS MSOA11 population-weighted centroids.
9. **ONS CPIH index** (series L522, 2015=100), monthly.
10. **ONS LAD names and codes, Dec 2024**: labels only.
11. **ONS MSOA21 super-generalised boundaries (England)**: map figures only.
    The MSOA11↔MSOA21 lookup is now used only for MSOA11 names.
12. Downloaded but **not used** in the final pipeline: ONSUD Feb 2024, Census
    2021 TS054 tenure. Both are marked as unused in the manifest.

## B. Housing transactions (the outcome)

1. **Extraction.** The nested zips were unpacked to CSV (two ~23 GB files).
   Everything was processed with DuckDB reading from disk. Nothing large was
   loaded into pandas.
2. **Schema check.** All columns the brief required were confirmed present
   (transaction id/date/price, postcode, property type, old/new, duration,
   floor area, `priceper`, `classt`, `LMK_KEY`, `UPRN`, and for the EPC file
   `TRANSACTION_TYPE`, `TENURE`, etc.). Schemas were saved to `outputs/qa/`.
3. **Period.** Kept transactions dated 2012-01-01 to 2023-12-31. The Jan–Oct
   2024 partial year was dropped.
4. **Established homes only.** Kept `oldnew = 'N'` so new-build sale prices
   never enter the outcome. This took the sample from 9,774,641 to 8,672,278
   rows.
5. **Standard transactions.** HMLR category-type and record-status filters were
   checked. Every row was already category A, status A, so the filter removed
   nothing.
6. **Validity.** Dropped non-positive prices and missing or non-positive floor
   areas, leaving 8,668,693 rows (3,585 had missing or invalid floor area,
   0.04%).
7. **Price per m².** `ppsqm = price / EPC total floor area`;
   `log_ppsqm = ln(ppsqm)`. I did not use postcode-mean / postcode-mean.
8. **Check against UCL.** My `ppsqm` equals UCL's `priceper` for 100% of rows
   (differences are floating-point rounding only, sd ≈ 4e-12).
9. **Real prices.** `real_ppsqm = ppsqm / (CPIH/100)` using the month of sale.
   All months matched. This is descriptive only and is not used in the panel
   outcome.
10. **Outliers.** Within each calendar year, trimmed `ppsqm` below the 0.5th and
    above the 99.5th percentile. This removed 86,658 rows, leaving 8,582,035
    in the E&W trimmed sample. An untrimmed copy is kept for robustness.
11. **Retained fields.** Id, date, year, postcode, price, floor area, ppsqm,
    log ppsqm, property type, old/new, duration, EPC construction age band,
    UCL match-quality `classt`.

## C. New-build construction (the treatment)

1. **Identify new builds** from the EPC file only (not the linked
   transactions, which would select on later resale). Criterion:
   `TRANSACTION_TYPE = 'new dwelling'`. This gave 2,998,461 EPC records.
2. **UPRN is the identifier.** 307,012 new-dwelling EPCs (10.2%) have no UPRN
   and cannot be located, so they are excluded.
3. **One event per dwelling.** Per UPRN, took the earliest lodgement date as
   the new-build event (`newbuild_date`, year, quarter, floor area,
   property type). 84,666 later duplicate EPCs were collapsed, leaving
   2,606,783 unique new-build UPRNs.
4. **Tenure retained for QA only.** Mapped the raw tenure text to
   SOCIAL / PRIVATE_NONSOCIAL / UNKNOWN (both "rental" and "rented" wordings
   handled; UNKNOWN never treated as private). It is not used in the treatment
   or moderators. (Reason: 70–100% of new-build EPCs per year are UNKNOWN. See
   section H.)

## D. Geography

1. **Primary key is MSOA11.** Income, IMD2015 and the Census 2011 tables are
   natively 2011 geography, so MSOA11 is the key and MSOA21 is a reference
   column.
2. **No boundary crosswalk used.** A MSOA21→MSOA11 crosswalk is ambiguous for
   ~19 merged MSOA21 areas. Instead both geographies come from direct postcode
   lookups:
   - transactions: postcode → MSOA11 (NSPL 2011 edition) and → MSOA21 /
     current LAD / region (NSPL21);
   - new builds: UPRN → postcode (NSUL) → MSOA11 and MSOA21 by the same
     postcode lookups.
3. **England only.** Filtered on `msoa11cd LIKE 'E02%'`. Wales dropped: 8,143,571
   trimmed England transactions; 2,473,242 England new builds.
4. **Fixed LAD.** `lad23cd_analysis` is the current (Nov 2024) LAD from NSPL21,
   named from the Dec 2024 LAD list. It is deliberately independent of the MSOA
   vintage, ready for LAD × year work.
5. **Match rates.**
   - Transactions: 99.9999% (MSOA11), 100% (MSOA21).
   - New-build UPRN→MSOA21: 99.96%.
   - New-build UPRN→MSOA11: 98.6% overall, ≥99.5% in every year to 2022, but
     **85.1% in 2023 and 55.3% in 2024**. The newest postcodes have not been
     given a 2011-geography best fit.

## E. Baseline characteristics (all fixed, one value per MSOA11)

All were computed natively at MSOA11, so there is no crosswalk and 6,791
England MSOA11s with 0% missing.

1. **Income.** Source is weekly, so I annualised ×52 (`baseline_income_bhc_2011_12`).
   Confidence limits are kept and also ×52. Then `log_baseline_income`,
   `income_z` (England-only, unweighted), `income_quartile`, `low_income_q1`,
   `below_median_income`. ONS rounds this series to £10/week (83 distinct
   values), so many ties are genuine. The QA check was set to catch a failed
   merge (one value dominating >20% of rows), not rounding.
2. **Deprivation (IMD 2015), revised.** Why: the overall IMD includes the
   Barriers to Housing and Services domain, which contains a housing
   affordability indicator (house prices relative to incomes). The outcome is
   house prices, so the overall IMD is not the primary moderator. Averaging
   LSOA ranks is invalid, so all rank-averaging was removed
   (`imd2015_rank_percentile_pw` no longer exists anywhere).
   1. **Load and assert.** File 7's real header is printed and mapped in
      `config.yaml` (the working-age header has a trailing space, stripped on
      load). Asserted: 32,844 unique England LSOA11s; each LSOA11 nests in
      exactly one MSOA11 (existing LSOA11→MSOA11 lookup, no crosswalk); all
      6,791 MSOA11s covered and identical to the panel's MSOA11 universe.
   2. **LSOA-level recombination** (Technical Report method). Per domain,
      `R = (N − rank + 1)/N` with N = 32,844 and R = 1 most deprived;
      `X = −23·ln(1 − R·(1 − exp(−100/23)))`; then the weighted sum of X with
      the published weights (Income .225, Employment .225, Education .135,
      Health .135, Crime .093, Barriers .093, Living Environment .093).
      The published weights sum to 0.999 and the published IMD uses them as
      published, so they are **rescaled to sum to 1 only for the reduced
      variants**.
   3. **Validation (hard stop).** Recombining all 7 domains reproduces the
      published IMD score: Spearman 1.000000, maximum absolute difference
      **0.0025** index points (mean 0.00025). The transform also reproduces
      the official File 9 transformed scores to within rounding (max 0.011).
      Unit test: `tests/test_imd_population_weighting.py`.
   4. **Rank source: a flagged deviation.** The brief says to rank each domain
      score. File 7's scores are rounded to 3 decimals, so re-ranking them
      creates large tie blocks (for Income only 1.7% of re-ranked ranks equal
      the published ranks; max transform error 0.46 vs File 9). The published
      ranks, which MHCLG computed from unrounded scores, are therefore the
      default (`imd2015.rank_source: published`). Re-ranking is available
      (`score`) and also passes the test (Spearman 0.999998, max difference
      0.164).
   5. **Three variants.** `imd_ex_housing` (6 domains, drops Barriers; the
      proposed primary), `imd_ex_housing_living` (5 domains, drops Barriers and
      Living Environment; sensitivity), `imd_ex_income_housing` (5 domains,
      drops Income and Barriers; for use with SAIE as the income moderator).
      Domain lists and weights are in `config.yaml`.
   6. **Aggregation to MSOA11:** population-weighted means of LSOA *values*,
      never ranks or deciles. Income domain rate weighted by total population
      (this gives the exact MSOA rate; `imd2015_income_rate_msoa`); Employment
      rate by working-age population (`imd2015_employment_rate_msoa`); the other
      domain scores, the three indices and the overall score by total
      population. The overall measure is renamed `imd2015_overall_score_pw`
      and labelled robustness only (contains housing affordability).
      QA: weights sum to each MSOA's population, an independent SQL
      recomputation agrees to 1e-14, and no values are missing.
   7. **Bottom-20 share.** Recomputed from national deciles of `imd_ex_housing`
      at LSOA level (`imd_ex_housing_bottom20_popshare`), as a secondary
      descriptive variable only: it is zero for 61.7% of MSOAs by
      construction. The old overall-IMD share is kept as
      `imd2015_overall_bottom20_popshare_legacy`, flagged NOT FOR ESTIMATION.
   8. **Moderator switches** (`config.yaml`, `moderators:`):
      `deprivation_moderator` ∈ {`imd_ex_housing` (default),
      `imd_ex_housing_living`, `imd_ex_income_housing`, `imd_income_rate`};
      `income_moderator` ∈ {`saie` (default), `imd_income_rate`}.
      `deprivation_z`, `deprivation_quartile` and `high_deprivation_q4` are
      rebuilt (England-only, unweighted, 4 = most deprived) from the chosen
      deprivation moderator in `14_merge_final_panel.py`. Choosing
      `imd_income_rate` for both is refused (hard stop). A warning is logged if
      the income domain is double-counted (e.g. `imd_income_rate` as income
      with `imd_ex_housing`). **Sign convention:** SAIE income higher = richer;
      every IMD measure higher = more deprived, so under `imd_income_rate`
      `income_z` flips sign (the `low_income_q1` flag keeps its "poor" meaning).
3. **Tenure (QS405EW).** `baseline_households_2011` (total), owner-occupied,
   shared ownership, social rented (council + other social), private rented;
   shares; `social_rent_z` and `social_rent_quartile`.
4. **Density (QS102EW).** Population and hectares → `population_2011`,
   `area_km2`, density per km², `log_population_density_2011`, `density_z`,
   `density_quartile`. This is an urban-form moderator, not strictly
   socioeconomic.
5. **Optional.** `degree_share_2011` = Level 4+ / residents 16+;
   `unemployment_rate_2011` = unemployed / economically active, each with a z
   score and quartile.
6. **Age structure (KS102EW, added 2026-10-08).** The 16 Nomis age bands are grouped
   into five non-overlapping groups (under 16, 16-24, 25-44, 45-64, 65 and over)
   and expressed as shares of all usual residents
   (`age_share_*_2011`) with England-only z-scores (`age_share_*_z`). Hard checks:
   6,791 unique England MSOA11s; the bands sum to the total in every MSOA11; the
   total equals the QS102EW population exactly (0 differences); the five shares
   sum to 1. The five shares are collinear by construction, so at most four can
   enter a regression together. No quartile columns were created for age.
   Script: `src/11b_prepare_census2011_age.py`.
7. **Accessibility (added 2026-10-08), baseline 2014 / 2004.** Script
   `src/11c_prepare_accessibility.py`; diagnostics `src/15c_accessibility_diagnostics.py`
   and `outputs/qa/accessibility/ACCESSIBILITY_REPORT.md`. Higher = less accessible
   throughout.
   1. *DfT times.* The eight LSOA11 tables give modelled minutes to the nearest
      employment centre (500-4,999 jobs, DfT's key-services definition), primary
      school, secondary school, further education, GP, hospital, food store and
      town centre, by public transport/walk (`..._pt_...`) and car (`..._car_...`);
      AM peak (Tuesday in October, 7-10am for public transport); capped at 120
      minutes (under 0.5% of LSOAs reach the cap, only for hospitals). Real column
      headers were printed and mapped in `config.yaml`. Quirks handled: the food
      store table (`jts0507`) carries "Town" in its column headers (a DfT labelling
      error; identified by its title and its values differ from the town-centre
      table); `jts0502` has one exact duplicate LSOA row (dropped, asserted
      identical); 2 and 5 LSOAs have no children aged 5-10 / 11-15 and get zero weight.
   2. *Aggregation.* Population-weighted means of LSOA minutes (never ranks), weights =
      each table's own service-user population (economically active 16-74 for
      employment, ages 5-10 / 11-15 / 16-19 for schools and FE, households for GP,
      hospital, food stores, town centres). LSOA11 nests in MSOA11, so no crosswalk.
      Asserted: 32,844 LSOA11s in every table, each in exactly one of 6,791 MSOA11s,
      no missing values, times within 0-120, independent recomputation agrees.
   3. *Composite.* `access_keyservices_{pt,car}_min_2014` = equal-weighted mean of
      the eight MSOA-level service times (DfT's key-services average, which DfT
      publishes only above LSOA level). England working-age-weighted average from
      these tables: 16.9 min (public transport/walk) and 10.6 min (car), matching
      DfT's published 2014 figures of about 17 and 10.
   4. *Distance.* `dist_town_centre_km` = straight-line km from the MSOA11
      population-weighted centroid to the nearest 2004 town-centre polygon (0 if
      inside), British National Grid. 1,232 polygons were downloaded; DfT cites
      1,211 town centres (not reconciled). `dist_town_centre_centroid_km` (to the
      polygon centroid) is a sensitivity variable (correlation 0.998).
   5. *Transforms and what is kept.* z-scores and logs only for the three headline measures
      (the two composites and the distance). From 2026-10-09 the final panel and baseline
      table carry ONLY the composites, the distance and their z-scores/logs (9 columns); the
      16 individual service times and the centroid-distance sensitivity variable stay in the interim table
      `msoa11_accessibility2014.parquet` and are used by the diagnostics only.
8. All z-scores and quartiles use the England-only distribution. Income z,
   quartile and the two income flags now follow the `income_moderator` switch
   (default SAIE, unchanged values).

## F. Building the panel

1. **Price panel.** Established, trimmed transactions grouped by MSOA11 × year:
   `sale_count`, median price, median and mean ppsqm, mean log ppsqm, p25/p75,
   median floor area, shares by property type (D/S/T/F) and leasehold,
   `log_median_ppsqm` (primary outcome), and `sales_under_10/20/30/50` flags.
   Low-sale cells are flagged and not dropped. 81,492 MSOA-years.
2. **New-build panel.** Counts by MSOA11 × year (total, houses, flats, median and
   mean floor area) on a full grid, so MSOA-years with no building are explicit
   zeros. `newbuilds_per_1000 = 1000 × newbuild_total / baseline_households_2011`
   (Census 2011 households, not 2021). Lags: `newbuilds_lag1/2/3(_per_1000)`
   and `newbuilds_prev3yr(_per_1000)` = t-1 + t-2 + t-3. Primary treatment is
   `newbuilds_lag1_per_1000`. 21.5% of MSOA-years have zero construction.
   Reproducibility of the per-1000 figure from raw counts is checked.
3. **Merge.** Joined the price, new-build, geography and six baseline tables
   onto `msoa11cd × year`, adding `main_sample` (2016–2023),
   `baseline_complete` and `analysis_complete_case`. Interaction terms are
   **not stored**; build them at regression time as
   `newbuilds_lag1_per_1000 × z-score`.
4. **Hard QA stops** (pipeline fails, not just logs): duplicate keys, any
   non-England row, `baseline_households_2011 ≤ 0`, any baseline value varying
   across years within an MSOA. All passed.
5. **Outputs** in `data/processed/`:
   - `final_msoa_year_dissertation_panel.parquet` + `.csv` (81,492 × 109; was 97 before accessibility and green belt, 87 before age, 78 before the IMD revision),
   - `msoa_baseline_characteristics.parquet` + `.csv` (6,791 × 71; was 59 before accessibility and green belt, 49 before age, 27 before the IMD revision),
   - `transactions_regression_ready.parquet` (8.1M rows, parquet only),
   - `transactions_analysis.parquet` and `newbuilds_analysis.parquet` (fuller
     detail).

## G. Diagnostics, tables, figures, Excel

1. Correlation matrix of the six z-scores, and VIF for the joint model
   (treatment + four interactions, computed in memory). Default build
   (SAIE + `imd_ex_housing`): largest VIF 4.54 (the deprivation interaction),
   so none exceeds 10. Nothing was dropped for correlation. The selected
   deprivation z-score correlates −0.66 with income and 0.93 with the optional
   unemployment z-score (the index contains the Employment domain), so the
   optional unemployment moderator largely duplicates it.
   `15b_imd_revision_diagnostics.py` adds the IMD revision diagnostics
   (section I).
2. QA summaries (`data_quality_summary`, `geography_match_summary`,
   `transaction_count_summary`) and Tables 1, 3, 4 (summary statistics;
   construction by income and deprivation quartile).
3. Eleven figures/data files, including England maps of cumulative
   construction and baseline deprivation.
4. `final_dissertation_data.xlsx` (aggregates only; no transaction rows): panel,
   baseline characteristics, summary statistics, correlations, QA summary,
   `imd_revision` (all IMD diagnostics and the scatter), variable dictionary,
   methodology notes.
5. 36 unit tests pass (price calculation, geography joins, new-build
   deduplication, IMD transform, reproduction of the published IMD,
   population weighting and moderator switches, income QA).

## H. Things you should know before using the data

1. **Tenure is not usable as a treatment.** 70–100% of new-build EPCs per year
   have unknown tenure; 82% of those are never re-inspected. Investigation:
   `outputs/excel/unknown_tenure_investigation.xlsx`. This is why the question
   was reframed around neighbourhood characteristics.
2. **2023 new-build counts are likely understated** (85% MSOA11 match rate; EPC
   lodgement lag). This affects the contemporaneous 2023 `newbuild_total`, not
   the lagged treatment, which uses 2022 (99.6% matched).
3. **Dwellings without a UPRN (10%) are not counted anywhere.** New-build
   intensity is therefore a lower bound.
4. **The bottom-20 deprivation share is zero for 61.7% of MSOAs** by
   construction, so it is secondary/descriptive only. The moderator is now the
   continuous `imd_ex_housing` index. Quartile membership of the *previous*
   moderator (a quartile split of that zero-inflated share) was arbitrary
   within the zero block, so it is not comparable.
5. **Baseline households are Census 2011**, a fixed denominator for all years.
6. **Place-based policy controls (FHSF, Towns Fund, LUF, CRF, UKSPF) are not
   built yet.** The exclusions you specified are recorded as planned.

## I. IMD revision: results (2026-10-02)

Full tables: `outputs/qa/imd_revision/IMD_REVISION_REPORT.md` and the
workbook's `imd_revision` sheet.

1. **Reproduction:** passed (Spearman 1.000000; max absolute difference
   0.0025 index points).
2. **SAIE vs IMD income rate** (6,791 MSOA11s): Pearson −0.683 on log SAIE
   (−0.628 on raw SAIE), Spearman −0.703. They overlap but are not the same
   measure. SAIE is modelled household income; the IMD rate is the share of
   people on income-related benefits.
3. **Barriers domain** correlates +0.109 (Pearson) with SAIE income: richer
   areas score as more deprived on the housing-affordability domain, the
   contamination the revision removes.
4. **Overall IMD vs `imd_ex_housing`:** Spearman 0.990; **10.5% of MSOA11s
   change deprivation quartile**, none by two or more; 95.4% of the
   most-deprived quartile stays in the top quartile. For
   `imd_ex_income_housing` it is 12.4%.
5. **Joint-model VIFs** (max): 4.58 under SAIE + `imd_ex_housing`;
   3.62 under SAIE + `imd_ex_income_housing`. Both well below 10.
6. **Effect on Table 4:** mean `newbuilds_lag1_per_1000` now falls
   monotonically from 8.75 (least deprived quartile) to 7.07 (most deprived),
   and the share of MSOA-years with zero construction rises from 12% to 33%.
7. **Panel columns:** final panel 78 → 87 (+12 added, −3 removed: the three
   old IMD columns); baseline table 27 → 49 (+25, −3). All hard stops
   still pass (no duplicate keys, England only, `baseline_households_2011 > 0`,
   baseline values constant within MSOA across years, now including the new
   moderator columns).

## J. Exploratory analysis and baseline specification lab (2026-10-02)

These steps read the final panel and write only to `outputs/eda/`, `outputs/models/` and `data/interim/eda/`.
Full results: `outputs/eda/EDA_REPORT.md` and `outputs/models/MODEL_LAB_REPORT.md`.

1. **Environment.** `linearmodels`, `pyfixest`, `esda` and `libpysal` were installed. The install upgraded pandas to 3.0.6,
   which DuckDB 1.2.1 cannot read (new string dtype); pandas was returned to 2.3.3 and `requirements.txt` pins `pandas<3`.
   The final panel and baseline table were regenerated and are identical to the earlier versions.
2. **New source.** ONS MSOA (December 2011) BGC boundaries, England subset (6,791 areas), downloaded by
   `src/eda/00_download_msoa11_boundaries.py` and recorded in `config/source_manifest.csv`.
3. **Outcome definition.** `log_median_ppsqm` is built from the **nominal** price per m2 (script 12 uses `nominal_ppsqm`); the
   CPIH-deflated `real_ppsqm` exists in the transaction files but is not the panel outcome. Year effects absorb the common
   nominal trend.
4. **Exploratory analysis** (scripts `eda_01`-`eda_07`): balance and quality, within/between variance, distributions, sale
   counts, correlations at four levels, trends, PCA, stationarity, cross-sectional dependence, spatial autocorrelation.
5. **Panel tests were implemented from the published formulas and validated by simulation** (N = 2,000; T = 12 and 8):
   Harris-Tzavalis (with and without unit trends), Hadri (finite-T null moments by simulation) and Pesaran CD
   (`src/eda/panel_tests.py`; unit tests in `tests/test_panel_tests_and_models.py`).
6. **Baseline specification lab** (`m01`-`m08`): estimation sample = main sample 2016-2023 excluding LADs with a single MSOA
   (54,312 MSOA-years; 6,789 MSOAs; 294 LADs); standard errors clustered by LAD; a registry of all 192 regressions run.
7. **Not done, by decision.** The place-based policy registry (Phase 2) is not built, so the funding-control check is not run;
   `m07` documents the algebra for any LAD-year variable.
8. **Known data limitation at the end of the sample (corrected 2026-10-09).** New-build counts in the panel are incomplete in
   2022 and especially 2023 because new-dwelling EPCs increasingly lack a UPRN: missing-UPRN share 1.5-2.9% in 2012-2020, 4.3% in
   2021, 11.2% in 2022, 51.5% in 2023 and 71.7% in 2024. Against MHCLG Live Table 120 (net additional dwellings, new-build
   completions, converted from financial to calendar years) the panel is 0.88-1.12 of the official count in 2012-2022 and 0.38 in
   2023. The raw EPC new-dwelling records are NOT short (0.98 of official in 2023); the loss is the missing UPRN. This replaces the
   earlier statement that the shortfall was due to lodgement lag. Leads that reach 2023 are analysed in a separate sample (`m04`).
   Details: `outputs/qa/sample_choice/`.
9. **Notes written:** `docs/geography_justification.md` (draft, for editing) and `docs/iv_literature_note.md` (citations
   written from memory; to be verified).

## K. Accessibility: what the diagnostics show (2026-10-08)

Full tables: `outputs/qa/accessibility/ACCESSIBILITY_REPORT.md`.

1. **The composite is a reasonable summary, not a perfect one.** Each of the eight
   public-transport service times correlates 0.73-0.85 (Pearson) with the key-services
   composite. PC1 of the eight times explains 69% of the variance (public transport/walk)
   and 50% (car); PC1 correlates 0.98 / 0.95 with the composite.
2. **Public transport and car agree closely in ranking** (composite Spearman 0.95, Pearson
   0.88), but car times are much less spread out (mean 10.7 vs 17.2 minutes).
3. **Distance to town centre is related but distinct.** It correlates 0.70 (Pearson) and 0.64
   (Spearman) with the public-transport composite, 0.75 / 0.63 with the car composite, and
   0.85-0.89 with DfT's own travel time to the town centre.
4. **Strong overlap with density.** The composites correlate -0.82 (public transport) and -0.74
   (car) with log population density, and distance -0.63 (denser = more accessible). Overlap
   with income is about zero (0.02 to 0.08) and with deprivation -0.2 to -0.4. Collinearity
   with the density moderator should be checked before both enter a regression.
5. **Extreme values.** The Isles of Scilly (E02006781) is a large outlier (composite 85.7 min
   by public transport; z above 20 for raw variables). Distances and times are right-skewed
   (skew 2.3-5.2), so log versions are provided.
6. **Not included:** airport/rail-station connectivity (no confirmed 2015 tables), cycle times
   (in the DfT tables, not extracted), and a 2015 baseline (2014 is the earliest confirmed
   year; 2014 and 2015-2016 employment data are not strictly comparable).

## L. Persistence of the 2011 baseline characteristics (2026-10-09)

Script `src/eda/eda_08_baseline_persistence.py`; report `outputs/eda/persistence/PERSISTENCE_REPORT.md`. Raw 2021 Census tables
(Nomis TS007A, TS066, TS067, TS006 at MSOA21; ONS TS054 already on disk) are in `data/raw/census2021/` and are used only for this check.

1. **Method.** The 2011 baseline value of each moderator is compared with the same definition built from the 2021 Census, linked by
   the ONS best-fit MSOA11 to MSOA21 lookup. Sample A = MSOA11s whose code is unchanged and not merged (6,677 of 6,791; code
   retention does not guarantee identical boundaries); sample B = all 6,791 via best fit (32 MSOA11s sit in 16 merged MSOA21s; 81
   MSOA21s are not the best fit of any MSOA11). Quartiles are formed within each census year's own England distribution.
2. **Results (sample A).** Spearman 2011 vs 2021: density 0.997, social-rent share 0.990, degree share 0.966, age 65+ 0.936,
   age 25-44 0.919, unemployment 0.889, age 45-64 0.843, age 15-24 0.834 (Pearson 0.946), age 0-14 0.874. Same quartile in both
   censuses: 94% density, 91% social rent, 80% degree, 73% age 65+, 71% age 25-44, 66% unemployment, 64% age 0-14, 61% age 15-24
   and 45-64. Moves of two or more quartiles are at most 4.3%. Sample B gives almost identical figures.
3. **Levels moved even where ranks did not** (degree share 27.0% to 33.5%, unemployment 6.4% to 4.7%, age 65+ 16.7% to 19.1%); the model
   uses England-wide z-scores and quartiles, so rank stability is what matters for the interactions.
4. **Stability is lower in low-density areas for the age and unemployment variables** (for example age 15-24 Spearman 0.66 in the
   least dense quartile against 0.82 in the densest).
5. **Not covered:** income (SAIE FYE2012) and IMD 2015 have no 2021 equivalent in this check; accessibility is not a Census measure.
   The 2021 Census was taken during COVID-19 restrictions, and economic-activity and qualification definitions differ slightly between censuses.

## M. Accessibility: 2011 series compared with 2014 (2026-10-09)

Scripts `src/11d_prepare_accessibility2011.py` and `src/eda/eda_09_accessibility_2011_vs_2014.py`; report
`outputs/eda/accessibility/ACCESSIBILITY_2011_VS_2014.md`. The 2011 table (`msoa11_accessibility2011.parquet`) is NOT merged into the panel.

1. **Source.** DfT Accessibility Statistics, LSOA tables ACS0501-0508 for 2011 (older series, 2001 LSOA codes, whole-minute times,
   car times from 2010 use Trafficmaster speeds). The 2001 LSOAs (32,482) are moved to 2011 LSOAs with the ONS best-fit lookup (units
   unchanged 31,672 rows, merged 293, split 881, irregular 151) and then to MSOA11. Split or irregular LSOAs share the 2001 LSOA's weight
   equally among their pieces (2.4% of employment weight). Weights are the tables' own service-user counts (rounded to tens).
2. **Validation.** The 7-service England averages from these tables are 14.3 minutes (public transport/walk) and 6.0 (car), matching DfT's
   published 2011 headline (about 14 and 6). Employment centres are defined as LSOAs with at least 500 jobs (2014: 500-4,999).
3. **Public transport rankings agree reasonably across the two years.** Composite (8 services): Spearman 0.83, Pearson 0.85, 61% in the same
   quartile, 5% move two or more quartiles; individual services 0.77-0.86 except hospitals (0.59).
4. **Car rankings agree poorly,** composite Spearman 0.66 (Pearson 0.42), 50% same quartile, 12% move two or more. The 2011 car times have a
   5-minute floor: 91-98% of MSOAs sit within half a minute of it for employment, GP and primary schools, 70% of MSOAs have a composite below 6, and
   the series has only 227-926 distinct values per service against 6,244-6,791 for 2014. The 2011 car series therefore cannot rank most
   areas; Spearman among MSOAs above the floor is 0.58.
5. **Data artefact in 2011.** Tower Hamlets 025 (E02000888) has a 2011 car time of 120 minutes (the cap) for all eight services; it is not
   plausible and is left as published. Isles of Scilly is at the cap in 2011 as well.
6. **Both composites relate to density in the same direction** (public transport -0.72 in 2011, -0.82 in 2014); the 2011 car composite is
   much weaker (-0.24), consistent with the floor.

## N. Green belt share (added 2026-10-09)

Script `src/11e_prepare_greenbelt.py`; QA in `outputs/qa/greenbelt/`.

1. **Variable.** `greenbelt_share_2011` = area of the MSOA11 inside designated green belt / area of the MSOA11, as at 31 March 2011;
   `greenbelt_share_z` (England-only unweighted z-score) and `greenbelt_any_2011` (share >= 1%). Fixed baseline, one value per MSOA11.
2. **Source.** MHCLG England Green Belt polygons for 2010/11 (WFS layer `dclg_inspire:England_Green_Belt_2010_11_WGS84`, as at 31 March 2011),
   overlaid on the ONS MSOA11 generalised clipped (BGC) boundaries in British National Grid; polygons from different authorities dissolved (overlap
   0.0001 ha), invalid geometries repaired. 2011/12 and 2014/15 layers were also downloaded for stability checks.
3. **Checks.** Dissolved national area 1,633,068 ha against MHCLG's published 1,639,530 ha for 2010/11 (-0.4%; the 2011/12 layer is -0.2%);
   99.99% of the green belt area lies inside the MSOA11 boundaries. 2014/15 layer vs 2010/11: shares correlate 0.9996 (23 MSOAs differ by more than 0.05).
4. **Data error in the 2011/12 layer.** It shows Guildford (9 MSOAs) and Basildon (4) wholly as green belt (share about 1.0), which neither the 2010/11
   nor the 2014/15 layer supports; the 2010/11 layer used for the variable is unaffected.
5. **Distribution.** Mean 0.153; zero for 60.9% of MSOA11s; at least 1% for 34.4%; fully inside (99%+) for 0.4%; median among positive 0.34.
   Highest regional mean in the East Midlands (0.27) and the North West (0.23); lowest in the South West (0.07) and London (0.08).
6. **Overlap with other moderators is small:** correlations of -0.22 with log density, -0.22 with deprivation, -0.16 with social rent, +0.13 with income,
   and +0.07 / +0.08 with the public-transport accessibility composite / distance to town centre.
7. **Limits.** Generalised boundaries make shares approximate near edges; designation is a snapshot at March 2011 (the national designated area changed by
   about 0.3% to 2015); green belt status is a planning designation, not land cover.

