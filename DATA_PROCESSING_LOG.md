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
   (economic activity).
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
6. All z-scores and quartiles use the England-only distribution. Income z,
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
   - `final_msoa_year_dissertation_panel.parquet` + `.csv` (81,492 × 87; was 78),
   - `msoa_baseline_characteristics.parquet` + `.csv` (6,791 × 49; was 27),
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
