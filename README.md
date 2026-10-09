# England MSOA11-year panel: new-build construction and neighbourhood socioeconomic heterogeneity

Data pipeline for: *Does the local house-price effect of new residential
development differ according to the socioeconomic characteristics of the
receiving neighbourhood?* England, 2012-2023 (principal regression sample
2016-2023, flagged via `main_sample`). Heterogeneity by baseline income,
deprivation, social-rented share, and population density (education/
unemployment optional).

> **MSOA11 is the primary analysis geography** (not MSOA21) - see "Geography"
> below. Tenure of the new-build itself is **not** the treatment - see "Why
> tenure is not the treatment" below.

## Setup

```bash
pip install -r requirements.txt
```

Place the two UCL ReShare 857911 bundle zips (`857911_bundle_1.zip`,
`857911_bundle_2.zip`, from https://reshare.ukdataservice.ac.uk/857911/ -
open access, no login required) in `data/raw/` or the parent project's
`data/raw/`. Every other source is downloaded automatically by the relevant
script (see `config/source_manifest.csv` for URLs/checksums/dates).

## Running the pipeline

Scripts run in numeric order from `src/`, each writing its own log to
`outputs/logs/` and QA tables to `outputs/qa/`:

| Script | Does |
|---|---|
| `00_check_environment.py` | package/directory/disk checks |
| `01_inventory_ucl_files.py` | extracts the UCL bundles, prints/saves raw schemas |
| `02_clean_transactions.py` | established-home price sample, 2012-2023, CPIH deflation |
| `03_clean_epc.py` | cleans the deposited EPC source |
| `04_construct_newbuilds.py` | new-build universe (tenure retained for QA only) |
| `05_attach_geography.py` | postcode/UPRN -> MSOA11 (primary) + MSOA21/LAD/region (reference) |
| `06_prepare_income.py` | ONS baseline income, FYE2012, natively at MSOA11 |
| `07_prepare_imd2015.py` | IMD2015 deprivation: domain recombination excluding housing, validated against the published IMD, population-weighted LSOA->MSOA11 |
| `08_prepare_census2011_tenure.py` | Census 2011 QS405EW tenure + baseline_households_2011 |
| `09_prepare_census2011_density.py` | Census 2011 QS102EW population density |
| `10_prepare_census2011_education.py` | Census 2011 QS501EW degree share (optional) |
| `11_prepare_census2011_unemployment.py` | Census 2011 QS601EW unemployment rate (optional) |
| `11b_prepare_census2011_age.py` | Census 2011 KS102EW age structure: five age-group shares and z-scores |
| `11e_prepare_greenbelt.py` | share of each MSOA11 inside designated green belt as at 31 March 2011 (MHCLG polygons overlaid on MSOA11 boundaries) |
| `11c_prepare_accessibility.py` | DfT Journey Time Statistics 2014 (8 services x public transport/car), key-services composites, straight-line distance to the nearest 2004 town centre |
| `12_build_msoa_price_panel.py` | MSOA11 x year price outcomes |
| `13_build_msoa_newbuild_panel.py` | MSOA11 x year new-build counts + lags (tenure-agnostic) |
| `13b_prepare_spatial_spillovers.py` | spatial neighbour lookups (queen, 5/10 km, distance decay) and neighbouring-construction / income / price exposure variables (`outputs/qa/spatial_spillovers/`) |
| `14_merge_final_panel.py` | final panel + baseline table + transaction-level robustness file |
| `15_run_diagnostics.py` | correlation matrix + VIF (interactions computed on the fly) |
| `15c_accessibility_diagnostics.py` | accessibility representativeness: correlations, PCA, public transport vs car, overlap with moderators, maps (`outputs/qa/accessibility/`) |
| `15b_imd_revision_diagnostics.py` | IMD revision: Pearson/Spearman table, scatter, VIFs under two moderator pairings, quartile movement, `IMD_REVISION_REPORT.md` |
| `16_investigate_unknown_tenure.py` | standalone QA deep-dive (unaffected by Phase 2) |
| `17_quality_assurance.py` | data_quality/geography_match/transaction_count summaries |
| `18_summary_statistics.py` | descriptive Tables 1, 3, 4 |
| `19_make_figures.py` | the 11 figures/data files, including two England maps |
| `20_export_excel.py` | `outputs/excel/final_dissertation_data.xlsx` |

```bash
cd src
for f in 00_check_environment 01_inventory_ucl_files 02_clean_transactions \
         03_clean_epc 04_construct_newbuilds 05_attach_geography \
         06_prepare_income 07_prepare_imd2015 08_prepare_census2011_tenure \
         09_prepare_census2011_density 10_prepare_census2011_education \
         11_prepare_census2011_unemployment 11b_prepare_census2011_age 11c_prepare_accessibility 11e_prepare_greenbelt 12_build_msoa_price_panel \
         13_build_msoa_newbuild_panel 13b_prepare_spatial_spillovers 14_merge_final_panel 15_run_diagnostics \
         15b_imd_revision_diagnostics 15c_accessibility_diagnostics \
         17_quality_assurance 18_summary_statistics 19_make_figures 20_export_excel; do
    python3 "${f}.py" || break
done
```

Run tests with `pytest tests/` (74 tests: price calculation, geography
joins, new-build dedup, IMD transform / reproduction of the published IMD /
population weighting / moderator switches, income-duplication QA, panel unit-root /
cross-section-dependence tests, within transformation, wild-cluster bootstrap).

## Exploratory analysis and baseline specification lab

Two further layers sit on top of the final panel. Neither writes to `data/processed`.

**`src/eda/`: exploratory data analysis** (report: `outputs/eda/EDA_REPORT.md`)

| Script | Does |
|---|---|
| `eda/00_download_msoa11_boundaries.py` | downloads the England MSOA11 boundaries (ONS BGC) to `data/raw/geography/msoa11_boundaries/` |
| `eda/eda_01_data_structure.py` | balance, missingness, within/between variance, distributions, outliers, sale-count profile, low-sale areas |
| `eda/eda_02_correlations.py` | pooled / between / within / two-way-within correlations, moderator correlations, partial correlations, binned scatters |
| `eda/eda_03_trends.py` | national series, paths by moderator quartile and region, common-shock shares, autocorrelation, heatmaps, maps |
| `eda/eda_04_pca.py` | PCA of the moderators and IMD domains; PCA of price paths; Bai-Ng factor count |
| `eda/eda_05_stationarity_dependence.py` | Harris-Tzavalis, Hadri and Fisher-ADF stationarity tests (implemented and validated by simulation in `panel_tests.py`); Pesaran CD test |
| `eda/eda_06_spatial.py` | queen-contiguity neighbours, Moran's I by year, LISA maps |
| `eda/eda_07_report.py` | assembles `EDA_REPORT.md` from the tables |
| `11d_prepare_accessibility2011.py` | older DfT Accessibility Statistics 2011 (2001 LSOA codes) moved to MSOA11; comparison only, not merged |
| `eda/eda_09_accessibility_2011_vs_2014.py` | rank agreement between the 2011 and 2014 accessibility series |
| `eda/eda_10_spillover_exposure.py` | distributions, maps and correlations of the spillover exposure variables |
| `eda/eda_08_baseline_persistence.py` | persistence of the 2011 baseline characteristics against the 2021 Census (`outputs/eda/persistence/`) |

**`src/models/`: baseline specification lab** (report: `outputs/models/MODEL_LAB_REPORT.md`; every regression is logged in `outputs/models/tables/spec_registry_ALL.csv`)

| Script | Does |
|---|---|
| `models/m01_fe_ladder.py` | y on x with no controls, then pooled, year FE, MSOA FE, MSOA+year, LAD+year, MSOA+region x year, MSOA+LAD x year; FE F-tests, Mundlak, serial correlation, heteroskedasticity, residual dependence |
| `models/m02_sales_threshold_weights.py` | minimum-sales thresholds, weighting by sale count, sample composition |
| `models/m03_moderators.py` | moderator interactions: singles, pairs, triples, all; with/without moderator x year controls; PCA composites; VIFs; Holm adjustment |
| `models/m04_endogeneity_pretrends.py` | leads and lags, reverse direction, pre-period balance and event study |
| `models/m05_spillovers.py` | neighbouring-MSOA construction variable |
| `models/m06_clustering_inference.py` | clustering comparison and wild-cluster bootstrap |
| `models/m07_iv_feasibility.py` | mechanical check of which variation survives the fixed effects |
| `models/m08_report.py` | assembles `MODEL_LAB_REPORT.md` |

Run order: `cd src/eda && for f in 00_download_msoa11_boundaries eda_01_data_structure eda_02_correlations eda_03_trends eda_04_pca eda_05_stationarity_dependence eda_06_spatial eda_07_report; do python3 $f.py || break; done`, then the same in `src/models` for `m01 ... m08`. Interaction terms, leads, lags and neighbour variables are built in memory; none is stored in a permanent dataset. Notes: `docs/geography_justification.md` (draft), `docs/iv_literature_note.md`, `docs/spillover_exposure_methodology.md`.

## Geography: MSOA11 is primary, MSOA21 is a reference column

Income (FYE2012), IMD2015, and all four Census 2011 moderators are natively
2011 statistical geography, so **MSOA11CD is the panel's key**. Rather than
crosswalking MSOA21 boundaries onto MSOA11 (ambiguous for the ~19-38 MSOA21
areas formed by merging multiple MSOA11 areas), this pipeline uses direct,
single-vintage **postcode** joins for both geographies:
- Transactions: postcode -> MSOA11 via the National Statistics Postcode
  Lookup - **2011 Census** (Feb 2024, still actively maintained); postcode
  -> MSOA21 via NSPL21 (Nov 2024).
- New-builds: UPRN -> postcode (via NSUL) -> MSOA11/MSOA21 the same way.

Result: **0% missingness** on every baseline characteristic once merged
into the panel (vs ~1% under a MSOA21-crosswalk approach), and match rates
of 99.999%+ (transactions) / 98.6% (new-build UPRNs overall - degrading to
~85% in 2023 and ~55% in 2024 for the very newest postcodes not yet
assigned a 2011-geography best-fit; flagged in
`outputs/qa/newbuild_geography_match_rate_by_year.csv`, not hidden).

## Deprivation moderator (IMD 2015, revised)

The overall IMD contains the Barriers to Housing and Services domain, whose
indicators include housing affordability (house prices relative to incomes),
so it cannot be the primary moderator for a house-price outcome. The
primary moderator is `imd_ex_housing`: the IMD recombined from the six other
domains at LSOA level (Technical Report method), validated against the
published IMD (Spearman 1.000000, max difference 0.0025), then
population-weighted to MSOA11. Variants, the SAIE-vs-IMD income choice and the
`config.yaml` `moderators:` switches are described in
`DATA_PROCESSING_LOG.md` section E; results are in
`outputs/qa/imd_revision/IMD_REVISION_REPORT.md` and the workbook's
`imd_revision` sheet. Task specification: `docs/imd_revision_brief.md`.

## Why tenure is not the treatment

An earlier pass at this pipeline used EPC-recorded tenure (social vs
private) as the primary treatment split. Investigating it
(`16_investigate_unknown_tenure.py`, output:
`outputs/excel/unknown_tenure_investigation.xlsx`) found that 70-100% of
new-build EPCs per year have **UNKNOWN** tenure - a genuine
data-generating-process limitation, not a processing bug. The brief was
reframed around neighbourhood socioeconomic characteristics instead.
`tenure_group`/`tenure_raw` are still retained in `newbuilds_analysis.parquet`
for QA/descriptive purposes.

## No precomputed interaction terms

`newbuild_x_income` etc. are deliberately **not** stored anywhere - build
them at regression time as `newbuilds_lag1_per_1000 * <zscore>`.
`15_run_diagnostics.py` computes them in-memory purely to report the
correlation matrix and VIF (`outputs/qa/interaction_vif.csv`), which is
also all it uses them for.

## Final outputs (`data/processed/`)

- `final_msoa_year_dissertation_panel.parquet` + `.csv` - one row per
  `msoa11cd x year`, 81,492 rows, 143 columns, zero duplicate keys, 0%
  missingness on baseline characteristics (the spatial spillover variables have documented, flagged missingness where a neighbourhood is undefined).
- `msoa_baseline_characteristics.parquet` + `.csv` - one row per MSOA11 (86 columns).
- `transactions_regression_ready.parquet` - transaction-level robustness
  file (parquet-only, 8.1M rows).
- `transactions_analysis.parquet`, `newbuilds_analysis.parquet` - fuller
  detail behind the regression-ready slice above.

## Not yet done: place-based policy registry

The brief's Phase 2 (Future High Streets Fund, Towns Fund, LUF R1/R2/R3,
Community Renewal Fund, UKSPF as funding intensity; Investment Zones/Pride
in Place/2025 schemes explicitly excluded) is not yet built. Most gov.uk
sources publish these as HTML tables rather than clean CSVs (confirmed by
direct inspection of the FHSF and Towns Fund pages), so this will need
per-scheme scraping plus place-name-to-LAD matching with an honestly
reported `mapping_confidence` field, per the brief.

## Source data provenance

See `config/source_manifest.csv` for every external file used (URL, download
date, filesize, SHA256 checksum, coverage period) - including sources from
an earlier iteration (ONSUD, Census 2021 TS054, the MSOA11<->MSOA21 boundary
crosswalk) that are no longer referenced by the current geography approach,
kept on disk and marked as such rather than deleted.
