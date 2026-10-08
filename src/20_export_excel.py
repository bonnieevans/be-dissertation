"""Export the final dissertation Excel workbook per the Phase-2 (MSOA11)
brief. Aggregates only - no transaction-level data goes into Excel.

Sheets: msoa_year_panel, baseline_characteristics, summary_statistics,
correlations, QA_summary, imd_revision, variable_dictionary, methodology_notes.
(`policy_lad_year` is added once Phase 2 - the place-based intervention
registry - lands.)
"""

from __future__ import annotations

import sys

import pandas as pd

from utils import PROJECT_ROOT, get_logger

log = get_logger("20_export_excel")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"
OUT_XLSX = PROJECT_ROOT / "outputs" / "excel" / "final_dissertation_data.xlsx"

QA_TABLES_FOR_SUMMARY = [
    ("Data quality summary (baseline characteristics)", "data_quality_summary.csv"),
    ("Geography match summary (MSOA11 primary, MSOA21 reference)", "geography_match_summary.csv"),
    ("Transaction count summary", "transaction_count_summary.csv"),
    ("Transactions: cleaning steps (row counts retained)", "transactions_cleaning_steps.csv"),
    ("Transactions: calculated_ppsqm vs UCL priceper discrepancy", "ppsqm_discrepancy_vs_ucl.csv"),
    ("Transactions: classt distribution", "transactions_classt_distribution.csv"),
    ("EPC: new-dwelling dedup summary", "epc_newbuild_dedup_summary.csv"),
    ("New-builds: UNKNOWN tenure share by year (QA only - not the treatment)", "newbuilds_unknown_tenure_share_by_year.csv"),
    ("New-builds: geography match rate by year (MSOA11 + MSOA21)", "newbuild_geography_match_rate_by_year.csv"),
    ("Income: quartile summary (FYE2012)", "income_quartile_summary.csv"),
    ("IMD2015 revision: MSOA11 measures summary", "imd_revision/imd2015_measures_summary.csv"),
    ("IMD2015 revision: 7-domain reproduction of the published IMD (hard stop)", "imd_revision/imd_reproduction_validation.csv"),
    ("Deprivation quartile summary (selected moderator)", "deprivation_quartile_summary.csv"),
    ("Census 2011 tenure: quartile summary", "census2011_tenure_quartile_summary.csv"),
    ("Census 2011 density: quartile summary", "census2011_density_quartile_summary.csv"),
    ("Census 2011 education: quartile summary", "census2011_education_quartile_summary.csv"),
    ("Census 2011 unemployment: quartile summary", "census2011_unemployment_quartile_summary.csv"),
    ("Census 2011 age structure: summary of age-group shares", "census2011_age_structure_summary.csv"),
    ("Accessibility 2014: LSOA time summary and share at the 120-minute cap", "accessibility/jts2014_lsoa_time_summary.csv"),
    ("Accessibility: each measure vs composite and vs distance to town centre", "accessibility/representativeness_each_measure_vs_composite_and_distance.csv"),
    ("Accessibility: PCA of the eight service times", "accessibility/pca_eight_services.csv"),
    ("Accessibility: public transport vs car", "accessibility/public_transport_vs_car.csv"),
    ("Accessibility: correlation with existing moderators (Pearson)", "accessibility/correlation_with_existing_moderators_pearson.csv"),
    ("Census 2011 age structure: correlations among age-group shares", "census2011_age_structure_correlations.csv"),
    ("Final panel: missingness summary", "final_panel_missingness_summary.csv"),
    ("Final panel: zero-construction share", "newbuild_panel_zero_construction_share.csv"),
    ("Price panel: sale_count distribution", "price_panel_sale_count_distribution.csv"),
    ("Interaction VIF (joint model, computed on the fly)", "interaction_vif.csv"),
]

VARIABLE_DICTIONARY = [
    # (variable_name, definition, source, source_year, original_variable, transformation, units, missing_value_treatment)
    ("msoa11cd", "MSOA (2011 boundaries) code - PRIMARY panel key", "ONS geography", "2011", "MSOA11CD", "none", "code", "n/a (key)"),
    ("msoa11nm", "MSOA11 name", "ONS MSOA11/21/LAD22 lookup", "2011", "MSOA11NM", "none", "text", "n/a"),
    ("msoa21cd_reference", "MSOA (2021 boundaries) reference code, via direct NSUL/NSPL postcode join (not a boundary crosswalk)", "ONS NSUL/NSPL21", "2021/Nov 2024", "msoa21", "none", "code", "small % unmatched, see geography_match_summary"),
    ("lad23cd_analysis / lad23nm_analysis", "FIXED current local authority district (Nov 2024 vintage) for LAD x year work", "ONS NSPL21 / LAD names Dec 2024", "current", "laua / LAD24NM", "none", "code/text", "n/a"),
    ("region_code", "Current region code", "ONS NSPL21", "current", "rgn", "none", "code", "n/a"),
    ("year", "Calendar year, 2012-2023 (raw housing data retained)", "UCL transactions / EPC lodgement date", "2012-2023", "dateoftransfer / LODGEMENT_DATE", "EXTRACT(year)", "year", "n/a (key)"),
    ("main_sample", "true if year in [2016,2023], the principal regression sample", "derived", "-", "year", "threshold", "bool", "n/a"),
    ("baseline_households_2011", "FIXED baseline household denominator", "Census 2011 QS405EW", "2011", "All categories: Tenure", "none", "count", "n/a - QA-checked > 0 (hard stop otherwise)"),
    ("newbuild_total", "Count of new-build dwellings (earliest 'new dwelling' EPC per UPRN)", "UCL EPC source", "2008-2024", "TRANSACTION_TYPE='new dwelling'", "dedup by UPRN, count per MSOA-year", "count", "0 if none (explicit zero, not missing)"),
    ("newbuild_houses / newbuild_flats", "New-build counts by EPC PROPERTY_TYPE (Houses=House+Bungalow; Flats=Flat)", "UCL EPC source", "2008-2024", "PROPERTY_TYPE", "count by category", "count", "0 if none"),
    ("newbuilds_per_1000", "newbuild_total normalised per 1,000 baseline households", "derived", "-", "-", "1000*newbuild_total/baseline_households_2011", "per 1,000 households", "n/a"),
    ("newbuilds_lag1_per_1000", "PRIMARY TREATMENT: newbuilds_per_1000 in year t-1", "derived", "-", "-", "LAG(newbuilds_per_1000,1) by MSOA", "per 1,000 households", "0 for the first study year (no t-1 data)"),
    ("newbuilds_lag2_per_1000 / newbuilds_lag3_per_1000", "newbuilds_per_1000 in year t-2 / t-3", "derived", "-", "-", "LAG(2) / LAG(3)", "per 1,000 households", "0 for early study years"),
    ("newbuilds_prev3yr_per_1000", "PRIMARY ROBUSTNESS: sum of newbuilds_per_1000 over t-1, t-2, t-3", "derived", "-", "-", "LAG(1)+LAG(2)+LAG(3)", "per 1,000 households", "partial-window years use available lags only"),
    ("sale_count", "Established-home (old_new='N') transaction count", "UCL linked transactions", "2012-2023", "oldnew", "count per MSOA-year", "count", "NULL if zero sales in that MSOA-year"),
    ("median_ppsqm / mean_ppsqm / mean_log_ppsqm / p25_ppsqm / p75_ppsqm", "Established-home nominal ppsqm, summarised per MSOA-year (0.5/99.5 pct trimmed per year)", "UCL linked transactions", "2012-2023", "price / TOTAL_FLOOR_AREA", "median/mean/quantile", "GBP per sqm", "NULL if sale_count=0"),
    ("log_median_ppsqm", "PRIMARY OUTCOME: ln(median_ppsqm)", "derived", "-", "median_ppsqm", "natural log", "log GBP/sqm", "NULL if median_ppsqm missing/non-positive"),
    ("sales_under_10/20/30/50", "Flag: sale_count below threshold (NOT auto-dropped)", "derived", "-", "sale_count", "threshold indicator", "bool", "n/a"),
    ("detached/semidetached/terraced/flat_sale_share", "Share of established sales by HMLR property type (D/S/T/F)", "UCL linked transactions", "2012-2023", "propertytype", "share of count", "proportion", "NULL if sale_count=0"),
    ("leasehold_sale_share", "Share of established sales with duration='L'", "UCL linked transactions", "2012-2023", "duration", "share of count", "proportion", "NULL if sale_count=0"),
    ("baseline_income_bhc_2011_12", "FIXED baseline: net annual household income before housing costs", "ONS Small Area Income Estimates", "FYE2012 (2011/12)", "Net weekly household income BHC", "annualised (x52)", "GBP/year", "n/a - fully populated (native MSOA11, no crosswalk)"),
    ("baseline_income_upper_ci / lower_ci", "ONS-supplied confidence interval, annualised", "ONS Small Area Income Estimates", "FYE2012", "Upper/Lower confidence limit", "annualised (x52)", "GBP/year", "same as baseline_income_bhc_2011_12"),
    ("log_baseline_income", "ln(baseline_income_bhc_2011_12)", "derived", "-", "baseline_income_bhc_2011_12", "natural log", "log GBP", "same as above"),
    ("income_moderator_used / income_moderator_direction / income_moderator_value", "Which income measure feeds income_z (config moderators.income_moderator: saie | imd_income_rate), its direction, and its raw value. SAIE: higher = RICHER. imd_income_rate: higher = MORE DEPRIVED.", "ONS SAIE FYE2012 or IMD2015 income domain", "2011/12 or 2015", "baseline_income_bhc_2011_12 or imd2015_income_rate_msoa", "config switch", "text / text / GBP per year or rate", "n/a"),
    ("income_z", "England-only unweighted z-score of the SELECTED income moderator (default SAIE, so higher = richer)", "derived", "-", "income_moderator_value", "(x-mean)/sd over 6,791 England MSOA11s", "z-score", "n/a"),
    ("income_quartile", "England-only quartile of the selected income moderator, in the source's own direction (SAIE: 1 = poorest; imd_income_rate: 4 = most income-deprived)", "derived", "-", "income_moderator_value", "ntile(4), ties broken by row order", "1-4", "n/a"),
    ("low_income_q1 / below_median_income", "'Poor' flags that keep their meaning under either income moderator (SAIE bottom quartile / below median; IMD rate top quartile / above median)", "derived", "-", "income_moderator_value", "threshold indicator", "0/1", "n/a"),
    ("imd2015_total_population / imd2015_working_age_population", "MSOA11 totals of the LSOA mid-2012 population and the Employment-domain working-age population (the aggregation weights)", "English Indices of Deprivation 2015, File 7", "2015 (mid-2012 population)", "Total population: mid 2012 / Working age population 18-59/64", "sum over LSOAs", "persons", "n/a - QA: weights sum to MSOA population"),
    ("imd2015_income_rate_msoa", "Income deprivation domain rate (share of population income-deprived). Population-weighted mean of LSOA rates by TOTAL population = the exact MSOA rate. Higher = more deprived.", "English Indices of Deprivation 2015, File 7", "2015", "Income Score (rate)", "sum(rate*pop)/sum(pop)", "proportion", "n/a - fully populated"),
    ("imd2015_employment_rate_msoa", "Employment deprivation domain rate, weighted by WORKING-AGE population (the domain's denominator). Higher = more deprived.", "English Indices of Deprivation 2015, File 7", "2015", "Employment Score (rate)", "sum(rate*wa_pop)/sum(wa_pop)", "proportion", "n/a"),
    ("imd2015_education_score_pw / _health_score_pw / _crime_score_pw / _living_score_pw", "Domain scores (not ranks), population-weighted by total population. Higher = more deprived. Health and Crime scores are standardised (mean ~0), not rates.", "English Indices of Deprivation 2015, File 7", "2015", "domain Score columns", "sum(score*pop)/sum(pop)", "score", "n/a"),
    ("imd2015_barriers_score_pw", "Barriers to Housing and Services domain score, population-weighted. CONTEXT ONLY: includes a housing-affordability indicator (house prices relative to incomes), so it must not be a moderator for a house-price outcome.", "English Indices of Deprivation 2015, File 7", "2015", "Barriers to Housing and Services Score", "sum(score*pop)/sum(pop)", "score", "n/a"),
    ("imd_ex_housing", "PRIMARY deprivation index: IMD recombined from 6 domains (drops Barriers to Housing and Services), LSOA level per Technical Report Section 3 / Appendix F (rank -> R -> exponential transform -> published weights rescaled to sum 1), then population-weighted to MSOA11. 0-100, higher = more deprived.", "English Indices of Deprivation 2015, File 7 + Technical Report", "2015", "7 domain scores/ranks", "see methodology_notes", "index 0-100", "n/a"),
    ("imd_ex_housing_living", "Sensitivity: 5 domains (drops Barriers and Living Environment). Same method.", "English Indices of Deprivation 2015", "2015", "domain ranks", "as imd_ex_housing", "index 0-100", "n/a"),
    ("imd_ex_income_housing", "5 domains (drops Income and Barriers): use when SAIE is the income moderator, to avoid double-counting income. Same method.", "English Indices of Deprivation 2015", "2015", "domain ranks", "as imd_ex_housing", "index 0-100", "n/a"),
    ("imd2015_overall_score_pw", "ROBUSTNESS ONLY - contains housing affordability. Population-weighted mean of the PUBLISHED overall IMD score.", "English Indices of Deprivation 2015, File 7", "2015", "Index of Multiple Deprivation (IMD) Score", "sum(score*pop)/sum(pop)", "IMD score", "n/a"),
    ("imd_ex_housing_bottom20_popshare", "SECONDARY / DESCRIPTIVE ONLY: population share in LSOAs in national deciles 1-2 of imd_ex_housing. Zero for ~62% of MSOAs by construction.", "derived", "-", "imd_ex_housing (LSOA)", "decile = ceil(10*rank/N); population share", "proportion", "n/a"),
    ("imd2015_overall_bottom20_popshare_legacy", "LEGACY - NOT FOR ESTIMATION. The previous moderator basis: share of population in LSOAs in published overall-IMD deciles 1-2 (includes the housing-affordability domain; zero-inflated).", "English Indices of Deprivation 2015", "2015", "published overall IMD decile", "population share", "proportion", "n/a"),
    ("deprivation_moderator_used / deprivation_moderator_value", "Which deprivation measure feeds deprivation_z (config moderators.deprivation_moderator: imd_ex_housing | imd_ex_housing_living | imd_ex_income_housing | imd_income_rate) and its raw value. Always higher = more deprived.", "derived", "-", "selected measure", "config switch", "text / index or rate", "n/a"),
    ("deprivation_z / deprivation_quartile / high_deprivation_q4", "England-only unweighted z-score, quartile (4 = most deprived) and top-quartile flag of the SELECTED deprivation moderator", "derived", "-", "deprivation_moderator_value", "zscore / ntile(4) / quartile==4", "z-score / 1-4 / 0-1", "n/a"),
    ("owner_occupied_2011 / shared_ownership_2011 / social_rented_2011 / private_rented_2011", "Census 2011 tenure category counts", "Census 2011 QS405EW", "2011", "C_TENHUK11 categories", "count", "count", "n/a - fully populated"),
    ("owner_share_2011 / social_rent_share_2011 / private_rent_share_2011", "Census 2011 tenure shares of baseline_households_2011", "Census 2011 QS405EW", "2011", "-", "share", "proportion", "n/a"),
    ("social_rent_z / social_rent_quartile", "Z-score/quartile of social_rent_share_2011 (England-only)", "derived", "-", "social_rent_share_2011", "zscore/ntile(4)", "z-score / 1-4", "n/a"),
    ("population_2011 / area_km2 / population_density_2011", "Usual residents, area, density (persons/km2), 2011", "Census 2011 QS102EW", "2011", "All usual residents / Area Hectares", "population/area", "persons / km2 / persons per km2", "n/a - fully populated"),
    ("log_population_density_2011 / density_z / density_quartile", "Log density and its England-only z-score/quartile", "derived", "-", "population_density_2011", "ln(), zscore/ntile(4)", "log persons/km2 / z-score / 1-4", "n/a"),
    ("degree_share_2011 (optional)", "Level 4+ qualification share of residents 16+", "Census 2011 QS501EW", "2011", "Highest level of qualification", "share", "proportion", "n/a - fully populated"),
    ("unemployment_rate_2011 (optional)", "unemployed / economically active", "Census 2011 QS601EW", "2011", "Economic activity", "share", "proportion", "n/a - fully populated"),
    ("age_share_under16_2011", "Share of all usual residents aged under 16 (ages 0-15); the five age shares sum to 1, so at most four can enter a regression together", "Census 2011 KS102EW (Nomis export)", "2011", "Age structure (16 bands grouped)", "sum of bands / all usual residents", "proportion", "n/a - fully populated"),
    ("age_share_under16_z", "England-only unweighted z-score of age_share_under16_2011 (higher = larger share)", "derived", "-", "age_share_under16_2011", "(x-mean)/sd over 6,791 England MSOA11s", "z-score", "n/a"),
    ("age_share_16_24_2011", "Share of all usual residents aged 16 to 24; the five age shares sum to 1, so at most four can enter a regression together", "Census 2011 KS102EW (Nomis export)", "2011", "Age structure (16 bands grouped)", "sum of bands / all usual residents", "proportion", "n/a - fully populated"),
    ("age_share_16_24_z", "England-only unweighted z-score of age_share_16_24_2011 (higher = larger share)", "derived", "-", "age_share_16_24_2011", "(x-mean)/sd over 6,791 England MSOA11s", "z-score", "n/a"),
    ("age_share_25_44_2011", "Share of all usual residents aged 25 to 44; the five age shares sum to 1, so at most four can enter a regression together", "Census 2011 KS102EW (Nomis export)", "2011", "Age structure (16 bands grouped)", "sum of bands / all usual residents", "proportion", "n/a - fully populated"),
    ("age_share_25_44_z", "England-only unweighted z-score of age_share_25_44_2011 (higher = larger share)", "derived", "-", "age_share_25_44_2011", "(x-mean)/sd over 6,791 England MSOA11s", "z-score", "n/a"),
    ("age_share_45_64_2011", "Share of all usual residents aged 45 to 64; the five age shares sum to 1, so at most four can enter a regression together", "Census 2011 KS102EW (Nomis export)", "2011", "Age structure (16 bands grouped)", "sum of bands / all usual residents", "proportion", "n/a - fully populated"),
    ("age_share_45_64_z", "England-only unweighted z-score of age_share_45_64_2011 (higher = larger share)", "derived", "-", "age_share_45_64_2011", "(x-mean)/sd over 6,791 England MSOA11s", "z-score", "n/a"),
    ("age_share_65plus_2011", "Share of all usual residents aged 65 and over; the five age shares sum to 1, so at most four can enter a regression together", "Census 2011 KS102EW (Nomis export)", "2011", "Age structure (16 bands grouped)", "sum of bands / all usual residents", "proportion", "n/a - fully populated"),
    ("age_share_65plus_z", "England-only unweighted z-score of age_share_65plus_2011 (higher = larger share)", "derived", "-", "age_share_65plus_2011", "(x-mean)/sd over 6,791 England MSOA11s", "z-score", "n/a"),
    ("access_employment_pt_min_2014", "Population-weighted mean minutes to the nearest employment centres (500-4,999 jobs) by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_employment_car_min_2014", "Population-weighted mean minutes to the nearest employment centres (500-4,999 jobs) by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_primary_school_pt_min_2014", "Population-weighted mean minutes to the nearest primary schools by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_primary_school_car_min_2014", "Population-weighted mean minutes to the nearest primary schools by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_secondary_school_pt_min_2014", "Population-weighted mean minutes to the nearest secondary schools by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_secondary_school_car_min_2014", "Population-weighted mean minutes to the nearest secondary schools by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_further_education_pt_min_2014", "Population-weighted mean minutes to the nearest further education by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_further_education_car_min_2014", "Population-weighted mean minutes to the nearest further education by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_gp_pt_min_2014", "Population-weighted mean minutes to the nearest GPs by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_gp_car_min_2014", "Population-weighted mean minutes to the nearest GPs by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_hospital_pt_min_2014", "Population-weighted mean minutes to the nearest hospitals by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_hospital_car_min_2014", "Population-weighted mean minutes to the nearest hospitals by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_food_store_pt_min_2014", "Population-weighted mean minutes to the nearest food stores by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_food_store_car_min_2014", "Population-weighted mean minutes to the nearest food stores by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_town_centre_pt_min_2014", "Population-weighted mean minutes to the nearest town centres by public transport/walk, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_town_centre_car_min_2014", "Population-weighted mean minutes to the nearest town centres by car, AM peak, 2014 (higher = less accessible; DfT caps at 120)", "DfT Journey Time Statistics 2014 (LSOA11 tables)", "2014", "Modelled travel time; service-user-population weights", "weighted mean of LSOA11 minutes", "minutes", "n/a - fully populated"),
    ("access_keyservices_pt_min_2014 / access_keyservices_car_min_2014", "DfT key-services average: equal-weighted mean of the eight MSOA-level service times, by public transport/walk and by car (higher = less accessible)", "derived from DfT JTS 2014", "2014", "Eight service times", "mean of 8", "minutes", "n/a"),
    ("dist_town_centre_km", "Straight-line distance from the MSOA11 population-weighted centroid to the nearest English Town Centres 2004 polygon (0 = inside one)", "MHCLG English Town Centres 2004; ONS MSOA11 population-weighted centroids", "2004 / 2011", "Polygon geometry, British National Grid", "nearest-polygon distance / 1000", "km", "n/a - fully populated"),
    ("dist_town_centre_centroid_km", "As dist_town_centre_km but to the town-centre polygon centroid (sensitivity)", "as above", "2004 / 2011", "Polygon centroid", "distance / 1000", "km", "n/a"),
    ("access_keyservices_pt_z / access_keyservices_car_z / dist_town_centre_z", "England-only unweighted z-scores (higher = less accessible); strongly right-skewed (the Isles of Scilly has z above 20), so the log versions are usually better", "derived", "-", "-", "(x-mean)/sd", "z-score", "n/a"),
    ("log_access_keyservices_pt_min_2014 / log_access_keyservices_car_min_2014 / log_dist_town_centre_km", "Natural log of the composites; log(1 + km) for distance", "derived", "-", "-", "ln / ln(1+x)", "log units", "n/a"),
    ("baseline_complete / analysis_complete_case", "QA flags: all baseline moderators present / baseline+sales present", "derived", "-", "-", "boolean AND of component nulls", "bool", "n/a"),
    ("NOTE on interactions", "newbuild_x_income etc. are NOT stored anywhere in this workbook or the processed parquet files - build them at regression time as newbuilds_lag1_per_1000 * <zscore>, per the user's explicit instruction (avoids a stale interaction column if the treatment definition changes).", "-", "-", "-", "-", "-", "-"),
]

METHODOLOGY_NOTES = [
    "GEOGRAPHY: MSOA11 IS THE PRIMARY KEY",
    "Per the user's explicit instruction: income (FYE2012), IMD2015, and Census 2011 tenure/density/"
    "education/unemployment are all natively 2011 statistical geography, so MSOA11CD is the panel's "
    "primary key - not MSOA21CD as in an earlier iteration of this pipeline. MSOA21CD is retained as a "
    "reference column (msoa21cd_reference).",
    "",
    "HOW MSOA11 WAS OBTAINED WITHOUT A BOUNDARY CROSSWALK",
    "Rather than crosswalking MSOA21 boundaries onto MSOA11 (which is ambiguous for the ~19-38 MSOA21 "
    "areas formed by merging multiple MSOA11 areas), this build uses direct, single-vintage postcode "
    "joins for both geographies: transactions join postcode -> MSOA11 via the National Statistics "
    "Postcode Lookup - 2011 Census (Feb 2024, still actively maintained) and postcode -> MSOA21 via "
    "NSPL21 (Nov 2024, bundled with the UCL data). New-builds join UPRN -> postcode via NSUL (Feb 2024) "
    "then postcode -> MSOA11/MSOA21 the same way. This gives 100% MSOA11 coverage for baseline "
    "characteristics (no crosswalk-driven missingness) and match rates of 99.999%+ (transactions) / "
    "98.6% (new-build UPRNs, degrading to ~85% in 2023 and ~55% in 2024 for the very newest postcodes "
    "not yet assigned a 2011-geography best-fit - see geography_match_summary and "
    "newbuild_geography_match_rate_by_year in QA_summary).",
    "",
    "WHY TENURE IS NOT THE TREATMENT",
    "An earlier build used EPC-recorded tenure (social/private) as the treatment split and found "
    "70-100% of new-build EPCs per year have UNKNOWN tenure (see "
    "outputs/excel/unknown_tenure_investigation.xlsx) - a genuine data-generating-process limitation, "
    "not a processing error. tenure_group/tenure_raw are retained in newbuilds_analysis.parquet for "
    "QA/descriptive purposes only.",
    "",
    "NO PRECOMPUTED INTERACTION TERMS",
    "newbuild_x_income, newbuild_x_deprivation, newbuild_x_socialrent, newbuild_x_density are "
    "deliberately NOT stored in the final panel, per the user's instruction - build them at regression "
    "time as newbuilds_lag1_per_1000 * <zscore>. 15_run_diagnostics.py computes them in-memory purely "
    "to report the correlation matrix and VIF.",
    "",
    "STUDY PERIOD",
    "Raw housing data retained 2012-2023. main_sample=true flags 2016-2023 (the principal regression "
    "sample); main_sample=false rows are retained, not dropped, to support pre-trend checks.",
    "",
    "SOURCE DATA",
    "UCL ReShare 857911 (House Price Per Square Metre in England and Wales, 1995-2024): linked PPD-EPC "
    "transactions and the deposited EPC source. Restricted to England only (MSOA11CD like 'E02%').",
    "",
    "HOUSE PRICE OUTCOME",
    "Established homes only (old_new='N'), new-build transactions excluded from the outcome. ppsqm = "
    "transaction_price / TOTAL_FLOOR_AREA, verified against UCL's own priceper (matched to "
    "floating-point rounding). Outliers trimmed at the 0.5/99.5 percentile of ppsqm WITHIN EACH "
    "CALENDAR YEAR; an untrimmed version is kept in data/interim/ for robustness.",
    "",
    "NEW-BUILD IDENTIFICATION",
    "EPC TRANSACTION_TYPE == 'new dwelling', one record per UPRN (earliest valid new-dwelling EPC). "
    "Estimated ONLY from the EPC source (never the linked transactions) to avoid selection bias.",
    "",
    "BASELINE INCOME (FYE2012)",
    "ONS Small Area Income Estimates, financial year ending 2012 (2011/12), before housing costs, "
    "natively at MSOA11 (no crosswalk needed). Reports NET WEEKLY income; annualised (x52). "
    "Upper/lower confidence-interval columns retained. Rounded by ONS to the nearest GBP10/week (~83 "
    "distinct values across 6,791 MSOA11s) - this is genuine ONS publication practice, verified by "
    "direct inspection, not a data-merge artefact (the pipeline's own QA check distinguishes the two: "
    "see 06_prepare_income.py's duplicate-value threshold, calibrated against this exact dataset).",
    "",
    "DEPRIVATION (IMD2015) - REVISED",
    "Why: the overall IMD includes the Barriers to Housing and Services domain, whose indicators "
    "include housing affordability (house prices relative to incomes). The outcome is house prices, so "
    "the overall IMD cannot be the primary moderator. Averaging LSOA ranks is invalid and has been "
    "removed everywhere (imd2015_rank_percentile_pw no longer exists).",
    "Method (IMD 2015 Technical Report, Section 3.6-3.7 and Appendix F): for each LSOA and domain, "
    "R = (N - rank + 1)/N with N = 32,844 (R = 1 most deprived); X = -23 ln(1 - R(1 - exp(-100/23))); "
    "the weighted sum of X with the published weights (Income .225, Employment .225, Education .135, "
    "Health .135, Crime .093, Barriers .093, Living Environment .093). The published weights sum to "
    "0.999 and the published IMD uses them as published; they are rescaled to sum to 1 ONLY for the "
    "reduced variants. VALIDATION (hard stop): recombining all 7 domains reproduces the published IMD "
    "score with Spearman 1.000000 and a maximum absolute difference of 0.0025 index points. The published "
    "domain RANKS are used (config imd2015.rank_source) rather than re-ranking the rounded published "
    "scores, because those scores are rounded to 3 decimals and re-ranking creates large tie blocks "
    "(for Income only 1.7% of ranks agree); the published ranks reproduce the official transformed "
    "scores (File 9) to within rounding. This is a flagged deviation from re-ranking the scores.",
    "Variants: imd_ex_housing (6 domains, PRIMARY), imd_ex_housing_living (5, sensitivity), "
    "imd_ex_income_housing (5, drops Income as well: pair it with SAIE). The overall published IMD is kept only as "
    "imd2015_overall_score_pw, ROBUSTNESS ONLY.",
    "Aggregation LSOA11 -> MSOA11 (nesting is exact, verified: 32,844 LSOAs, each in exactly one of 6,791 "
    "MSOA11s): population-weighted means of LSOA VALUES (never ranks or deciles). Income rate by total "
    "population (gives the exact MSOA rate); Employment rate by working-age population; all other domain "
    "scores and the recombined indices by total population. QA: weights sum to the MSOA population and an "
    "independent SQL recomputation agrees to 1e-14.",
    "Bottom-20 share: recomputed from national deciles of imd_ex_housing "
    "(imd_ex_housing_bottom20_popshare). SECONDARY and descriptive only: it is zero for 61.7% of MSOAs by "
    "construction. The old overall-IMD share is kept as imd2015_overall_bottom20_popshare_legacy, NOT FOR "
    "ESTIMATION.",
    "Moderator switches (config.yaml, moderators:): deprivation_moderator in {imd_ex_housing (default), "
    "imd_ex_housing_living, imd_ex_income_housing, imd_income_rate}; income_moderator in {saie (default), "
    "imd_income_rate}. Choosing imd_income_rate for both is refused. deprivation_z, deprivation_quartile "
    "and high_deprivation_q4 are rebuilt (England-only, unweighted) from the chosen deprivation moderator. "
    "SIGN CONVENTION: SAIE income higher = richer; every IMD measure higher = more deprived. If "
    "imd_income_rate is the income moderator, income_z follows it (higher = more deprived), so the sign "
    "of the income interaction flips.",
    "Findings (see the imd_revision sheet): overall IMD vs imd_ex_housing Spearman 0.990 and 10.5% of "
    "MSOA11s change quartile (none by two or more); SAIE vs the IMD income rate Pearson -0.683 (log "
    "SAIE) and Spearman -0.703, so they overlap but are far from the same measure; the Barriers domain is "
    "positively correlated with SAIE income (0.109), the contamination the revision removes. Joint-model "
    "VIFs peak at about 4.6 for the deprivation interaction.",
    "",
    "CENSUS 2011 MODERATORS",
    "QS405EW (tenure), QS102EW (density), QS501EW (education, optional), QS601EW (unemployment, "
    "optional) at MSOA11 via the Nomis API (TYPE297 = 2011 middle-layer SOAs) - no crosswalk needed. "
    "unemployment_rate_2011 = unemployed / economically active (per the brief's stated preference). "
    "ACCESSIBILITY: DfT Journey Time Statistics 2014 LSOA11 tables (modelled minutes to the nearest of eight service types, AM peak, by public transport/walk and car) are population-weighted to MSOA11; the key-services average is the equal-weighted mean of the eight. dist_town_centre_km is the straight-line distance from the population-weighted MSOA11 centroid to the nearest 2004 town-centre polygon. Higher = less accessible throughout. See outputs/qa/accessibility/ACCESSIBILITY_REPORT.md.", "Age structure comes from KS102EW (Nomis export, MSOA11): the 16 age bands are grouped into five shares "
    "of all usual residents (under 16, 16-24, 25-44, 45-64, 65+) with England-only z-scores; the bands were "
    "checked to sum to the total and the total matches QS102EW exactly in all 6,791 MSOA11s.",
    "",
    "DIAGNOSTICS",
    "Correlation matrix and VIF are reported for information only - a variable is never dropped solely "
    "for being correlated with another. See baseline_characteristic_correlations and interaction_vif "
    "in QA_summary; no VIF exceeded 10 for the four required interaction terms.",
    "",
    "PLACE-BASED POLICY CONTROLS",
    "Not yet built in this pass (Future High Streets Fund, Towns Fund, LUF R1/R2/R3, Community Renewal "
    "Fund, UKSPF) - see the separate place-based intervention registry work in progress. Investment "
    "Zones, Pride in Place, the 2025 Local Growth Fund, UKSPF 2025/26, and Local Regeneration Fund are "
    "explicitly EXCLUDED as they post-date the 2016-2023 sample.",
    "",
    "ROBUSTNESS DATA",
    "data/processed/transactions_regression_ready.parquet (parquet-only, millions of rows) supports "
    "log_ppsqm ~ development exposure + property characteristics + MSOA FE + year/month FE outside "
    "this Excel export. data/processed/final_msoa_year_dissertation_panel.parquet and "
    "msoa_baseline_characteristics.parquet also have .csv copies for manual inspection.",
]


def write_stacked_sheet(writer, sheet_name: str, tables: list) -> object:
    startrow = 0
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#DDEBF7"})
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    for title, fname in tables:
        fpath = QA_DIR / fname
        worksheet.write(startrow, 0, title, header_fmt)
        startrow += 1
        if not fpath.exists():
            worksheet.write(startrow, 0, f"[missing: {fname}]")
            startrow += 2
            continue
        df = pd.read_csv(fpath)
        if df.columns[0].startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "variable"})
        for j, col in enumerate(df.columns):
            worksheet.write(startrow, j, col, header_fmt)
        for i, row in enumerate(df.itertuples(index=False)):
            for j, val in enumerate(row):
                worksheet.write(startrow + 1 + i, j, "" if pd.isna(val) else val)
        startrow += len(df) + 3
    return worksheet


IMD_REVISION_TABLES = [
    ("Moderator selection in this build (config.yaml moderators:)", "imd_revision/moderator_selection.csv"),
    ("Reproduction of the published IMD from 7 recombined domains (hard stop: Spearman > 0.999)", "imd_revision/imd_reproduction_validation.csv"),
    ("Transform check vs the official File 9 exponentially transformed domain scores", "imd_revision/imd_transform_vs_file9.csv"),
    ("Why published ranks are used: agreement of re-ranked rounded scores with published ranks", "imd_revision/imd_rank_source_agreement.csv"),
    ("Aggregation QA (weights = MSOA population; independent SQL recomputation)", "imd_revision/imd_aggregation_qa.csv"),
    ("MSOA11 IMD measures: descriptive statistics", "imd_revision/imd2015_measures_summary.csv"),
    ("Pearson correlations (6,791 MSOA11s). SAIE higher = richer; IMD higher = more deprived", "imd_revision/corr_pearson.csv"),
    ("Spearman correlations", "imd_revision/corr_spearman.csv"),
    ("Joint-model VIFs under (a) SAIE + imd_ex_housing and (b) SAIE + imd_ex_income_housing", "imd_revision/joint_model_vif_by_model.csv"),
    ("z-score correlation matrices under (a) and (b)", "imd_revision/zscore_correlation_matrix_by_model.csv"),
    ("Quartile movement summary (overall IMD -> imd_ex_housing is the primary comparison)", "imd_revision/quartile_movement_summary.csv"),
    ("Quartile transition counts: overall IMD (rows) -> imd_ex_housing (columns)", "imd_revision/quartile_transition_overall_imd_to_imd_ex_housing.csv"),
    ("Quartile transition counts: overall IMD (rows) -> imd_ex_income_housing (columns)", "imd_revision/quartile_transition_overall_imd_to_imd_ex_income_housing.csv"),
    ("Deprivation quartile summary (selected moderator)", "deprivation_quartile_summary.csv"),
]


def write_qa_summary_sheet(writer) -> None:
    write_stacked_sheet(writer, "QA_summary", QA_TABLES_FOR_SUMMARY)


def write_imd_revision_sheet(writer) -> None:
    ws = write_stacked_sheet(writer, "imd_revision", IMD_REVISION_TABLES)
    png = QA_DIR / "imd_revision" / "scatter_saie_vs_imd_income_rate.png"
    if png.exists():
        ws.insert_image("N2", str(png), {"x_scale": 0.55, "y_scale": 0.55})


def main() -> int:
    panel_path = PROCESSED_DIR / "final_msoa_year_dissertation_panel.parquet"
    baseline_path = PROCESSED_DIR / "msoa_baseline_characteristics.parquet"
    summary_stats_path = TABLES_DIR / "summary_statistics.csv"
    corr_path = QA_DIR / "baseline_characteristic_correlations.csv"

    for p in [panel_path, baseline_path, summary_stats_path]:
        if not p.exists():
            log.error(f"{p} not found - run the preceding pipeline scripts first.")
            return 1

    panel_df = pd.read_parquet(panel_path)
    baseline_df = pd.read_parquet(baseline_path)
    summary_df = pd.read_csv(summary_stats_path)
    corr_df = pd.read_csv(corr_path, index_col=0) if corr_path.exists() else pd.DataFrame()

    variable_dict_df = pd.DataFrame(
        VARIABLE_DICTIONARY,
        columns=["variable_name", "definition", "source", "source_year", "original_variable",
                 "transformation", "units", "missing_value_treatment"],
    )
    methodology_df = pd.DataFrame({"note": METHODOLOGY_NOTES})

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        panel_df.to_excel(writer, sheet_name="msoa_year_panel", index=False)
        baseline_df.to_excel(writer, sheet_name="baseline_characteristics", index=False)
        summary_df.to_excel(writer, sheet_name="summary_statistics", index=False)
        if not corr_df.empty:
            corr_df.to_excel(writer, sheet_name="correlations")
        write_qa_summary_sheet(writer)
        write_imd_revision_sheet(writer)
        variable_dict_df.to_excel(writer, sheet_name="variable_dictionary", index=False)
        methodology_df.to_excel(writer, sheet_name="methodology_notes", index=False, header=False)

    log.info(f"Wrote {OUT_XLSX} ({panel_df.shape[0]:,} panel rows x {panel_df.shape[1]} cols, 8 sheets)")
    log.info("Excel export complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
