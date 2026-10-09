"""Merge the price panel, new-build panel, geography reference, and all six
baseline-characteristics tables (income, IMD2015 deprivation, Census 2011
tenure/density/education/unemployment) into the final England MSOA11CD x
year analysis panel.

Per the user's Phase-2 brief:
- MSOA11CD is the PRIMARY key (not MSOA21CD - see 05_attach_geography.py).
- NO precomputed interaction terms - build newbuilds_lag1_per_1000 * income_z
  etc. at regression time instead, so a treatment-definition change never
  leaves a stale interaction column behind.
- Three outputs: the main panel, a separate one-row-per-MSOA11 baseline
  table, and a transaction-level robustness file - each of the first two
  also gets a small CSV copy for manual inspection (the transaction-level
  file stays parquet-only: millions of rows, CSV would be slow and lose
  type fidelity).
- Hard QA stops (return 1, not just a log line) for: duplicate keys,
  baseline characteristics varying across years within an MSOA,
  baseline_households_2011 <= 0, non-England rows present.
"""

from __future__ import annotations

import sys

import pandas as pd

import imd_methods as im
from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("14_merge_final_panel")

GEO_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"
MSOA11_NAMES_CSV = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa_crosswalks" / "MSOA11_MSOA21_LAD22_EW_LU_v2.csv"
LAD_NAMES_CSV = PROJECT_ROOT / "data" / "raw" / "geography" / "lad_names" / "LAD_DEC_2024_UK_NC.csv"

REGION_NAMES = {
    "E12000001": "North East", "E12000002": "North West",
    "E12000003": "Yorkshire and The Humber", "E12000004": "East Midlands",
    "E12000005": "West Midlands", "E12000006": "East of England",
    "E12000007": "London", "E12000008": "South East", "E12000009": "South West",
}

REQUIRED_FILES = [
    "msoa_year_price_panel_trimmed.parquet",
    "msoa_year_newbuild_panel.parquet",
    "msoa11_geography_reference.parquet",
    "msoa11_baseline_income.parquet",
    "msoa11_imd2015.parquet",
    "msoa11_census2011_tenure.parquet",
    "msoa11_census2011_density.parquet",
    "msoa11_census2011_education.parquet",
    "msoa11_census2011_unemployment.parquet",
    "msoa11_census2011_age.parquet",
    "msoa11_accessibility2014.parquet",
    "msoa11_greenbelt2011.parquet",
    "msoa_year_spillover_exposure.parquet",
    "msoa11_spatial_characteristics.parquet",
]


def _existing_columns(path) -> list[str]:
    if not path.exists():
        return []
    import pyarrow.parquet as pq
    return list(pq.read_schema(path).names)


def _log_column_change(label: str, old: list[str], new: list[str]) -> None:
    removed = [c for c in old if c not in new]
    added = [c for c in new if c not in old]
    log.info(f"Column count, {label}: {len(old) or 'n/a'} -> {len(new)} "
             f"(+{len(added)} added, -{len(removed)} removed)")
    if added:
        log.info(f"  added:   {added}")
    if removed:
        log.info(f"  removed: {removed}")


def main() -> int:
    cfg = load_config()
    reg_start = cfg["study_period"]["regression_sample_start_year"]
    reg_end = cfg["study_period"]["regression_sample_end_year"]

    for fname in REQUIRED_FILES:
        if not (GEO_DIR / fname).exists():
            log.error(f"{GEO_DIR / fname} not found - run the preceding src/0N_*.py scripts first.")
            return 1

    con = get_duckdb_connection()
    for view, fname in [
        ("newbuild_v", "msoa_year_newbuild_panel.parquet"),
        ("price_v", "msoa_year_price_panel_trimmed.parquet"),
        ("geog_v", "msoa11_geography_reference.parquet"),
        ("income_v", "msoa11_baseline_income.parquet"),
        ("imd_v", "msoa11_imd2015.parquet"),
        ("tenure_v", "msoa11_census2011_tenure.parquet"),
        ("density_v", "msoa11_census2011_density.parquet"),
        ("education_v", "msoa11_census2011_education.parquet"),
        ("unemployment_v", "msoa11_census2011_unemployment.parquet"),
        ("age_v", "msoa11_census2011_age.parquet"),
        ("access_v", "msoa11_accessibility2014.parquet"),
        ("greenbelt_v", "msoa11_greenbelt2011.parquet"),
        ("spill_v", "msoa_year_spillover_exposure.parquet"),
        ("spstat_v", "msoa11_spatial_characteristics.parquet"),
    ]:
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{(GEO_DIR / fname).as_posix()}')")

    # --- Selected moderators (config switches: moderators.deprivation_moderator /
    # moderators.income_moderator). Built once per MSOA11, England-only unweighted. ---
    try:
        dep_choice, inc_choice = im.resolve_moderators(cfg)
    except ValueError as e:
        log.error(f"STOP: {e}")
        return 1
    for w in im.moderator_overlap_warnings(dep_choice, inc_choice):
        log.warning(w)
    log.info(f"Moderators: deprivation = {dep_choice}; income = {inc_choice} "
             "(sign: SAIE higher = richer; every IMD measure higher = more deprived).")
    base = (pd.read_parquet(GEO_DIR / "msoa11_imd2015.parquet")
            .merge(pd.read_parquet(GEO_DIR / "msoa11_baseline_income.parquet"), on="msoa11cd", how="inner")
            .set_index("msoa11cd").sort_index())
    mods = im.build_moderators(base, cfg)
    mods.index.name = "msoa11cd"
    con.register("mod_v", mods.reset_index())
    pd.DataFrame([{"deprivation_moderator": dep_choice, "income_moderator": inc_choice,
                   "n_msoa11": len(mods)}]).to_csv(QA_DIR / "imd_revision" / "moderator_selection.csv", index=False)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE msoa11_names AS
        SELECT DISTINCT MSOA11CD AS msoa11cd, MSOA11NM AS msoa11nm
        FROM read_csv_auto('{MSOA11_NAMES_CSV.as_posix()}', ALL_VARCHAR=true)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE lad_names AS
        SELECT DISTINCT LAD24CD AS lad_code, LAD24NM AS lad_name
        FROM read_csv_auto('{LAD_NAMES_CSV.as_posix()}', ALL_VARCHAR=true)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE panel AS
        SELECT
            nb.msoa11cd,
            mn.msoa11nm,
            g.msoa21cd_reference,
            g.lad23cd_analysis,
            ln.lad_name AS lad23nm_analysis,
            g.rgn_code AS region_code,
            nb.year,
            CASE WHEN nb.year BETWEEN {reg_start} AND {reg_end} THEN true ELSE false END AS main_sample,

            nb.baseline_households_2011,

            p.sale_count, p.median_price, p.median_ppsqm, p.log_median_ppsqm, p.mean_log_ppsqm,
            p.mean_ppsqm, p.p25_ppsqm, p.p75_ppsqm, p.median_floor_area,
            p.detached_sale_share, p.semidetached_sale_share, p.terraced_sale_share,
            p.flat_sale_share, p.leasehold_sale_share,
            p.sales_under_10, p.sales_under_20, p.sales_under_30, p.sales_under_50,

            nb.newbuild_total, nb.newbuild_houses, nb.newbuild_flats,
            nb.median_newbuild_floor_area, nb.mean_newbuild_floor_area,
            nb.newbuilds_per_1000,
            nb.newbuilds_lag1, nb.newbuilds_lag1_per_1000,
            nb.newbuilds_lag2, nb.newbuilds_lag2_per_1000,
            nb.newbuilds_lag3, nb.newbuilds_lag3_per_1000,
            nb.newbuilds_prev3yr, nb.newbuilds_prev3yr_per_1000,

            inc.baseline_income_bhc_2011_12, inc.baseline_income_upper_ci, inc.baseline_income_lower_ci,
            inc.log_baseline_income,
            md.income_moderator_used, md.income_moderator_direction, md.income_moderator_value,
            md.income_z, md.income_quartile, md.low_income_q1, md.below_median_income,

            imd.imd_ex_housing, imd.imd_ex_housing_living, imd.imd_ex_income_housing,
            imd.imd2015_income_rate_msoa, imd.imd2015_overall_score_pw,
            imd.imd_ex_housing_bottom20_popshare, imd.imd2015_overall_bottom20_popshare_legacy,
            md.deprivation_moderator_used, md.deprivation_moderator_value,
            md.deprivation_z, md.deprivation_quartile, md.high_deprivation_q4,

            tn.owner_occupied_2011, tn.shared_ownership_2011, tn.social_rented_2011, tn.private_rented_2011,
            tn.owner_share_2011, tn.social_rent_share_2011, tn.private_rent_share_2011,
            tn.social_rent_z, tn.social_rent_quartile,

            dn.population_2011, dn.area_km2, dn.population_density_2011,
            dn.log_population_density_2011, dn.density_z, dn.density_quartile,

            ed.degree_share_2011, ed.degree_share_z, ed.degree_share_quartile,

            un.unemployment_rate_2011, un.unemployment_z, un.unemployment_quartile,

            ag.age_share_under16_2011, ag.age_share_16_24_2011, ag.age_share_25_44_2011,
            ag.age_share_45_64_2011, ag.age_share_65plus_2011,
            ag.age_share_under16_z, ag.age_share_16_24_z, ag.age_share_25_44_z,
            ag.age_share_45_64_z, ag.age_share_65plus_z,
            ac.access_keyservices_pt_min_2014,
            ac.access_keyservices_car_min_2014,
            ac.dist_town_centre_km,
            ac.access_keyservices_pt_z,
            ac.access_keyservices_car_z,
            ac.dist_town_centre_z,
            ac.log_access_keyservices_pt_min_2014,
            ac.log_access_keyservices_car_min_2014,
            ac.log_dist_town_centre_km,

            gbt.greenbelt_share_2011, gbt.greenbelt_share_z, gbt.greenbelt_any_2011,

            sp.* EXCLUDE (msoa11cd, year),
            ss.* EXCLUDE (msoa11cd),

            (p.sale_count IS NOT NULL AND inc.baseline_income_bhc_2011_12 IS NOT NULL
                AND imd.imd_ex_housing IS NOT NULL AND tn.social_rent_share_2011 IS NOT NULL
                AND dn.population_density_2011 IS NOT NULL) AS baseline_complete

        FROM newbuild_v nb
        LEFT JOIN price_v p ON nb.msoa11cd = p.msoa11cd AND nb.year = p.year
        LEFT JOIN geog_v g ON nb.msoa11cd = g.msoa11cd
        LEFT JOIN msoa11_names mn ON nb.msoa11cd = mn.msoa11cd
        LEFT JOIN lad_names ln ON g.lad23cd_analysis = ln.lad_code
        LEFT JOIN income_v inc ON nb.msoa11cd = inc.msoa11cd
        LEFT JOIN imd_v imd ON nb.msoa11cd = imd.msoa11cd
        LEFT JOIN mod_v md ON nb.msoa11cd = md.msoa11cd
        LEFT JOIN tenure_v tn ON nb.msoa11cd = tn.msoa11cd
        LEFT JOIN density_v dn ON nb.msoa11cd = dn.msoa11cd
        LEFT JOIN education_v ed ON nb.msoa11cd = ed.msoa11cd
        LEFT JOIN unemployment_v un ON nb.msoa11cd = un.msoa11cd
        LEFT JOIN age_v ag ON nb.msoa11cd = ag.msoa11cd
        LEFT JOIN access_v ac ON nb.msoa11cd = ac.msoa11cd
        LEFT JOIN greenbelt_v gbt ON nb.msoa11cd = gbt.msoa11cd
        LEFT JOIN spill_v sp ON nb.msoa11cd = sp.msoa11cd AND nb.year = sp.year
        LEFT JOIN spstat_v ss ON nb.msoa11cd = ss.msoa11cd
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE panel AS
        SELECT *, (baseline_complete AND sale_count IS NOT NULL) AS analysis_complete_case
        FROM panel
        """
    )

    # =================================================================
    # HARD QA STOPS (per the brief's section 17 - fail, don't just log)
    # =================================================================
    n_rows = con.sql("SELECT count(*) FROM panel").fetchone()[0]
    n_msoa = con.sql("SELECT count(DISTINCT msoa11cd) FROM panel").fetchone()[0]

    n_dupe_keys = con.sql(
        "SELECT count(*) FROM (SELECT msoa11cd, year, count(*) c FROM panel GROUP BY 1,2 HAVING c > 1)"
    ).fetchone()[0]
    if n_dupe_keys:
        log.error(f"STOP: {n_dupe_keys} duplicate msoa11cd x year keys in the final panel.")
        return 1

    n_non_england = con.sql("SELECT count(*) FROM panel WHERE msoa11cd NOT LIKE 'E02%'").fetchone()[0]
    if n_non_england:
        log.error(f"STOP: {n_non_england} non-England rows found in the final panel.")
        return 1

    n_bad_households = con.sql("SELECT count(*) FROM panel WHERE baseline_households_2011 <= 0").fetchone()[0]
    if n_bad_households:
        log.error(f"STOP: {n_bad_households} rows have baseline_households_2011 <= 0.")
        return 1

    # Baseline characteristics must be constant across years within an MSOA
    # (they are fixed 2011/2012/2015 measures, never updated by year).
    n_varying = con.sql(
        """
        SELECT count(*) FROM (
            SELECT msoa11cd, count(DISTINCT baseline_income_bhc_2011_12) AS n_income,
                   count(DISTINCT imd_ex_housing) AS n_imd,
                   count(DISTINCT imd2015_income_rate_msoa) AS n_imd_inc,
                   count(DISTINCT deprivation_z) AS n_dep_z,
                   count(DISTINCT deprivation_quartile) AS n_dep_q,
                   count(DISTINCT income_z) AS n_inc_z,
                   count(DISTINCT social_rent_share_2011) AS n_tenure,
                   count(DISTINCT population_density_2011) AS n_density,
                   count(DISTINCT age_share_65plus_2011) AS n_age65, count(DISTINCT age_share_25_44_2011) AS n_age2544,
                   count(DISTINCT access_keyservices_pt_min_2014) AS n_acc_pt, count(DISTINCT access_keyservices_car_min_2014) AS n_acc_car,
                   count(DISTINCT dist_town_centre_km) AS n_dist, count(DISTINCT greenbelt_share_2011) AS n_gb,
                   count(DISTINCT nbr_queen_n) AS n_qn, count(DISTINCT nbr_queen_income_bhc2012_hhmean) AS n_qinc,
                   count(DISTINCT income_gap_own_minus_nbr_queen_log) AS n_gap, count(DISTINCT nbr_5km_n) AS n_5n
            FROM panel GROUP BY msoa11cd
            HAVING n_income > 1 OR n_imd > 1 OR n_imd_inc > 1 OR n_dep_z > 1 OR n_dep_q > 1
                   OR n_inc_z > 1 OR n_tenure > 1 OR n_density > 1 OR n_age65 > 1 OR n_age2544 > 1 OR n_acc_pt > 1 OR n_acc_car > 1 OR n_dist > 1 OR n_gb > 1 OR n_qn > 1 OR n_qinc > 1 OR n_gap > 1 OR n_5n > 1
        )
        """
    ).fetchone()[0]
    if n_varying:
        log.error(f"STOP: {n_varying} MSOA11 areas have baseline characteristics that vary across years "
                  "(they must be fixed).")
        return 1

    log.info(f"Final panel: {n_rows:,} MSOA-year rows, {n_msoa:,} unique MSOA11 areas, all QA stops passed.")

    n_main_sample = con.sql("SELECT sum(CASE WHEN main_sample THEN 1 ELSE 0 END) FROM panel").fetchone()[0]
    n_complete_case = con.sql("SELECT sum(CASE WHEN analysis_complete_case THEN 1 ELSE 0 END) FROM panel").fetchone()[0]
    log.info(f"main_sample (2016-2023): {n_main_sample:,} rows; analysis_complete_case: {n_complete_case:,} rows")

    missingness = con.sql(
        """
        SELECT
            sum(CASE WHEN baseline_income_bhc_2011_12 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_income,
            sum(CASE WHEN imd_ex_housing IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_imd,
            sum(CASE WHEN social_rent_share_2011 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_tenure2011,
            sum(CASE WHEN population_density_2011 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_density,
            sum(CASE WHEN degree_share_2011 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_education,
            sum(CASE WHEN unemployment_rate_2011 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_unemployment,
            sum(CASE WHEN age_share_65plus_2011 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_age_structure,
            sum(CASE WHEN access_keyservices_pt_min_2014 IS NULL OR dist_town_centre_km IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_accessibility,
            sum(CASE WHEN greenbelt_share_2011 IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_greenbelt,
            sum(CASE WHEN sale_count IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_any_sales,
            sum(CASE WHEN lad23cd_analysis IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS pct_missing_lad
        FROM panel
        """
    ).df()
    missingness.to_csv(QA_DIR / "final_panel_missingness_summary.csv", index=False)
    log.info(f"Final panel missingness summary:\n{missingness.T}")

    # =================================================================
    # Write outputs
    # =================================================================
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    old_panel_cols = _existing_columns(PROCESSED_DIR / "final_msoa_year_dissertation_panel.parquet")
    old_base_cols = _existing_columns(PROCESSED_DIR / "msoa_baseline_characteristics.parquet")
    panel_df = con.sql("SELECT * FROM panel ORDER BY msoa11cd, year").df()
    panel_df.to_parquet(PROCESSED_DIR / "final_msoa_year_dissertation_panel.parquet", compression="zstd", index=False)
    panel_df.to_csv(PROCESSED_DIR / "final_msoa_year_dissertation_panel.csv", index=False)
    log.info(f"Wrote final_msoa_year_dissertation_panel.parquet + .csv ({len(panel_df):,} rows x {panel_df.shape[1]} cols)")
    _log_column_change("final panel", old_panel_cols, list(panel_df.columns))

    # --- Baseline characteristics: one row per MSOA11 ---
    con.execute(
        """
        CREATE OR REPLACE TABLE baseline_characteristics AS
        SELECT
            g.msoa11cd, mn.msoa11nm, g.lad23cd_analysis, ln.lad_name AS lad23nm_analysis, g.rgn_code AS region_code,
            tn.baseline_households_2011, dn.population_2011,
            inc.baseline_income_bhc_2011_12, inc.baseline_income_upper_ci, inc.baseline_income_lower_ci,
            inc.log_baseline_income,
            md.income_moderator_used, md.income_moderator_direction, md.income_moderator_value,
            md.income_z, md.income_quartile, md.low_income_q1, md.below_median_income,
            imd.imd2015_total_population, imd.imd2015_working_age_population,
            imd.imd2015_income_rate_msoa, imd.imd2015_employment_rate_msoa,
            imd.imd2015_education_score_pw, imd.imd2015_health_score_pw, imd.imd2015_crime_score_pw,
            imd.imd2015_barriers_score_pw, imd.imd2015_living_score_pw,
            imd.imd_ex_housing, imd.imd_ex_housing_living, imd.imd_ex_income_housing,
            imd.imd2015_overall_score_pw,
            imd.imd_ex_housing_bottom20_popshare, imd.imd2015_overall_bottom20_popshare_legacy,
            md.deprivation_moderator_used, md.deprivation_moderator_value,
            md.deprivation_z, md.deprivation_quartile, md.high_deprivation_q4,
            tn.social_rent_share_2011, tn.owner_share_2011, tn.private_rent_share_2011, tn.social_rent_z,
            dn.population_density_2011, dn.log_population_density_2011, dn.density_z,
            ed.degree_share_2011, ed.degree_share_z,
            un.unemployment_rate_2011, un.unemployment_z,
            ag.age_share_under16_2011, ag.age_share_16_24_2011, ag.age_share_25_44_2011,
            ag.age_share_45_64_2011, ag.age_share_65plus_2011,
            ag.age_share_under16_z, ag.age_share_16_24_z, ag.age_share_25_44_z,
            ag.age_share_45_64_z, ag.age_share_65plus_z,
            ac.access_keyservices_pt_min_2014,
            ac.access_keyservices_car_min_2014,
            ac.dist_town_centre_km,
            ac.access_keyservices_pt_z,
            ac.access_keyservices_car_z,
            ac.dist_town_centre_z,
            ac.log_access_keyservices_pt_min_2014,
            ac.log_access_keyservices_car_min_2014,
            ac.log_dist_town_centre_km,
            gbt.greenbelt_share_2011, gbt.greenbelt_share_z, gbt.greenbelt_any_2011,
            ss.* EXCLUDE (msoa11cd)
        FROM geog_v g
        LEFT JOIN msoa11_names mn ON g.msoa11cd = mn.msoa11cd
        LEFT JOIN lad_names ln ON g.lad23cd_analysis = ln.lad_code
        LEFT JOIN income_v inc ON g.msoa11cd = inc.msoa11cd
        LEFT JOIN imd_v imd ON g.msoa11cd = imd.msoa11cd
        LEFT JOIN mod_v md ON g.msoa11cd = md.msoa11cd
        LEFT JOIN tenure_v tn ON g.msoa11cd = tn.msoa11cd
        LEFT JOIN density_v dn ON g.msoa11cd = dn.msoa11cd
        LEFT JOIN education_v ed ON g.msoa11cd = ed.msoa11cd
        LEFT JOIN unemployment_v un ON g.msoa11cd = un.msoa11cd
        LEFT JOIN age_v ag ON g.msoa11cd = ag.msoa11cd
        LEFT JOIN access_v ac ON g.msoa11cd = ac.msoa11cd
        LEFT JOIN greenbelt_v gbt ON g.msoa11cd = gbt.msoa11cd
        LEFT JOIN spstat_v ss ON g.msoa11cd = ss.msoa11cd
        """
    )
    baseline_df = con.sql("SELECT * FROM baseline_characteristics ORDER BY msoa11cd").df()
    baseline_df.to_parquet(PROCESSED_DIR / "msoa_baseline_characteristics.parquet", compression="zstd", index=False)
    baseline_df.to_csv(PROCESSED_DIR / "msoa_baseline_characteristics.csv", index=False)
    log.info(f"Wrote msoa_baseline_characteristics.parquet + .csv ({len(baseline_df):,} rows x {baseline_df.shape[1]} cols)")
    _log_column_change("baseline characteristics", old_base_cols, list(baseline_df.columns))

    # deprivation quartile summary for the selected moderator (QA)
    dq = (baseline_df.groupby("deprivation_quartile")
          .agg(n=("msoa11cd", "size"), min_value=("deprivation_moderator_value", "min"),
               max_value=("deprivation_moderator_value", "max"),
               mean_value=("deprivation_moderator_value", "mean")).reset_index())
    dq.insert(0, "deprivation_moderator", dep_choice)
    dq.to_csv(QA_DIR / "deprivation_quartile_summary.csv", index=False)

    # --- Transaction-level robustness file (parquet only) ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE transactions_regression_ready AS
        SELECT
            t.transaction_id, t.transaction_date, t.transaction_year AS year,
            EXTRACT(month FROM t.transaction_date) AS month,
            t.msoa11cd, g.lad23cd_analysis,
            t.transaction_price AS price, t.total_floor_area AS floor_area,
            t.nominal_ppsqm AS ppsqm, t.log_ppsqm,
            t.property_type_ppd AS property_type,
            CASE WHEN t.duration = 'L' THEN true ELSE false END AS leasehold,
            t.construction_age_band AS construction_age,
            nb.newbuilds_lag1_per_1000, nb.newbuilds_prev3yr_per_1000,
            md.income_z, md.deprivation_z, tn.social_rent_z, dn.density_z
        FROM read_parquet('{(GEO_DIR / "transactions_england_2012_2023_trimmed_geo.parquet").as_posix()}') t
        LEFT JOIN geog_v g ON t.msoa11cd = g.msoa11cd
        LEFT JOIN newbuild_v nb ON t.msoa11cd = nb.msoa11cd AND t.transaction_year = nb.year
        LEFT JOIN mod_v md ON t.msoa11cd = md.msoa11cd
        LEFT JOIN tenure_v tn ON t.msoa11cd = tn.msoa11cd
        LEFT JOIN density_v dn ON t.msoa11cd = dn.msoa11cd
        WHERE t.msoa11cd IS NOT NULL
        """
    )
    n_trans = con.sql("SELECT count(*) FROM transactions_regression_ready").fetchone()[0]
    con.execute(
        f"""
        COPY transactions_regression_ready TO '{(PROCESSED_DIR / "transactions_regression_ready.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote transactions_regression_ready.parquet ({n_trans:,} rows, parquet-only)")

    # --- Also keep the full new-build/transaction detail files (already
    # England+MSOA11-keyed) for anyone who wants raw detail beyond the
    # regression-ready slice above ---
    con.execute(
        f"""
        COPY (SELECT * FROM read_parquet('{(GEO_DIR / "transactions_england_2012_2023_trimmed_geo.parquet").as_posix()}'))
        TO '{(PROCESSED_DIR / "transactions_analysis.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (SELECT * FROM read_parquet('{(GEO_DIR / "newbuilds_with_geography.parquet").as_posix()}'))
        TO '{(PROCESSED_DIR / "newbuilds_analysis.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info("Wrote transactions_analysis.parquet, newbuilds_analysis.parquet")
    log.info("Final panel merge complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
