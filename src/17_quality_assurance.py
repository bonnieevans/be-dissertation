"""Consolidate the quality-assurance checklist into the three files the
brief names explicitly: data_quality_summary.csv, geography_match_summary.csv,
transaction_count_summary.csv. Most of the underlying numbers were already
computed and logged by the scripts that produced each dataset (02-14); this
script re-derives the specific summary shapes the brief asks for from the
already-cleaned interim/processed parquet files (not the raw 23GB CSVs).
"""

from __future__ import annotations

import sys

import pandas as pd

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("17_quality_assurance")

GEO_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
NEWBUILDS_GEO = GEO_DIR / "newbuilds_with_geography.parquet"
TRANS_TRIMMED = GEO_DIR / "transactions_england_2012_2023_trimmed_geo.parquet"
FINAL_PANEL = PROJECT_ROOT / "data" / "processed" / "final_msoa_year_dissertation_panel.parquet"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    cfg = load_config()
    thresholds = cfg["transactions"]["low_sales_count_thresholds"]

    for p in (NEWBUILDS_GEO, TRANS_TRIMMED, FINAL_PANEL):
        if not p.exists():
            log.error(f"{p} not found - run the preceding pipeline scripts first.")
            return 1

    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE VIEW trans AS SELECT * FROM read_parquet('{TRANS_TRIMMED.as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW nb AS SELECT * FROM read_parquet('{NEWBUILDS_GEO.as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{FINAL_PANEL.as_posix()}')")

    # =================================================================
    # transaction_count_summary.csv
    # =================================================================
    n_transactions_established = con.sql("SELECT count(*) FROM trans").fetchone()[0]
    sale_count_pctiles = con.sql(
        """
        SELECT quantile_cont(sale_count, 0.10) AS p10, quantile_cont(sale_count, 0.25) AS p25,
               quantile_cont(sale_count, 0.50) AS p50, quantile_cont(sale_count, 0.75) AS p75,
               quantile_cont(sale_count, 0.90) AS p90, median(sale_count) AS median_transactions_per_msoa_year
        FROM panel
        """
    ).df()
    threshold_shares = {}
    for t in thresholds:
        threshold_shares[f"share_msoa_years_sales_under_{t}"] = con.sql(
            f"SELECT avg(sales_under_{t}::INT) FROM panel"
        ).fetchone()[0]

    transaction_count_summary = pd.DataFrame([{
        "n_transactions_established_trimmed": n_transactions_established,
        **sale_count_pctiles.iloc[0].to_dict(),
        **threshold_shares,
    }])
    transaction_count_summary.to_csv(QA_DIR / "transaction_count_summary.csv", index=False)
    log.info(f"transaction_count_summary.csv:\n{transaction_count_summary.T}")

    # =================================================================
    # geography_match_summary.csv
    # =================================================================
    # newbuilds_with_geography.parquet is already restricted to
    # MSOA11-matched + England rows (see 05_attach_geography.py), so a
    # match rate computed from it alone would trivially be 1.0. The true
    # pre-restriction E&W match rates (both MSOA11 via postcode, and MSOA21
    # direct) were computed there and saved per-year; re-derive the overall
    # figures from that file.
    by_year_match = pd.read_csv(QA_DIR / "newbuild_geography_match_rate_by_year.csv")
    n_newbuilds_ew = int(by_year_match["n_newbuilds"].sum())
    n_matched_msoa11_ew = int(by_year_match["n_matched_msoa11"].sum())
    n_matched_msoa21_ew = int(by_year_match["n_matched_msoa21"].sum())

    n_newbuilds_england = con.sql("SELECT count(*) FROM nb").fetchone()[0]
    n_transactions_geo = con.sql("SELECT count(*) FROM trans").fetchone()[0]

    trans_geo_summary = pd.read_csv(QA_DIR / "transactions_geography_match_rate.csv")

    geography_match_summary = pd.DataFrame([{
        "n_newbuild_uprns_total_ew": n_newbuilds_ew,
        "n_newbuild_uprns_matched_msoa11_ew": n_matched_msoa11_ew,
        "newbuild_uprn_match_rate_msoa11_ew": n_matched_msoa11_ew / n_newbuilds_ew,
        "n_newbuild_uprns_matched_msoa21_ew": n_matched_msoa21_ew,
        "newbuild_uprn_match_rate_msoa21_ew": n_matched_msoa21_ew / n_newbuilds_ew,
        "n_newbuild_uprns_england_final": n_newbuilds_england,
        "n_transactions_geocoded_england": n_transactions_geo,
        "transaction_postcode_to_msoa11_match_rate": trans_geo_summary["match_rate_msoa11"].mean(),
        "transaction_postcode_to_msoa21_match_rate": trans_geo_summary["match_rate_msoa21"].mean(),
    }])
    geography_match_summary.to_csv(QA_DIR / "geography_match_summary.csv", index=False)
    log.info(f"geography_match_summary.csv:\n{geography_match_summary.T}")

    # =================================================================
    # data_quality_summary.csv - one row per baseline characteristic
    # =================================================================
    checks = []
    for label, col in [
        ("baseline_income", "baseline_income_bhc_2011_12"),
        ("imd_ex_housing (primary deprivation moderator)", "imd_ex_housing"),
        ("imd2015_income_rate_msoa", "imd2015_income_rate_msoa"),
        ("deprivation_moderator_value (selected)", "deprivation_moderator_value"),
        ("income_moderator_value (selected)", "income_moderator_value"),
        ("census2011_tenure", "social_rent_share_2011"),
        ("census2011_density", "population_density_2011"),
        ("census2011_education", "degree_share_2011"),
        ("census2011_unemployment", "unemployment_rate_2011"),
    ]:
        row = con.sql(
            f"""
            SELECT count(*) AS n_msoa_years,
                   sum(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS n_missing,
                   sum(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS missing_rate,
                   min("{col}") AS min_value, max("{col}") AS max_value
            FROM panel
            """
        ).df().iloc[0].to_dict()
        checks.append({"characteristic": label, **row})
    data_quality_summary = pd.DataFrame(checks)
    data_quality_summary.to_csv(QA_DIR / "data_quality_summary.csv", index=False)
    log.info(f"data_quality_summary.csv:\n{data_quality_summary}")

    log.info("Quality assurance summary complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
