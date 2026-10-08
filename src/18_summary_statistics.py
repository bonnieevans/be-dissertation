"""Descriptive tables for the final panel:
- Table 1: overall summary statistics -> outputs/tables/summary_statistics.csv
- Table 3: construction intensity by baseline income quartile ->
  outputs/tables/construction_by_income_quartile.csv
- Table 4: construction intensity by deprivation quartile (of the selected
  deprivation moderator, config moderators.deprivation_moderator) ->
  outputs/tables/construction_by_deprivation_quartile.csv
(Table 2 "neighbourhood baseline characteristics" is
data/processed/msoa_baseline_characteristics.parquet itself, exposed as the
`baseline_characteristics` Excel sheet in 20_export_excel.py. Table 5 is the
correlation matrix from 15_run_diagnostics.py.)
"""

from __future__ import annotations

import sys

import pandas as pd

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("18_summary_statistics")

FINAL_PANEL = PROJECT_ROOT / "data" / "processed" / "final_msoa_year_dissertation_panel.parquet"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

SUMMARY_VARS = [
    "sale_count", "median_price", "median_ppsqm", "log_median_ppsqm",
    "newbuild_total", "newbuild_houses", "newbuild_flats",
    "newbuilds_per_1000", "newbuilds_lag1_per_1000", "newbuilds_prev3yr_per_1000",
    "baseline_income_bhc_2011_12", "log_baseline_income",
    "imd_ex_housing", "imd_ex_housing_living", "imd_ex_income_housing", "imd2015_income_rate_msoa",
    "imd2015_overall_score_pw", "imd_ex_housing_bottom20_popshare", "deprivation_moderator_value",
    "social_rent_share_2011", "population_density_2011", "log_population_density_2011",
    "degree_share_2011", "unemployment_rate_2011",
    "age_share_under16_2011", "age_share_16_24_2011", "age_share_25_44_2011", "age_share_45_64_2011", "age_share_65plus_2011",
    "access_keyservices_pt_min_2014", "access_keyservices_car_min_2014", "dist_town_centre_km",
    "detached_sale_share", "flat_sale_share", "leasehold_sale_share",
]


def main() -> int:
    if not FINAL_PANEL.exists():
        log.error(f"{FINAL_PANEL} not found - run src/14_merge_final_panel.py first.")
        return 1

    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{FINAL_PANEL.as_posix()}')")

    rows = []
    for var in SUMMARY_VARS:
        stats = con.sql(
            f"""
            SELECT count("{var}") AS n, avg("{var}") AS mean, stddev("{var}") AS sd,
                   min("{var}") AS min, quantile_cont("{var}", 0.5) AS median, max("{var}") AS max
            FROM panel
            """
        ).df().iloc[0]
        rows.append({"variable": var, **stats.to_dict()})

    summary_df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(TABLES_DIR / "summary_statistics.csv", index=False)
    log.info(f"Wrote Table 1 (summary statistics, {len(SUMMARY_VARS)} variables) -> {TABLES_DIR / 'summary_statistics.csv'}")

    # --- Table 3: construction by income quartile ---
    by_income_q = con.sql(
        """
        SELECT income_quartile,
               count(*) AS n_msoa_years,
               avg(newbuilds_per_1000) AS mean_newbuilds_per_1000,
               avg(newbuilds_lag1_per_1000) AS mean_newbuilds_lag1_per_1000,
               sum(CASE WHEN newbuild_total = 0 THEN 1 ELSE 0 END)::DOUBLE / count(*) AS share_zero_construction,
               avg(log_median_ppsqm) AS mean_log_median_ppsqm
        FROM panel
        WHERE income_quartile IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_income_q.to_csv(TABLES_DIR / "construction_by_income_quartile.csv", index=False)
    log.info(f"Table 3 (construction by income quartile):\n{by_income_q}")

    # --- Table 4: construction by deprivation quartile ---
    by_dep_q = con.sql(
        """
        SELECT deprivation_quartile,
               count(*) AS n_msoa_years,
               avg(newbuilds_per_1000) AS mean_newbuilds_per_1000,
               avg(newbuilds_lag1_per_1000) AS mean_newbuilds_lag1_per_1000,
               sum(CASE WHEN newbuild_total = 0 THEN 1 ELSE 0 END)::DOUBLE / count(*) AS share_zero_construction,
               avg(log_median_ppsqm) AS mean_log_median_ppsqm
        FROM panel
        WHERE deprivation_quartile IS NOT NULL
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_dep_q.to_csv(TABLES_DIR / "construction_by_deprivation_quartile.csv", index=False)
    log.info(f"Table 4 (construction by deprivation quartile):\n{by_dep_q}")

    log.info("Summary statistics complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
