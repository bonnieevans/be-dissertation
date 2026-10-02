"""Prepare the baseline income moderator: ONS small-area income estimates,
financial year ending 2012 (2011/12), before housing costs, natively at
MSOA11 - the PRIMARY analysis geography, so no crosswalk is needed at all.
Treated as a FIXED baseline characteristic, never updated across years.
England only; quartiles/z-scores computed on the England-only distribution.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("06_prepare_income")

INCOME_XLS = PROJECT_ROOT / "data" / "raw" / "income" / "ons_small_area_income" / "income_fye2012.xls"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"

# ONS publishes this income series rounded to the nearest GBP 10/week, so
# dozens of MSOA11s legitimately sharing the same rounded value is NORMAL
# (confirmed by inspection: ~83 distinct values across 6791 MSOA11s, max
# single-value share ~4%). A failed geography merge (e.g. a join that
# fanned out or silently defaulted many rows to one source row) would
# instead show as one value dominating an implausibly large share of all
# rows - that is what this threshold actually detects.
MAX_ACCEPTABLE_SINGLE_VALUE_SHARE = 0.20


def main() -> int:
    if not INCOME_XLS.exists():
        log.error(f"{INCOME_XLS} not found.")
        return 1

    import xlrd

    wb = xlrd.open_workbook(str(INCOME_XLS))
    ws = wb.sheet_by_name("Net income before housing costs")
    header_row = 4  # confirmed by direct inspection: row 0-3 are title/blank, row 4 is the header
    rows = []
    for r in range(header_row + 1, ws.nrows):
        vals = ws.row_values(r)
        if not vals[0] or not str(vals[0])[:3] == "E02" or not isinstance(vals[6], (int, float)):
            continue  # England only; skips blank rows and the trailing "Source: ONS" footer row
        rows.append(
            {
                "msoa11cd": vals[0],
                "msoa_name": vals[1],
                "lad_code": vals[2],
                "lad_name": vals[3],
                "region_code": vals[4],
                "region_name": vals[5],
                "net_weekly_income_bhc": vals[6],
                "upper_ci_weekly": vals[7] if isinstance(vals[7], (int, float)) else None,
                "lower_ci_weekly": vals[8] if isinstance(vals[8], (int, float)) else None,
            }
        )
    log.info(f"Read {len(rows)} England MSOA11 rows from {INCOME_XLS.name}")

    import pandas as pd

    income_df = pd.DataFrame(rows)

    # --- QA per the brief: one row per MSOA11, no duplicate keys ---
    n_msoa11 = income_df["msoa11cd"].nunique()
    n_rows = len(income_df)
    if n_msoa11 != n_rows:
        log.error(f"STOP: {n_rows - n_msoa11} duplicate msoa11cd rows in the income source (expected one row each).")
        return 1

    # --- QA per the brief: flag suspiciously widespread identical income
    # values (a failed-geography-merge symptom) ---
    value_counts = income_df["net_weekly_income_bhc"].value_counts()
    n_distinct_values = len(value_counts)
    top_value, top_count = value_counts.index[0], int(value_counts.iloc[0])
    top_share = top_count / n_msoa11
    log.info(f"Income duplicate-value check: {n_distinct_values} distinct rounded income values across "
              f"{n_msoa11} MSOA11s; most common value {top_value} appears {top_count} times ({top_share:.2%}). "
              "ONS rounds this series to the nearest GBP 10/week, so many ties are expected.")
    if top_share > MAX_ACCEPTABLE_SINGLE_VALUE_SHARE:
        log.error(
            f"STOP: a single income value covers {top_share:.2%} of all MSOA11s, exceeding the "
            f"{MAX_ACCEPTABLE_SINGLE_VALUE_SHARE:.0%} threshold - this looks like a failed geography merge "
            "(e.g. many rows defaulting to one source value), not genuine ONS rounding. Investigate before proceeding."
        )
        return 1

    income_df["baseline_income_bhc_2011_12"] = income_df["net_weekly_income_bhc"] * 52
    income_df["baseline_income_upper_ci"] = income_df["upper_ci_weekly"] * 52
    income_df["baseline_income_lower_ci"] = income_df["lower_ci_weekly"] * 52

    n_missing = income_df["baseline_income_bhc_2011_12"].isna().sum()
    log.info(f"Income MSOA11 rows: {n_msoa11}, missing income: {n_missing}")

    con = get_duckdb_connection()
    con.register("income_msoa11", income_df[["msoa11cd", "baseline_income_bhc_2011_12",
                                              "baseline_income_upper_ci", "baseline_income_lower_ci"]])

    con.execute(
        """
        CREATE OR REPLACE TABLE income_final AS
        SELECT
            msoa11cd,
            baseline_income_bhc_2011_12,
            baseline_income_upper_ci,
            baseline_income_lower_ci,
            ln(baseline_income_bhc_2011_12) AS log_baseline_income,
            (baseline_income_bhc_2011_12 - avg(baseline_income_bhc_2011_12) OVER ())
                / stddev(baseline_income_bhc_2011_12) OVER () AS income_z,
            ntile(4) OVER (ORDER BY baseline_income_bhc_2011_12) AS income_quartile,
            CASE WHEN ntile(4) OVER (ORDER BY baseline_income_bhc_2011_12) = 1 THEN 1 ELSE 0 END AS low_income_q1,
            CASE WHEN baseline_income_bhc_2011_12 < median(baseline_income_bhc_2011_12) OVER () THEN 1 ELSE 0 END AS below_median_income
        FROM income_msoa11
        """
    )

    quartile_summary = con.sql(
        """
        SELECT income_quartile, count(*) AS n, min(baseline_income_bhc_2011_12) AS min_income,
               max(baseline_income_bhc_2011_12) AS max_income
        FROM income_final GROUP BY 1 ORDER BY 1
        """
    ).df()
    quartile_summary.to_csv(QA_DIR / "income_quartile_summary.csv", index=False)
    log.info(f"Baseline income quartile summary:\n{quartile_summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY income_final TO '{(OUT_DIR / "msoa11_baseline_income.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'msoa11_baseline_income.parquet'} ({n_msoa11} MSOA11 rows)")
    log.info("Income preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
