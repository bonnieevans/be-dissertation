"""Aggregate established-home transactions to the England MSOA11CD x year
level (2012-2023) - MSOA11 is the PRIMARY analysis geography. Primary panel
uses the trimmed sample; the untrimmed version remains available for
robustness.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("12_build_msoa_price_panel")

GEO_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"

PRICE_PANEL_SQL = """
SELECT
    msoa11cd,
    transaction_year AS year,
    count(*) AS sale_count,
    median(transaction_price) AS median_price,
    median(nominal_ppsqm) AS median_ppsqm,
    avg(nominal_ppsqm) AS mean_ppsqm,
    avg(log_ppsqm) AS mean_log_ppsqm,
    quantile_cont(nominal_ppsqm, 0.25) AS p25_ppsqm,
    quantile_cont(nominal_ppsqm, 0.75) AS p75_ppsqm,
    median(total_floor_area) AS median_floor_area,
    sum(CASE WHEN property_type_ppd = 'D' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS detached_sale_share,
    sum(CASE WHEN property_type_ppd = 'S' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS semidetached_sale_share,
    sum(CASE WHEN property_type_ppd = 'T' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS terraced_sale_share,
    sum(CASE WHEN property_type_ppd = 'F' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS flat_sale_share,
    sum(CASE WHEN duration = 'L' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS leasehold_sale_share
FROM read_parquet('{path}')
WHERE msoa11cd IS NOT NULL
GROUP BY msoa11cd, transaction_year
"""


def build_panel(con, parquet_path, thresholds: list[int]):
    con.execute(f"CREATE OR REPLACE TABLE _panel_raw AS {PRICE_PANEL_SQL.format(path=parquet_path.as_posix())}")
    flag_cols = ",\n               ".join(
        f"CASE WHEN sale_count < {t} THEN 1 ELSE 0 END AS sales_under_{t}" for t in thresholds
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE _panel AS
        SELECT *,
               ln(median_ppsqm) AS log_median_ppsqm,
               {flag_cols}
        FROM _panel_raw
        """
    )
    return con.sql("SELECT * FROM _panel").df()


def main() -> int:
    cfg = load_config()
    thresholds = cfg["transactions"]["low_sales_count_thresholds"]

    con = get_duckdb_connection()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("transactions_england_2012_2023_trimmed_geo.parquet", "msoa_year_price_panel_trimmed.parquet"),
        ("transactions_england_2012_2023_untrimmed_geo.parquet", "msoa_year_price_panel_untrimmed.parquet"),
    ]

    for in_name, out_name in jobs:
        in_path = GEO_DIR / in_name
        if not in_path.exists():
            log.error(f"{in_path} not found - run src/05_attach_geography.py first.")
            return 1
        df = build_panel(con, in_path, thresholds)
        out_path = OUT_DIR / out_name
        con.sql("SELECT * FROM _panel").write_parquet(str(out_path), compression="zstd")
        n_msoa = df["msoa11cd"].nunique()
        log.info(f"{out_name}: {len(df):,} MSOA-year rows, {n_msoa:,} MSOAs")
        for t in thresholds:
            n_under = int(df[f"sales_under_{t}"].sum())
            log.info(f"  sales_under_{t}: {n_under:,} ({n_under/len(df)*100:.1f}%)")

    # QA: sale-count distribution + percentiles on the primary (trimmed) panel
    main_df = con.sql(
        f"SELECT * FROM read_parquet('{(OUT_DIR / 'msoa_year_price_panel_trimmed.parquet').as_posix()}')"
    ).df()
    sale_count_desc = main_df["sale_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    sale_count_desc.to_csv(QA_DIR / "price_panel_sale_count_distribution.csv")
    log.info(f"sale_count distribution (trimmed panel):\n{sale_count_desc}")

    log.info("MSOA price panel construction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
