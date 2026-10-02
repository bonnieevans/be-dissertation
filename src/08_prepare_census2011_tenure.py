"""Prepare the baseline social-rented-share moderator AND the fixed
baseline-household denominator from Census 2011 QS405EW (household tenure),
natively at MSOA11 - the PRIMARY analysis geography, no crosswalk needed.
England only.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("08_prepare_census2011_tenure")

QS405_CSV = PROJECT_ROOT / "data" / "raw" / "census2011" / "tenure_qs405ew" / "qs405ew_msoa11_tenure.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    if not QS405_CSV.exists():
        log.error(f"{QS405_CSV} not found.")
        return 1

    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE TABLE qs405_long AS SELECT * FROM read_csv_auto('{QS405_CSV.as_posix()}')")
    n_rows = con.sql("SELECT count(*) FROM qs405_long").fetchone()[0]
    n_msoa = con.sql("SELECT count(DISTINCT GEOGRAPHY_CODE) FROM qs405_long").fetchone()[0]
    log.info(f"QS405EW: {n_rows:,} rows for {n_msoa:,} MSOA11 areas")

    con.execute(
        """
        CREATE OR REPLACE TABLE tenure_msoa11 AS
        SELECT
            GEOGRAPHY_CODE AS msoa11cd,
            sum(CASE WHEN C_TENHUK11_NAME = 'All categories: Tenure' THEN OBS_VALUE ELSE 0 END) AS baseline_households_2011,
            sum(CASE WHEN C_TENHUK11_NAME = 'Owned: Total' THEN OBS_VALUE ELSE 0 END) AS owner_occupied_2011,
            sum(CASE WHEN C_TENHUK11_NAME = 'Shared ownership (part owned and part rented)' THEN OBS_VALUE ELSE 0 END) AS shared_ownership_2011,
            sum(CASE WHEN C_TENHUK11_NAME = 'Social rented: Total' THEN OBS_VALUE ELSE 0 END) AS social_rented_2011,
            sum(CASE WHEN C_TENHUK11_NAME = 'Private rented: Total' THEN OBS_VALUE ELSE 0 END) AS private_rented_2011
        FROM qs405_long
        WHERE GEOGRAPHY_CODE LIKE 'E02%'
        GROUP BY GEOGRAPHY_CODE
        """
    )
    n_msoa11 = con.sql("SELECT count(*) FROM tenure_msoa11").fetchone()[0]
    log.info(f"Census 2011 tenure: {n_msoa11:,} England MSOA11 areas")

    n_zero_households = con.sql("SELECT count(*) FROM tenure_msoa11 WHERE baseline_households_2011 <= 0").fetchone()[0]
    if n_zero_households:
        log.error(f"STOP: {n_zero_households} MSOA11 areas have baseline_households_2011 <= 0.")
        return 1

    con.execute(
        """
        CREATE OR REPLACE TABLE tenure_final AS
        SELECT
            *,
            owner_occupied_2011::DOUBLE / baseline_households_2011 AS owner_share_2011,
            social_rented_2011::DOUBLE / baseline_households_2011 AS social_rent_share_2011,
            private_rented_2011::DOUBLE / baseline_households_2011 AS private_rent_share_2011,
            (social_rented_2011::DOUBLE / baseline_households_2011
                - avg(social_rented_2011::DOUBLE / baseline_households_2011) OVER ())
                / stddev(social_rented_2011::DOUBLE / baseline_households_2011) OVER () AS social_rent_z,
            ntile(4) OVER (ORDER BY social_rented_2011::DOUBLE / baseline_households_2011) AS social_rent_quartile
        FROM tenure_msoa11
        """
    )

    summary = con.sql(
        """
        SELECT social_rent_quartile, count(*) AS n,
               min(social_rent_share_2011) AS min_share, max(social_rent_share_2011) AS max_share
        FROM tenure_final GROUP BY 1 ORDER BY 1
        """
    ).df()
    summary.to_csv(QA_DIR / "census2011_tenure_quartile_summary.csv", index=False)
    log.info(f"Social-rent-share quartile summary:\n{summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY tenure_final TO '{(OUT_DIR / "msoa11_census2011_tenure.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'msoa11_census2011_tenure.parquet'} ({n_msoa11:,} rows)")
    log.info("Census 2011 tenure preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
