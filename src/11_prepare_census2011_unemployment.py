"""Optional moderator: Census 2011 QS601EW (economic activity), natively at
MSOA11 - the PRIMARY analysis geography, no crosswalk needed. England only.

unemployment_rate_2011 = unemployed / economically active (the brief's
stated preference), NOT unemployed / total population.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("11_prepare_census2011_unemployment")

QS601_CSV = PROJECT_ROOT / "data" / "raw" / "census2011" / "unemployment_qs601ew" / "qs601ew_msoa11_unemployment.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    if not QS601_CSV.exists():
        log.error(f"{QS601_CSV} not found.")
        return 1

    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE TABLE qs601_long AS SELECT * FROM read_csv_auto('{QS601_CSV.as_posix()}')")

    con.execute(
        """
        CREATE OR REPLACE TABLE unemployment_msoa11 AS
        SELECT
            GEOGRAPHY_CODE AS msoa11cd,
            sum(CASE WHEN CELL_NAME = 'Economically active: Total' THEN OBS_VALUE ELSE 0 END) AS economically_active,
            sum(CASE WHEN CELL_NAME = 'Economically active: Unemployed' THEN OBS_VALUE ELSE 0 END) AS unemployed,
            sum(CASE WHEN CELL_NAME = 'Economically active: Unemployed' THEN OBS_VALUE ELSE 0 END)::DOUBLE
                / NULLIF(sum(CASE WHEN CELL_NAME = 'Economically active: Total' THEN OBS_VALUE ELSE 0 END), 0) AS unemployment_rate_2011
        FROM qs601_long
        WHERE GEOGRAPHY_CODE LIKE 'E02%'
        GROUP BY GEOGRAPHY_CODE
        """
    )
    n_msoa11 = con.sql("SELECT count(*) FROM unemployment_msoa11").fetchone()[0]
    log.info(f"Unemployment: {n_msoa11:,} England MSOA11 areas")

    con.execute(
        """
        CREATE OR REPLACE TABLE unemployment_final AS
        SELECT
            *,
            (unemployment_rate_2011 - avg(unemployment_rate_2011) OVER ()) / stddev(unemployment_rate_2011) OVER () AS unemployment_z,
            ntile(4) OVER (ORDER BY unemployment_rate_2011) AS unemployment_quartile
        FROM unemployment_msoa11
        """
    )

    summary = con.sql(
        """
        SELECT unemployment_quartile, count(*) AS n,
               min(unemployment_rate_2011) AS min_rate, max(unemployment_rate_2011) AS max_rate
        FROM unemployment_final GROUP BY 1 ORDER BY 1
        """
    ).df()
    summary.to_csv(QA_DIR / "census2011_unemployment_quartile_summary.csv", index=False)
    log.info(f"Unemployment-rate quartile summary:\n{summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY unemployment_final TO '{(OUT_DIR / "msoa11_census2011_unemployment.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'msoa11_census2011_unemployment.parquet'} ({n_msoa11:,} rows)")
    log.info("Census 2011 unemployment preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
