"""Optional moderator: Census 2011 QS501EW (highest level of qualification),
natively at MSOA11 - the PRIMARY analysis geography, no crosswalk needed.
England only.

degree_share_2011 = residents aged 16+ with Level 4+ qualifications /
all residents aged 16+ with a qualification classification.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("10_prepare_census2011_education")

QS501_CSV = PROJECT_ROOT / "data" / "raw" / "census2011" / "education_qs501ew" / "qs501ew_msoa11_education.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    if not QS501_CSV.exists():
        log.error(f"{QS501_CSV} not found.")
        return 1

    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE TABLE qs501_long AS SELECT * FROM read_csv_auto('{QS501_CSV.as_posix()}')")

    con.execute(
        """
        CREATE OR REPLACE TABLE education_msoa11 AS
        SELECT
            GEOGRAPHY_CODE AS msoa11cd,
            sum(CASE WHEN CELL_NAME = 'All categories: Highest level of qualification' THEN OBS_VALUE ELSE 0 END) AS residents_16plus,
            sum(CASE WHEN CELL_NAME = 'Level 4 qualifications and above' THEN OBS_VALUE ELSE 0 END) AS residents_level4_plus,
            sum(CASE WHEN CELL_NAME = 'Level 4 qualifications and above' THEN OBS_VALUE ELSE 0 END)::DOUBLE
                / NULLIF(sum(CASE WHEN CELL_NAME = 'All categories: Highest level of qualification' THEN OBS_VALUE ELSE 0 END), 0) AS degree_share_2011
        FROM qs501_long
        WHERE GEOGRAPHY_CODE LIKE 'E02%'
        GROUP BY GEOGRAPHY_CODE
        """
    )
    n_msoa11 = con.sql("SELECT count(*) FROM education_msoa11").fetchone()[0]
    log.info(f"Education: {n_msoa11:,} England MSOA11 areas")

    con.execute(
        """
        CREATE OR REPLACE TABLE education_final AS
        SELECT
            *,
            (degree_share_2011 - avg(degree_share_2011) OVER ()) / stddev(degree_share_2011) OVER () AS degree_share_z,
            ntile(4) OVER (ORDER BY degree_share_2011) AS degree_share_quartile
        FROM education_msoa11
        """
    )

    summary = con.sql(
        """
        SELECT degree_share_quartile, count(*) AS n,
               min(degree_share_2011) AS min_share, max(degree_share_2011) AS max_share
        FROM education_final GROUP BY 1 ORDER BY 1
        """
    ).df()
    summary.to_csv(QA_DIR / "census2011_education_quartile_summary.csv", index=False)
    log.info(f"Degree-share quartile summary:\n{summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY education_final TO '{(OUT_DIR / "msoa11_census2011_education.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'msoa11_census2011_education.parquet'} ({n_msoa11:,} rows)")
    log.info("Census 2011 education preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
