"""Prepare the baseline population-density moderator from Census 2011
QS102EW, natively at MSOA11 - the PRIMARY analysis geography, no crosswalk
needed. England only.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("09_prepare_census2011_density")

QS102_CSV = PROJECT_ROOT / "data" / "raw" / "census2011" / "density_qs102ew" / "qs102ew_msoa11_density.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    if not QS102_CSV.exists():
        log.error(f"{QS102_CSV} not found.")
        return 1

    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE TABLE qs102_long AS SELECT * FROM read_csv_auto('{QS102_CSV.as_posix()}')")

    con.execute(
        """
        CREATE OR REPLACE TABLE density_msoa11 AS
        SELECT
            GEOGRAPHY_CODE AS msoa11cd,
            sum(CASE WHEN CELL_NAME = 'All usual residents' THEN OBS_VALUE ELSE 0 END) AS population_2011,
            sum(CASE WHEN CELL_NAME = 'Area Hectares' THEN OBS_VALUE ELSE 0 END) AS area_hectares_2011
        FROM qs102_long
        WHERE GEOGRAPHY_CODE LIKE 'E02%'
        GROUP BY GEOGRAPHY_CODE
        """
    )
    n_msoa11 = con.sql("SELECT count(*) FROM density_msoa11").fetchone()[0]
    log.info(f"QS102EW aggregated to {n_msoa11:,} England MSOA11 areas")

    con.execute(
        """
        CREATE OR REPLACE TABLE density_final AS
        SELECT
            msoa11cd, population_2011, area_hectares_2011,
            area_hectares_2011 / 100.0 AS area_km2,
            population_2011 / (area_hectares_2011 / 100.0) AS population_density_2011
        FROM density_msoa11
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE density_final AS
        SELECT
            *,
            ln(population_density_2011) AS log_population_density_2011,
            (ln(population_density_2011) - avg(ln(population_density_2011)) OVER ())
                / stddev(ln(population_density_2011)) OVER () AS density_z,
            ntile(4) OVER (ORDER BY population_density_2011) AS density_quartile
        FROM density_final
        """
    )

    summary = con.sql(
        """
        SELECT density_quartile, count(*) AS n,
               min(population_density_2011) AS min_density, max(population_density_2011) AS max_density
        FROM density_final GROUP BY 1 ORDER BY 1
        """
    ).df()
    summary.to_csv(QA_DIR / "census2011_density_quartile_summary.csv", index=False)
    log.info(f"Density quartile summary (persons/km2):\n{summary}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY density_final TO '{(OUT_DIR / "msoa11_census2011_density.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'msoa11_census2011_density.parquet'} ({n_msoa11:,} rows)")
    log.info("Census 2011 density preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
