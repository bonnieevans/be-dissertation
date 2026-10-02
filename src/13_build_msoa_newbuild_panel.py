"""Aggregate new-build dwellings to England MSOA11CD x year (MSOA11 is the
PRIMARY analysis geography), normalise against the fixed Census 2011
baseline_households_2011 denominator, and construct lagged exposure
measures.

Per the brief, EPC tenure is NOT used to define the treatment here (kept
only in newbuilds_with_geography.parquet for QA/descriptive purposes - see
outputs/excel/unknown_tenure_investigation.xlsx).
newbuilds_lag1_per_1000 is the PRIMARY treatment;
newbuilds_prev3yr_per_1000 is the primary robustness variant.
Interaction terms are NOT precomputed here - build them at regression time.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("13_build_msoa_newbuild_panel")

NEWBUILDS_GEO = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "newbuilds_with_geography.parquet"
CENSUS_TENURE = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_census2011_tenure.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def main() -> int:
    cfg = load_config()
    start_year = cfg["study_period"]["main_panel_start_year"]
    end_year = cfg["study_period"]["main_panel_end_year"]

    if not NEWBUILDS_GEO.exists():
        log.error(f"{NEWBUILDS_GEO} not found - run src/05_attach_geography.py first.")
        return 1
    if not CENSUS_TENURE.exists():
        log.error(f"{CENSUS_TENURE} not found - run src/08_prepare_census2011_tenure.py first.")
        return 1

    con = get_duckdb_connection()

    con.execute(
        f"""
        CREATE OR REPLACE TABLE newbuild_year_counts AS
        SELECT
            msoa11cd,
            newbuild_year AS year,
            count(*) AS newbuild_total,
            median(newbuild_floor_area) AS median_newbuild_floor_area,
            avg(newbuild_floor_area) AS mean_newbuild_floor_area,
            sum(CASE WHEN newbuild_property_type IN ('House', 'Bungalow') THEN 1 ELSE 0 END) AS newbuild_houses,
            sum(CASE WHEN newbuild_property_type = 'Flat' THEN 1 ELSE 0 END) AS newbuild_flats
        FROM read_parquet('{NEWBUILDS_GEO.as_posix()}')
        WHERE msoa11cd IS NOT NULL
        GROUP BY msoa11cd, newbuild_year
        """
    )

    # --- Full MSOA x year grid (2012-2023 plus lookback years for lags) x
    # every MSOA11 with a Census 2011 baseline record, so MSOA-years with
    # zero development are explicit zero rows, not missing ---
    con.execute(f"CREATE OR REPLACE VIEW census_tenure AS SELECT * FROM read_parquet('{CENSUS_TENURE.as_posix()}')")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE msoa_year_grid AS
        SELECT m.msoa11cd, y.year
        FROM (SELECT DISTINCT msoa11cd FROM census_tenure) m
        CROSS JOIN (SELECT unnest(generate_series({start_year} - 3, {end_year})) AS year) y
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE newbuild_panel_full AS
        SELECT
            g.msoa11cd,
            g.year,
            coalesce(n.newbuild_total, 0) AS newbuild_total,
            coalesce(n.newbuild_houses, 0) AS newbuild_houses,
            coalesce(n.newbuild_flats, 0) AS newbuild_flats,
            n.median_newbuild_floor_area,
            n.mean_newbuild_floor_area,
            c.baseline_households_2011
        FROM msoa_year_grid g
        LEFT JOIN newbuild_year_counts n ON g.msoa11cd = n.msoa11cd AND g.year = n.year
        LEFT JOIN census_tenure c ON g.msoa11cd = c.msoa11cd
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE newbuild_panel_norm AS
        SELECT
            *,
            1000.0 * newbuild_total / NULLIF(baseline_households_2011, 0) AS newbuilds_per_1000
        FROM newbuild_panel_full
        """
    )

    # --- Lagged exposure: lag1/2/3 individually, PLUS prev3yr = lag1+lag2+lag3 ---
    con.execute(
        """
        CREATE OR REPLACE TABLE newbuild_panel_lagged AS
        SELECT
            *,
            lag(newbuild_total, 1) OVER w AS newbuilds_lag1,
            lag(newbuilds_per_1000, 1) OVER w AS newbuilds_lag1_per_1000,
            lag(newbuild_total, 2) OVER w AS newbuilds_lag2,
            lag(newbuilds_per_1000, 2) OVER w AS newbuilds_lag2_per_1000,
            lag(newbuild_total, 3) OVER w AS newbuilds_lag3,
            lag(newbuilds_per_1000, 3) OVER w AS newbuilds_lag3_per_1000,
            (coalesce(lag(newbuild_total, 1) OVER w, 0) + coalesce(lag(newbuild_total, 2) OVER w, 0)
                + coalesce(lag(newbuild_total, 3) OVER w, 0)) AS newbuilds_prev3yr,
            (coalesce(lag(newbuilds_per_1000, 1) OVER w, 0) + coalesce(lag(newbuilds_per_1000, 2) OVER w, 0)
                + coalesce(lag(newbuilds_per_1000, 3) OVER w, 0)) AS newbuilds_prev3yr_per_1000
        FROM newbuild_panel_norm
        WINDOW w AS (PARTITION BY msoa11cd ORDER BY year)
        """
    )

    # Keep only the requested study years (the pre-start years existed only to compute lags)
    con.execute(f"CREATE OR REPLACE TABLE newbuild_panel_final AS SELECT * FROM newbuild_panel_lagged WHERE year BETWEEN {start_year} AND {end_year}")

    n_rows = con.sql("SELECT count(*) FROM newbuild_panel_final").fetchone()[0]
    n_msoa = con.sql("SELECT count(DISTINCT msoa11cd) FROM newbuild_panel_final").fetchone()[0]
    log.info(f"New-build panel: {n_rows:,} MSOA-year rows across {n_msoa:,} MSOAs")

    zero_construction = con.sql(
        "SELECT sum(CASE WHEN newbuild_total = 0 THEN 1 ELSE 0 END)::DOUBLE / count(*) FROM newbuild_panel_final"
    ).fetchone()[0]
    log.info(f"MSOA-years with zero construction: {zero_construction:.2%}")

    QA_DIR.mkdir(parents=True, exist_ok=True)
    con.sql(f"SELECT {zero_construction} AS share_zero_construction").df().to_csv(
        QA_DIR / "newbuild_panel_zero_construction_share.csv", index=False
    )

    intensity_desc = con.sql(
        "SELECT newbuilds_per_1000 FROM newbuild_panel_final"
    ).df()["newbuilds_per_1000"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    intensity_desc.to_csv(QA_DIR / "newbuild_intensity_distribution.csv")
    log.info(f"newbuilds_per_1000 distribution:\n{intensity_desc}")

    # --- QA per the brief: reproduce construction intensity from raw counts ---
    reproduced = con.sql(
        """
        SELECT max(abs(newbuilds_per_1000 - 1000.0 * newbuild_total / NULLIF(baseline_households_2011, 0))) AS max_discrepancy
        FROM newbuild_panel_final WHERE baseline_households_2011 > 0
        """
    ).fetchone()[0]
    if reproduced is not None and reproduced > 1e-6:
        log.error(f"STOP: newbuilds_per_1000 does not reproduce from raw counts (max discrepancy {reproduced}).")
        return 1
    log.info("QA: newbuilds_per_1000 reproduces exactly from newbuild_total/baseline_households_2011.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY newbuild_panel_final TO '{(OUT_DIR / "msoa_year_newbuild_panel.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'msoa_year_newbuild_panel.parquet'}")
    log.info("MSOA new-build panel construction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
