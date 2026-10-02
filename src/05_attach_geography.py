"""Attach geography to both new-builds (via UPRN) and transactions (via
postcode). MSOA11CD is the PRIMARY analysis geography (per the user's
Phase-2 brief: income/IMD2015/Census-2011 are all natively 2011 geography,
so MSOA11 is the well-behaved key rather than something crosswalked onto
MSOA21). MSOA21CD is retained as a reference column.

Geography paths (both single-postcode-vintage, no MSOA-boundary crosswalk
needed for either):
  - Transactions: postcode -> MSOA11CD via NSPL "2011 Census" (Feb 2024,
    still-maintained), postcode -> MSOA21CD via NSPL21 (Nov 2024, bundled
    with the UCL data) - as before.
  - New-builds: UPRN -> PCDS (via NSUL, Feb 2024) -> MSOA11CD (via the same
    NSPL 2011-Census postcode lookup); UPRN -> MSOA21CD directly from NSUL's
    own field, as before.

LAD/region are taken from the CURRENT (Nov 2024) geography throughout -
this is `lad23cd_analysis`, a single fixed grouping for later LAD x year
policy work, independent of which MSOA vintage a row's rows use.

England filter is applied on MSOA11CD (the primary key); MSOA21CD prefix is
cross-checked and any disagreement is logged, not silently resolved.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("05_attach_geography")

NSUL_GLOB = PROJECT_ROOT / "data" / "raw" / "geography" / "nsul" / "*.csv"
NSPL21_CSV = PROJECT_ROOT / "data" / "raw" / "geography" / "nspl" / "NSPL21_NOV_2024_UK.csv"
NSPL2011_CSV = PROJECT_ROOT / "data" / "raw" / "geography" / "nspl2011" / "NSPL_2011_FEB_2024_UK.csv"
NEWBUILDS_PARQUET = PROJECT_ROOT / "data" / "interim" / "newbuilds" / "newbuilds_analysis.parquet"
TRANS_DIR = PROJECT_ROOT / "data" / "interim" / "transactions_clean"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"

# Below this overall UPRN match rate, new-build counts by MSOA would be
# materially undercounted - stop and report rather than proceed silently.
MIN_ACCEPTABLE_UPRN_MATCH_RATE = 0.90

TRANS_FILES = ["transactions_2012_2023_trimmed.parquet", "transactions_2012_2023_untrimmed.parquet"]


def main() -> int:
    cfg = load_config()
    england_prefix = cfg["geography"]["england_prefix"]

    if not NEWBUILDS_PARQUET.exists():
        log.error(f"{NEWBUILDS_PARQUET} not found - run src/04_construct_newbuilds.py first.")
        return 1
    if not NSPL21_CSV.exists() or not NSPL2011_CSV.exists():
        log.error("NSPL21 and/or NSPL-2011-Census file not found - run src/01_inventory_ucl_files.py "
                   "and download NSPL_2011_FEB_2024_UK first.")
        return 1

    con = get_duckdb_connection(memory_limit="12GB")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Postcode -> MSOA11 (primary), current + maintained NSPL-2011-Census ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE postcode_2011 AS
        SELECT DISTINCT
            upper(trim(pcds)) AS postcode_std,
            msoa11 AS msoa11cd,
            lsoa11 AS lsoa11cd
        FROM read_csv_auto('{NSPL2011_CSV.as_posix()}', ALL_VARCHAR=true)
        WHERE msoa11 IS NOT NULL AND msoa11 != ''
        """
    )
    n_pc2011 = con.sql("SELECT count(*) FROM postcode_2011").fetchone()[0]
    log.info(f"NSPL-2011-Census postcode->MSOA11 lookup rows: {n_pc2011:,}")

    # --- Postcode -> MSOA21 (reference) + current LAD/region (analysis geography) ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE postcode_2021 AS
        SELECT DISTINCT
            upper(trim(pcds)) AS postcode_std,
            msoa21 AS msoa21cd,
            laua AS lad23cd_analysis,
            rgn AS rgn_code,
            TRY_CAST(oseast1m AS DOUBLE) AS easting,
            TRY_CAST(osnrth1m AS DOUBLE) AS northing,
            TRY_CAST(lat AS DOUBLE) AS latitude,
            TRY_CAST(long AS DOUBLE) AS longitude
        FROM read_csv_auto('{NSPL21_CSV.as_posix()}', ALL_VARCHAR=true)
        WHERE msoa21 IS NOT NULL AND msoa21 != ''
        """
    )
    n_pc2021 = con.sql("SELECT count(*) FROM postcode_2021").fetchone()[0]
    log.info(f"NSPL21 postcode->MSOA21/current-LAD lookup rows: {n_pc2021:,}")

    # =================================================================
    # New-builds: UPRN -> PCDS + MSOA21 (NSUL) -> MSOA11 (NSPL-2011-Census join)
    # =================================================================
    con.execute(
        f"""
        CREATE OR REPLACE TABLE uprn_geog AS
        SELECT DISTINCT
            TRY_CAST(UPRN AS BIGINT) AS uprn,
            upper(trim(PCDS)) AS postcode_std,
            msoa21cd,
            lad23cd AS nsul_lad23cd,
            rgn23cd AS nsul_rgn23cd,
            TRY_CAST(GRIDGB1E AS DOUBLE) AS easting,
            TRY_CAST(GRIDGB1N AS DOUBLE) AS northing
        FROM read_csv_auto('{NSUL_GLOB.as_posix()}', union_by_name=true, ALL_VARCHAR=true)
        WHERE msoa21cd IS NOT NULL AND msoa21cd != ''
        """
    )
    n_uprn_geog = con.sql("SELECT count(*) FROM uprn_geog").fetchone()[0]
    log.info(f"NSUL UPRN->MSOA21+postcode lookup rows (E&W): {n_uprn_geog:,}")

    con.execute(f"CREATE OR REPLACE VIEW newbuilds AS SELECT * FROM read_parquet('{NEWBUILDS_PARQUET.as_posix()}')")
    n_newbuilds = con.sql("SELECT count(*) FROM newbuilds").fetchone()[0]

    con.execute(
        """
        CREATE OR REPLACE TABLE newbuilds_geo_ew AS
        SELECT
            nb.*,
            u.msoa21cd,
            p11.msoa11cd,
            p11.lsoa11cd,
            p21.lad23cd_analysis,
            p21.rgn_code,
            u.easting, u.northing
        FROM newbuilds nb
        LEFT JOIN uprn_geog u USING (uprn)
        LEFT JOIN postcode_2011 p11 ON u.postcode_std = p11.postcode_std
        LEFT JOIN postcode_2021 p21 ON u.postcode_std = p21.postcode_std
        """
    )
    n_matched_msoa21 = con.sql("SELECT count(*) FROM newbuilds_geo_ew WHERE msoa21cd IS NOT NULL").fetchone()[0]
    n_matched_msoa11 = con.sql("SELECT count(*) FROM newbuilds_geo_ew WHERE msoa11cd IS NOT NULL").fetchone()[0]
    overall_match_rate_msoa11 = n_matched_msoa11 / n_newbuilds
    log.info(f"New-build UPRN->MSOA21 match rate (E&W): {n_matched_msoa21:,}/{n_newbuilds:,} = {n_matched_msoa21/n_newbuilds:.4f}")
    log.info(f"New-build UPRN->MSOA11 match rate (E&W, via postcode): {n_matched_msoa11:,}/{n_newbuilds:,} = {overall_match_rate_msoa11:.4f}")

    by_year = con.sql(
        """
        SELECT newbuild_year,
               count(*) AS n_newbuilds,
               sum(CASE WHEN msoa11cd IS NOT NULL THEN 1 ELSE 0 END) AS n_matched_msoa11,
               sum(CASE WHEN msoa21cd IS NOT NULL THEN 1 ELSE 0 END) AS n_matched_msoa21,
               sum(CASE WHEN msoa11cd IS NOT NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS match_rate_msoa11,
               sum(CASE WHEN msoa21cd IS NOT NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS match_rate_msoa21
        FROM newbuilds_geo_ew GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_year.to_csv(QA_DIR / "newbuild_geography_match_rate_by_year.csv", index=False)
    log.info(f"Match rate by newbuild_year:\n{by_year}")

    if overall_match_rate_msoa11 < MIN_ACCEPTABLE_UPRN_MATCH_RATE:
        log.error(
            f"STOP: overall UPRN->MSOA11 match rate ({overall_match_rate_msoa11:.2%}) is below the "
            f"{MIN_ACCEPTABLE_UPRN_MATCH_RATE:.0%} threshold and would materially undercount new-builds by MSOA."
        )
        return 1

    # --- England filter, primary on MSOA11CD; cross-check MSOA21CD agrees ---
    con.execute(
        f"CREATE OR REPLACE TABLE newbuilds_geo AS SELECT * FROM newbuilds_geo_ew "
        f"WHERE msoa11cd LIKE '{england_prefix}02%'"
    )
    n_england = con.sql("SELECT count(*) FROM newbuilds_geo").fetchone()[0]
    n_disagree = con.sql(
        f"SELECT count(*) FROM newbuilds_geo WHERE msoa21cd IS NOT NULL AND msoa21cd NOT LIKE '{england_prefix}02%'"
    ).fetchone()[0]
    log.info(f"New-builds restricted to England via MSOA11CD: {n_england:,} kept "
             f"({n_disagree:,} of those have a non-England MSOA21CD - logged, not dropped further)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY newbuilds_geo TO '{(OUT_DIR / "newbuilds_with_geography.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'newbuilds_with_geography.parquet'}")

    # =================================================================
    # Transactions: postcode -> MSOA11 (primary) + MSOA21/LAD/region (reference)
    # =================================================================
    trans_match_summary = []
    for fname in TRANS_FILES:
        fpath = TRANS_DIR / fname
        if not fpath.exists():
            log.warning(f"{fpath} not found, skipping.")
            continue
        con.execute(f"CREATE OR REPLACE VIEW trans_v AS SELECT * FROM read_parquet('{fpath.as_posix()}')")
        n_total = con.sql("SELECT count(*) FROM trans_v").fetchone()[0]
        con.execute(
            """
            CREATE OR REPLACE TABLE trans_geo_ew AS
            SELECT
                t.*,
                p11.msoa11cd, p11.lsoa11cd,
                p21.msoa21cd, p21.lad23cd_analysis, p21.rgn_code,
                p21.easting, p21.northing, p21.latitude, p21.longitude
            FROM trans_v t
            LEFT JOIN postcode_2011 p11 ON upper(trim(t.postcode)) = p11.postcode_std
            LEFT JOIN postcode_2021 p21 ON upper(trim(t.postcode)) = p21.postcode_std
            """
        )
        n_matched_11 = con.sql("SELECT count(*) FROM trans_geo_ew WHERE msoa11cd IS NOT NULL").fetchone()[0]
        n_matched_21 = con.sql("SELECT count(*) FROM trans_geo_ew WHERE msoa21cd IS NOT NULL").fetchone()[0]

        con.execute(f"CREATE OR REPLACE TABLE trans_geo AS SELECT * FROM trans_geo_ew WHERE msoa11cd LIKE '{england_prefix}02%'")
        n_england_t = con.sql("SELECT count(*) FROM trans_geo").fetchone()[0]

        trans_match_summary.append({
            "file": fname, "n_total": n_total,
            "n_matched_msoa11": n_matched_11, "match_rate_msoa11": n_matched_11 / n_total if n_total else float("nan"),
            "n_matched_msoa21": n_matched_21, "match_rate_msoa21": n_matched_21 / n_total if n_total else float("nan"),
            "n_england": n_england_t,
        })
        log.info(f"{fname}: MSOA11 match {n_matched_11:,}/{n_total:,}, MSOA21 match {n_matched_21:,}/{n_total:,}; "
                 f"{n_england_t:,} rows in England")

        out_name = fname.replace(".parquet", "_geo.parquet").replace("transactions_", "transactions_england_")
        con.execute(
            f"""
            COPY trans_geo TO '{(OUT_DIR / out_name).as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    import pandas as pd
    pd.DataFrame(trans_match_summary).to_csv(QA_DIR / "transactions_geography_match_rate.csv", index=False)

    # --- Clean MSOA11-level geography reference table (one row per MSOA11):
    # MSOA21CD reference, LAD, region ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE msoa11_geography_reference AS
        SELECT
            p11.msoa11cd,
            any_value(p21.msoa21cd) AS msoa21cd_reference,
            any_value(p21.lad23cd_analysis) AS lad23cd_analysis,
            any_value(p21.rgn_code) AS rgn_code
        FROM postcode_2011 p11
        LEFT JOIN postcode_2021 p21 ON p11.postcode_std = p21.postcode_std
        WHERE p11.msoa11cd LIKE '{england_prefix}02%'
        GROUP BY p11.msoa11cd
        """
    )
    con.execute(
        f"""
        COPY msoa11_geography_reference TO '{(OUT_DIR / "msoa11_geography_reference.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    n_geo_ref = con.sql("SELECT count(*) FROM msoa11_geography_reference").fetchone()[0]
    log.info(f"Wrote msoa11_geography_reference.parquet ({n_geo_ref:,} England MSOA11 rows: MSOA21CD ref/LAD/region)")

    log.info("Geography attachment complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
