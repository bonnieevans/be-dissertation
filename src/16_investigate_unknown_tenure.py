"""Deep-dive investigation into why so many new-build EPCs have UNKNOWN
tenure, requested after the headline finding in 04_construct_newbuilds.py /
05_attach_geography.py QA (70-100% UNKNOWN per year).

Produces outputs/excel/unknown_tenure_investigation.xlsx with:
  - national UnknownShare_t = UnknownNewBuilds_t / TotalNewBuilds_t by year
  - unknown share by region, property type, income quartile, development
    size, and MSOA
  - a formal check that Social + Private + Unknown = Total
  - the share of MSOA-years with construction but zero known-tenure new-builds
  - a profile of the UNKNOWN group itself (property type / built form /
    construction age band vs the known-tenure group)
  - what happens next: the subsequent EPC's transaction_type and time-gap
    for UNKNOWN-tenure dwellings (does tenure only become knowable once the
    dwelling is later sold/let?)
"""

from __future__ import annotations

import sys

import pandas as pd

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("16_investigate_unknown_tenure")

NEWBUILDS_GEO = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "newbuilds_with_geography.parquet"
EPC_CLEAN = PROJECT_ROOT / "data" / "interim" / "epc_clean" / "epc_clean.parquet"
NSUL_GLOB = PROJECT_ROOT / "data" / "raw" / "geography" / "nsul" / "*.csv"
INCOME_PARQUET = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa21_baseline_income.parquet"
OUT_XLSX = PROJECT_ROOT / "outputs" / "excel" / "unknown_tenure_investigation.xlsx"

REGION_NAMES = {
    "E12000001": "North East", "E12000002": "North West",
    "E12000003": "Yorkshire and The Humber", "E12000004": "East Midlands",
    "E12000005": "West Midlands", "E12000006": "East of England",
    "E12000007": "London", "E12000008": "South East",
    "E12000009": "South West", "W92000004": "Wales", "W99999999": "Wales",
}

# Spatial-only proxy for "development size": no site/planning-application
# identifier exists in either source, so dwellings are grouped into 100m x
# 100m grid cells (using NSUL easting/northing) and development_size counts
# every new-build UPRN ever recorded in that cell, across all years. This is
# a documented heuristic, not a true scheme boundary.
GRID_SIZE_M = 100


def main() -> int:
    if not NEWBUILDS_GEO.exists():
        log.error(f"{NEWBUILDS_GEO} not found - run src/05_attach_geography.py first.")
        return 1

    con = get_duckdb_connection(memory_limit="10GB")
    con.execute(f"CREATE OR REPLACE VIEW nb AS SELECT * FROM read_parquet('{NEWBUILDS_GEO.as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW epc AS SELECT * FROM read_parquet('{EPC_CLEAN.as_posix()}')")

    n_total = con.sql("SELECT count(*) FROM nb").fetchone()[0]
    log.info(f"Total new-build dwellings: {n_total:,}")

    sheets: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------
    # 1. National UnknownShare_t by year, with the Social+Private+Unknown=Total check
    # ---------------------------------------------------------------
    by_year = con.sql(
        """
        SELECT
            newbuild_year AS year,
            count(*) AS total_newbuilds,
            sum(CASE WHEN tenure_group = 'SOCIAL' THEN 1 ELSE 0 END) AS social_newbuilds,
            sum(CASE WHEN tenure_group = 'PRIVATE_NONSOCIAL' THEN 1 ELSE 0 END) AS private_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_newbuilds
        FROM nb GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_year["unknown_share"] = by_year["unknown_newbuilds"] / by_year["total_newbuilds"]
    by_year["identity_check_sums_to_total"] = (
        by_year["social_newbuilds"] + by_year["private_newbuilds"] + by_year["unknown_newbuilds"]
        == by_year["total_newbuilds"]
    )
    sheets["unknown_share_by_year"] = by_year
    log.info(f"National unknown share by year:\n{by_year[['year','total_newbuilds','unknown_share']]}")

    overall_check = pd.DataFrame([{
        "total_newbuilds": n_total,
        "social_newbuilds": int(by_year["social_newbuilds"].sum()),
        "private_newbuilds": int(by_year["private_newbuilds"].sum()),
        "unknown_newbuilds": int(by_year["unknown_newbuilds"].sum()),
        "sums_to_total": bool(by_year["identity_check_sums_to_total"].all()),
        "overall_unknown_share": by_year["unknown_newbuilds"].sum() / n_total,
    }])
    sheets["identity_check_overall"] = overall_check

    # ---------------------------------------------------------------
    # 2. By region (via fresh NSUL join for rgn23cd - not carried in the
    #    main pipeline's newbuilds_with_geography.parquet)
    # ---------------------------------------------------------------
    con.execute(
        f"""
        CREATE OR REPLACE TABLE uprn_region AS
        SELECT DISTINCT TRY_CAST(UPRN AS BIGINT) AS uprn, rgn23cd, lad23cd
        FROM read_csv_auto('{NSUL_GLOB.as_posix()}', union_by_name=true, ALL_VARCHAR=true)
        WHERE rgn23cd IS NOT NULL AND rgn23cd != ''
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE nb_region AS
        SELECT nb.*, r.rgn23cd, r.lad23cd
        FROM nb LEFT JOIN uprn_region r USING (uprn)
        """
    )
    by_region = con.sql(
        """
        SELECT
            rgn23cd,
            count(*) AS total_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM nb_region
        GROUP BY 1 ORDER BY unknown_share DESC
        """
    ).df()
    by_region["region_name"] = by_region["rgn23cd"].map(REGION_NAMES)
    by_region["region_name"] = by_region["region_name"].fillna(by_region["rgn23cd"].fillna("Unmatched to region"))
    sheets["unknown_share_by_region"] = by_region[
        ["rgn23cd", "region_name", "total_newbuilds", "unknown_newbuilds", "unknown_share"]
    ]
    log.info(f"Unknown share by region:\n{sheets['unknown_share_by_region']}")

    # ---------------------------------------------------------------
    # 3. By property type (and built_form / construction_age_band), plus a
    #    side-by-side profile of UNKNOWN vs known-tenure composition
    # ---------------------------------------------------------------
    by_proptype = con.sql(
        """
        SELECT
            property_type,
            count(*) AS total_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM nb GROUP BY 1 ORDER BY unknown_share DESC
        """
    ).df()
    sheets["unknown_share_by_property_type"] = by_proptype
    log.info(f"Unknown share by property type:\n{by_proptype}")

    profile = con.sql(
        """
        WITH labelled AS (
            SELECT
                CASE WHEN tenure_group = 'UNKNOWN' THEN 'UNKNOWN' ELSE 'KNOWN (SOCIAL or PRIVATE_NONSOCIAL)' END AS group_label,
                property_type
            FROM nb
        ),
        counted AS (
            SELECT group_label, property_type, count(*) AS n
            FROM labelled GROUP BY 1, 2
        )
        SELECT group_label, property_type, n,
               n::DOUBLE / sum(n) OVER (PARTITION BY group_label) AS share_within_group
        FROM counted ORDER BY group_label, n DESC
        """
    ).df()
    sheets["property_type_profile_unknown_vs_known"] = profile

    by_built_form = con.sql(
        """
        SELECT built_form, count(*) AS total_newbuilds,
               sum(CASE WHEN tenure_group='UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM nb GROUP BY 1 ORDER BY unknown_share DESC
        """
    ).df()
    sheets["unknown_share_by_built_form"] = by_built_form

    by_age_band = con.sql(
        """
        SELECT construction_age_band, count(*) AS total_newbuilds,
               sum(CASE WHEN tenure_group='UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM nb GROUP BY 1 ORDER BY 2 DESC
        """
    ).df()
    sheets["unknown_share_by_construction_age_band"] = by_age_band

    # ---------------------------------------------------------------
    # 4. By income quartile (of the MSOA the new-build sits in)
    # ---------------------------------------------------------------
    if INCOME_PARQUET.exists():
        con.execute(f"CREATE OR REPLACE VIEW income AS SELECT * FROM read_parquet('{INCOME_PARQUET.as_posix()}')")
        by_income_q = con.sql(
            """
            SELECT
                i.income_quartile,
                count(*) AS total_newbuilds,
                sum(CASE WHEN n.tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_newbuilds,
                sum(CASE WHEN n.tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
            FROM nb n
            JOIN income i ON n.msoa21cd = i.msoa21cd
            GROUP BY 1 ORDER BY 1
            """
        ).df()
        sheets["unknown_share_by_income_quartile"] = by_income_q
        log.info(f"Unknown share by income quartile:\n{by_income_q}")
    else:
        log.warning(f"{INCOME_PARQUET} not found, skipping income-quartile breakdown.")

    # ---------------------------------------------------------------
    # 5. By development size (spatial proxy - see module docstring)
    # ---------------------------------------------------------------
    con.execute(
        f"""
        CREATE OR REPLACE TABLE nb_gridded AS
        SELECT *,
               floor(easting / {GRID_SIZE_M}) * {GRID_SIZE_M} AS grid_e,
               floor(northing / {GRID_SIZE_M}) * {GRID_SIZE_M} AS grid_n
        FROM nb
        WHERE easting IS NOT NULL AND northing IS NOT NULL
        """
    )
    n_geocoded = con.sql("SELECT count(*) FROM nb_gridded").fetchone()[0]
    con.execute(
        """
        CREATE OR REPLACE TABLE dev_size AS
        SELECT grid_e, grid_n, count(*) AS development_size
        FROM nb_gridded GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE nb_with_dev_size AS
        SELECT g.*, d.development_size,
            CASE
                WHEN d.development_size = 1 THEN '1 (isolated infill)'
                WHEN d.development_size BETWEEN 2 AND 4 THEN '2-4'
                WHEN d.development_size BETWEEN 5 AND 9 THEN '5-9'
                WHEN d.development_size BETWEEN 10 AND 19 THEN '10-19'
                WHEN d.development_size BETWEEN 20 AND 49 THEN '20-49'
                WHEN d.development_size BETWEEN 50 AND 99 THEN '50-99'
                ELSE '100+'
            END AS development_size_band
        FROM nb_gridded g JOIN dev_size d USING (grid_e, grid_n)
        """
    )
    by_dev_size = con.sql(
        """
        SELECT
            development_size_band,
            count(*) AS total_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM nb_with_dev_size
        GROUP BY 1
        ORDER BY CASE development_size_band
            WHEN '1 (isolated infill)' THEN 1 WHEN '2-4' THEN 2 WHEN '5-9' THEN 3
            WHEN '10-19' THEN 4 WHEN '20-49' THEN 5 WHEN '50-99' THEN 6 ELSE 7 END
        """
    ).df()
    sheets["unknown_share_by_development_size"] = by_dev_size
    log.info(
        f"Development-size proxy built from {n_geocoded:,}/{n_total:,} geocoded new-builds "
        f"({GRID_SIZE_M}m grid cells, no time restriction). Unknown share by band:\n{by_dev_size}"
    )

    # ---------------------------------------------------------------
    # 6. By MSOA (top 50 by unknown count, plus full distribution stats)
    # ---------------------------------------------------------------
    by_msoa = con.sql(
        """
        SELECT
            msoa21cd,
            count(*) AS total_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS unknown_newbuilds,
            sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM nb WHERE msoa21cd IS NOT NULL
        GROUP BY 1
        """
    ).df()
    top50_by_unknown_count = by_msoa.sort_values("unknown_newbuilds", ascending=False).head(50)
    sheets["top50_msoa_by_unknown_count"] = top50_by_unknown_count
    unknown_share_dist = by_msoa["unknown_share"].describe().to_frame(name="unknown_share_across_msoas")
    sheets["msoa_unknown_share_distribution"] = unknown_share_dist.reset_index().rename(columns={"index": "statistic"})
    log.info(f"MSOA-level unknown_share distribution:\n{unknown_share_dist}")

    # ---------------------------------------------------------------
    # 7. Proportion of MSOA-years with construction but zero known-tenure new-builds
    # ---------------------------------------------------------------
    cfg = load_config()
    start_year, end_year = cfg["study_period"]["main_panel_start_year"], cfg["study_period"]["main_panel_end_year"]
    # NOTE: filtering on newbuild_year BETWEEN ... and grouping in the same
    # query triggers a DuckDB 1.2.1 internal assertion crash
    # (compress_integral.cpp, "min_val <= input") on this dataset. Materialising
    # the filtered rows into a table first, then grouping in a separate
    # query, avoids it.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE nb_study_period AS
        SELECT * FROM nb WHERE msoa21cd IS NOT NULL AND newbuild_year BETWEEN {start_year} AND {end_year}
        """
    )
    msoa_year = con.sql(
        """
        SELECT msoa21cd, newbuild_year AS year,
               count(*) AS total_newbuilds,
               sum(CASE WHEN tenure_group IN ('SOCIAL','PRIVATE_NONSOCIAL') THEN 1 ELSE 0 END) AS known_tenure_newbuilds
        FROM nb_study_period
        GROUP BY 1, 2
        """
    ).df()
    msoa_years_with_construction = msoa_year[msoa_year["total_newbuilds"] > 0]
    zero_known = msoa_years_with_construction[msoa_years_with_construction["known_tenure_newbuilds"] == 0]
    pct_zero_known = len(zero_known) / len(msoa_years_with_construction)
    log.info(
        f"MSOA-years with construction ({len(msoa_years_with_construction):,}), of which "
        f"{len(zero_known):,} ({pct_zero_known:.2%}) have ZERO known-tenure new-builds."
    )
    sheets["msoa_years_zero_known_tenure"] = pd.DataFrame([{
        "n_msoa_years_with_construction": len(msoa_years_with_construction),
        "n_with_zero_known_tenure": len(zero_known),
        "pct_with_zero_known_tenure": pct_zero_known,
    }])

    # ---------------------------------------------------------------
    # 8. What happens next: subsequent EPC transaction_type for UNKNOWN dwellings
    # ---------------------------------------------------------------
    con.execute(
        """
        CREATE OR REPLACE TABLE unknown_next_epc AS
        SELECT
            u.uprn,
            first(e.transaction_type ORDER BY e.lodgement_date ASC) FILTER (WHERE e.lodgement_date > u.newbuild_date) AS next_transaction_type,
            first(e.lodgement_date ORDER BY e.lodgement_date ASC) FILTER (WHERE e.lodgement_date > u.newbuild_date) AS next_lodgement_date
        FROM nb u
        LEFT JOIN epc e ON e.uprn = u.uprn
        WHERE u.tenure_group = 'UNKNOWN'
        GROUP BY u.uprn, u.newbuild_date
        """
    )
    n_unknown = con.sql("SELECT count(*) FROM unknown_next_epc").fetchone()[0]
    n_has_next = con.sql("SELECT count(*) FROM unknown_next_epc WHERE next_transaction_type IS NOT NULL").fetchone()[0]
    log.info(f"Of {n_unknown:,} UNKNOWN-tenure new-builds, {n_has_next:,} ({n_has_next/n_unknown:.1%}) have ANY later EPC record at all.")

    next_type_dist = con.sql(
        """
        SELECT coalesce(next_transaction_type, '(no later EPC found)') AS next_transaction_type,
               count(*) AS n,
               count(*)::DOUBLE / (SELECT count(*) FROM unknown_next_epc) AS share
        FROM unknown_next_epc
        GROUP BY 1 ORDER BY n DESC
        """
    ).df()
    sheets["unknown_subsequent_epc_transaction_type"] = next_type_dist
    log.info(f"Subsequent EPC transaction_type for UNKNOWN-tenure dwellings:\n{next_type_dist}")

    gap_stats = con.sql(
        """
        SELECT
            date_diff('day', u.newbuild_date, x.next_lodgement_date) AS days_to_next_epc
        FROM unknown_next_epc x
        JOIN nb u USING (uprn)
        WHERE x.next_lodgement_date IS NOT NULL
        """
    ).df()["days_to_next_epc"].describe().to_frame(name="days_to_next_epc")
    sheets["unknown_days_to_next_epc_distribution"] = gap_stats.reset_index().rename(columns={"index": "statistic"})
    log.info(f"Days from new-build EPC to next EPC (UNKNOWN-tenure dwellings only):\n{gap_stats}")

    # ---------------------------------------------------------------
    # Methodology notes
    # ---------------------------------------------------------------
    sheets["methodology_notes"] = pd.DataFrame({"note": [
        "WHY THIS INVESTIGATION",
        "The main pipeline found 70-100% of new-build EPCs per year have tenure_group = UNKNOWN "
        "(raw TENURE values 'unknown' / 'not defined - use in the case of a new dwelling...' / "
        "'no data!' / blank). This workbook investigates the pattern before any econometric use "
        "of the tenure split.",
        "",
        "DEVELOPMENT SIZE PROXY - LIMITATION",
        f"Neither the EPC source nor the linked transactions carry a site/scheme/planning-application "
        f"identifier. 'development_size' is a spatial proxy only: new-build UPRNs are grouped into "
        f"{GRID_SIZE_M}m x {GRID_SIZE_M}m grid cells using NSUL easting/northing, with NO time "
        f"restriction (a phased estate built out over several years will still count as one 'development'). "
        f"Treat this as an approximate, not an authoritative, measure of scheme size.",
        "",
        "'SUBSEQUENT EPC TRANSACTION TYPE' - WHAT IT MEASURES",
        "For each UNKNOWN-tenure new-build UPRN, this looks at the SAME dwelling's next chronological "
        "EPC record (of any type - re-inspection, sale, rental, etc.) after the new-build EPC itself, "
        "and reports its transaction_type. This tests whether tenure is unknown simply because the "
        "dwelling had not yet been sold or let at the time the new-dwelling EPC was lodged.",
        "",
        "IDENTITY CHECK",
        "Social + Private_nonsocial + Unknown = Total holds exactly by construction (tenure_group is "
        "a three-way partition of every new-build record with no other category) - the "
        "identity_check_overall / unknown_share_by_year sheets report this as a formal cross-check, "
        "not because it was in doubt.",
    ]})

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    log.info(f"Wrote {OUT_XLSX} with {len(sheets)} sheets.")
    log.info("Unknown-tenure investigation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
