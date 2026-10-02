"""Identify the universe of new-build dwellings from the EPC source and
classify their tenure at the point of construction.

Per the brief:
- TRANSACTION_TYPE == 'new dwelling' identifies a new-build EPC
- UPRN is the primary property identifier
- for every UPRN, the earliest valid new-dwelling EPC defines the
  new-build event (newbuild_date) - one record per dwelling
- tenure_group in {SOCIAL, PRIVATE_NONSOCIAL, UNKNOWN}; UNKNOWN is never
  folded into PRIVATE
- primary tenure measure = tenure on the new-dwelling EPC itself
- optional robustness: if that tenure is unknown, look at the same UPRN's
  later EPCs (within 24 months) for the first non-unknown tenure, kept as
  a *separate* tenure_group_imputed column - the primary is never
  overwritten
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("04_construct_newbuilds")

EPC_PARQUET = PROJECT_ROOT / "data" / "interim" / "epc_clean" / "epc_clean.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "newbuilds"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def build_tenure_case_sql(tenure_groups: dict, column: str = "tenure_raw") -> str:
    """Build a SQL CASE expression mapping tenure_raw -> tenure_group using
    the config-driven lists, falling back to UNKNOWN for anything else."""
    clauses = []
    for group, raw_values in tenure_groups.items():
        if group == "UNKNOWN":
            continue
        values_sql = ", ".join(f"'{v}'" for v in raw_values)
        clauses.append(f"WHEN lower(trim({column})) IN ({values_sql}) THEN '{group}'")
    case_sql = "CASE " + " ".join(clauses) + " ELSE 'UNKNOWN' END"
    return case_sql


def main() -> int:
    cfg = load_config()
    tenure_groups = cfg["newbuilds"]["tenure_groups"]
    tx_type_value = cfg["newbuilds"]["transaction_type_value"]
    imputation_window_months = cfg["newbuilds"]["tenure_imputation_window_months"]

    if not EPC_PARQUET.exists():
        log.error(f"{EPC_PARQUET} not found - run src/03_clean_epc.py first.")
        return 1

    con = get_duckdb_connection(memory_limit="10GB")
    con.execute(f"CREATE OR REPLACE VIEW epc_clean AS SELECT * FROM read_parquet('{EPC_PARQUET.as_posix()}')")

    n_epc_total = con.sql("SELECT count(*) FROM epc_clean").fetchone()[0]

    # --- Universe of new-dwelling EPCs ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE newdwelling_epcs AS
        SELECT * FROM epc_clean WHERE transaction_type = '{tx_type_value}'
        """
    )
    n_newdwelling = con.sql("SELECT count(*) FROM newdwelling_epcs").fetchone()[0]
    log.info(f"New-dwelling EPC records: {n_newdwelling:,} of {n_epc_total:,} total EPC records")

    n_missing_uprn = con.sql("SELECT count(*) FROM newdwelling_epcs WHERE uprn IS NULL").fetchone()[0]
    log.info(f"New-dwelling EPCs missing UPRN: {n_missing_uprn:,} ({n_missing_uprn/n_newdwelling*100:.2f}%)")

    n_missing_lodgement = con.sql(
        "SELECT count(*) FROM newdwelling_epcs WHERE lodgement_date IS NULL"
    ).fetchone()[0]
    log.info(f"New-dwelling EPCs missing lodgement_date: {n_missing_lodgement:,}")

    # --- Earliest valid new-dwelling EPC per UPRN = the new-build event ---
    tenure_case_sql = build_tenure_case_sql(tenure_groups, "tenure_raw")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE newbuild_events AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT
                *,
                {tenure_case_sql} AS tenure_group,
                'new_dwelling_epc' AS tenure_source,
                FALSE AS tenure_imputed,
                lodgement_date AS newbuild_date,
                EXTRACT(year FROM lodgement_date) AS newbuild_year,
                (EXTRACT(quarter FROM lodgement_date))::INTEGER AS newbuild_quarter,
                total_floor_area AS newbuild_floor_area,
                property_type AS newbuild_property_type,
                ROW_NUMBER() OVER (PARTITION BY uprn ORDER BY lodgement_date ASC, inspection_date ASC) AS rn
            FROM newdwelling_epcs
            WHERE uprn IS NOT NULL AND lodgement_date IS NOT NULL
        )
        WHERE rn = 1
        """
    )
    n_newbuild_uprns = con.sql("SELECT count(*) FROM newbuild_events").fetchone()[0]
    log.info(f"Unique new-build UPRNs (one row per dwelling): {n_newbuild_uprns:,}")

    n_duplicate_epcs = n_newdwelling - n_missing_uprn - n_missing_lodgement - n_newbuild_uprns
    log.info(f"Duplicate new-dwelling EPCs collapsed (same UPRN, later lodgement): {max(n_duplicate_epcs, 0):,}")

    # --- Optional robustness: 24-month forward search for non-unknown tenure ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE tenure_lookahead AS
        SELECT
            nb.uprn,
            nb.tenure_group AS tenure_group_primary,
            first(({tenure_case_sql.replace("tenure_raw", "e.tenure_raw")}) ORDER BY e.lodgement_date ASC) FILTER (
                WHERE e.lodgement_date > nb.newbuild_date
                  AND e.lodgement_date <= nb.newbuild_date + INTERVAL '{imputation_window_months} months'
                  AND {tenure_case_sql.replace("tenure_raw", "e.tenure_raw")} != 'UNKNOWN'
            ) AS first_nonunknown_later_tenure
        FROM newbuild_events nb
        LEFT JOIN epc_clean e ON e.uprn = nb.uprn
        GROUP BY nb.uprn, nb.tenure_group
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE newbuild_events_final AS
        SELECT
            nb.*,
            CASE WHEN nb.tenure_group = 'UNKNOWN' AND tl.first_nonunknown_later_tenure IS NOT NULL
                 THEN tl.first_nonunknown_later_tenure
                 ELSE nb.tenure_group
            END AS tenure_group_imputed
        FROM newbuild_events nb
        LEFT JOIN tenure_lookahead tl USING (uprn)
        """
    )
    n_imputed_affected = con.sql(
        "SELECT count(*) FROM newbuild_events_final WHERE tenure_group = 'UNKNOWN' AND tenure_group_imputed != 'UNKNOWN'"
    ).fetchone()[0]
    log.info(
        f"Records where the {imputation_window_months}-month look-ahead robustness reclassified an "
        f"UNKNOWN primary tenure: {n_imputed_affected:,} "
        "(kept only in tenure_group_imputed; tenure_group / primary measure is untouched)"
    )

    # --- QA: cross-tab by year x raw tenure x grouped tenure x property type ---
    crosstab = con.sql(
        """
        SELECT newbuild_year, tenure_raw, tenure_group, property_type, count(*) AS n
        FROM newbuild_events_final
        GROUP BY 1, 2, 3, 4
        ORDER BY 1, 5 DESC
        """
    ).df()
    crosstab.to_csv(QA_DIR / "newbuilds_by_year_tenure_propertytype.csv", index=False)

    unknown_share_by_year = con.sql(
        """
        SELECT newbuild_year,
               count(*) AS n_newbuilds,
               sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END) AS n_unknown_tenure,
               sum(CASE WHEN tenure_group = 'UNKNOWN' THEN 1 ELSE 0 END)::DOUBLE / count(*) AS unknown_share
        FROM newbuild_events_final
        GROUP BY 1 ORDER BY 1
        """
    ).df()
    unknown_share_by_year.to_csv(QA_DIR / "newbuilds_unknown_tenure_share_by_year.csv", index=False)
    log.info(f"UNKNOWN tenure share by newbuild_year:\n{unknown_share_by_year}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY newbuild_events_final TO '{(OUT_DIR / "newbuilds_analysis.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {OUT_DIR / 'newbuilds_analysis.parquet'} ({n_newbuild_uprns:,} rows)")
    log.info("New-build construction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
