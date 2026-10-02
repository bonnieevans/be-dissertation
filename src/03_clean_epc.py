"""Clean the UCL deposited EPC source (epc_2024.csv) - standardise the
TRANSACTION_TYPE and TENURE text fields and produce QA counts by year.
This is the universe of EPC-recorded dwellings used to identify new builds
in 04_construct_newbuilds.py. It is *not* used to estimate house prices.
"""

from __future__ import annotations

import sys

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("03_clean_epc")

EPC_CSV = PROJECT_ROOT / "data" / "raw" / "ucl" / "epc_source" / "epc_2024.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "epc_clean"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"

REQUIRED_COLUMNS = [
    "LMK_KEY", "UPRN", "LODGEMENT_DATE", "INSPECTION_DATE", "TRANSACTION_TYPE",
    "TENURE", "TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS",
    "CONSTRUCTION_AGE_BAND", "PROPERTY_TYPE", "BUILT_FORM",
]


def main() -> int:
    if not EPC_CSV.exists():
        log.error(f"{EPC_CSV} not found - run src/01_inventory_ucl_files.py first.")
        return 1

    con = get_duckdb_connection(memory_limit="10GB")

    header = con.sql(f"SELECT * FROM read_csv_auto('{EPC_CSV.as_posix()}', sample_size=1) LIMIT 0").columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        log.error(
            f"STOP: required EPC column(s) missing from deposited source: {missing}. "
            "Per the brief, do not invent a substitute - reporting and halting."
        )
        return 1
    log.info("All required EPC columns present: " + ", ".join(REQUIRED_COLUMNS))

    con.execute(
        f"""
        CREATE OR REPLACE TABLE epc_raw AS
        SELECT
            LMK_KEY AS lmk_key,
            TRY_CAST(UPRN AS BIGINT) AS uprn,
            UPRN_SOURCE AS uprn_source,
            TRY_CAST(LODGEMENT_DATE AS DATE) AS lodgement_date,
            TRY_CAST(INSPECTION_DATE AS DATE) AS inspection_date,
            lower(trim(TRANSACTION_TYPE)) AS transaction_type,
            lower(trim(TENURE)) AS tenure_raw,
            TRY_CAST(TOTAL_FLOOR_AREA AS DOUBLE) AS total_floor_area,
            TRY_CAST(NUMBER_HABITABLE_ROOMS AS DOUBLE) AS number_habitable_rooms,
            CONSTRUCTION_AGE_BAND AS construction_age_band,
            PROPERTY_TYPE AS property_type,
            BUILT_FORM AS built_form
        FROM read_csv_auto('{EPC_CSV.as_posix()}')
        """
    )
    n_total = con.sql("SELECT count(*) FROM epc_raw").fetchone()[0]
    log.info(f"EPC records loaded: {n_total:,}")

    by_year = con.sql(
        """
        SELECT EXTRACT(year FROM lodgement_date) AS lodgement_year, count(*) AS n
        FROM epc_raw GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_year.to_csv(QA_DIR / "epc_records_by_year.csv", index=False)

    tx_type_dist = con.sql(
        "SELECT transaction_type, count(*) AS n FROM epc_raw GROUP BY 1 ORDER BY n DESC"
    ).df()
    tx_type_dist.to_csv(QA_DIR / "epc_transaction_type_distribution.csv", index=False)
    log.info(f"transaction_type distribution:\n{tx_type_dist}")

    tenure_dist = con.sql(
        "SELECT tenure_raw, count(*) AS n FROM epc_raw GROUP BY 1 ORDER BY n DESC"
    ).df()
    tenure_dist.to_csv(QA_DIR / "epc_tenure_raw_distribution.csv", index=False)
    log.info(f"tenure_raw distribution:\n{tenure_dist}")

    n_missing_uprn = con.sql("SELECT count(*) FROM epc_raw WHERE uprn IS NULL").fetchone()[0]
    log.info(f"EPC records missing UPRN: {n_missing_uprn:,} ({n_missing_uprn/n_total*100:.2f}%)")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY epc_raw TO '{(INTERIM_DIR / "epc_clean.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    log.info(f"Wrote {INTERIM_DIR / 'epc_clean.parquet'}")
    log.info("EPC cleaning complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
