"""Clean the UCL linked PPD-EPC transactions into the established-home
outcome sample used for the house-price side of the panel.

Per the brief:
- retain raw housing data 2012-2023 (principal regression sample is a
  2016-2023 subset flagged later, in 14_merge_final_panel.py)
- established homes only (oldnew = 'N') -- we are measuring prices in the
  surrounding existing stock, not the new-build homes themselves
- retain standard/full-market-value transactions where identifiable
- exclude invalid/nonpositive price and floor area
- calculated_ppsqm = transaction_price / total_floor_area, compared against
  UCL's own `priceper`; discrepancies are reported, not silently resolved
- retain UCL's match-quality variable `classt`
- nominal_ppsqm / log_ppsqm always constructed; real_ppsqm via CPIH for
  descriptive use
- outlier procedure: inspect annual distributions, baseline trim at
  0.5/99.5 pct per year, but always keep an untrimmed robustness copy and
  report what was removed
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

from utils import (
    PROJECT_ROOT,
    get_duckdb_connection,
    get_logger,
    load_config,
)

log = get_logger("02_clean_transactions")

TRANS_CSV = PROJECT_ROOT / "data" / "raw" / "ucl" / "linked_transactions" / "tranall_link_26122024.csv"
CPIH_DIR = PROJECT_ROOT / "data" / "raw" / "macro" / "cpih"
CPIH_CSV = CPIH_DIR / "cpih_l522_mm23.csv"
CPIH_URL = "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/l522/mm23"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "transactions_clean"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def ensure_cpih() -> Path:
    if CPIH_CSV.exists():
        log.info(f"CPIH series already present: {CPIH_CSV}")
        return CPIH_CSV
    CPIH_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading CPIH (L522/MM23) from {CPIH_URL}")
    resp = requests.get(CPIH_URL, timeout=60)
    resp.raise_for_status()
    CPIH_CSV.write_bytes(resp.content)
    log.info(f"Saved CPIH series -> {CPIH_CSV} ({len(resp.content):,} bytes)")

    manifest_path = PROJECT_ROOT / "config" / "source_manifest.csv"
    text = manifest_path.read_text()
    import hashlib
    from datetime import date

    checksum = hashlib.sha256(resp.content).hexdigest()
    old_row_prefix = '"ONS CPIH (D7G7 or equivalent series), monthly index",TBD'
    if old_row_prefix in text:
        new_row = (
            '"ONS CPIH INDEX 00: ALL ITEMS 2015=100 (CDID L522, dataset MM23)",'
            f"{CPIH_URL},{date.today().isoformat()},cpih_l522_mm23.csv,"
            f"{len(resp.content)},{checksum},1988 to present,"
            '"Downloaded via ONS website CSV generator; old api.ons.gov.uk v0 endpoint is retired (410/404)"'
        )
        text = text.replace(
            text[text.index(old_row_prefix): text.index("\n", text.index(old_row_prefix))],
            new_row,
        )
        manifest_path.write_text(text)
        log.info("Updated config/source_manifest.csv with CPIH provenance.")
    return CPIH_CSV


def parse_cpih_monthly(con) -> None:
    """Load the CPIH CSV (title/metadata rows, then annual + 'YYYY MON' rows)
    into a DuckDB table of (year, month, cpih_index) keeping only monthly rows."""
    # The ONS export stacks annual rows ("1988"), then quarterly ("1988 Q1"),
    # then monthly ("1988 JAN") rows in one column - only the monthly rows
    # have a recognised month token, so an inner CASE + outer NULL-filter
    # is used to keep exactly those.
    con.execute(
        f"""
        CREATE OR REPLACE TABLE cpih_monthly AS
        SELECT cpih_year, cpih_month, cpih_index FROM (
            SELECT
                TRY_CAST(split_part(period, ' ', 1) AS INTEGER) AS cpih_year,
                CASE upper(split_part(period, ' ', 2))
                    WHEN 'JAN' THEN 1 WHEN 'FEB' THEN 2 WHEN 'MAR' THEN 3
                    WHEN 'APR' THEN 4 WHEN 'MAY' THEN 5 WHEN 'JUN' THEN 6
                    WHEN 'JUL' THEN 7 WHEN 'AUG' THEN 8 WHEN 'SEP' THEN 9
                    WHEN 'OCT' THEN 10 WHEN 'NOV' THEN 11 WHEN 'DEC' THEN 12
                    ELSE NULL
                END AS cpih_month,
                CAST(cpih_index AS DOUBLE) AS cpih_index
            FROM read_csv('{CPIH_CSV.as_posix()}', header=false, skip=8,
                           columns={{'period': 'VARCHAR', 'cpih_index': 'VARCHAR'}})
        )
        WHERE cpih_month IS NOT NULL
        """
    )
    n = con.sql("SELECT count(*) FROM cpih_monthly").fetchone()[0]
    log.info(f"Loaded {n} monthly CPIH observations.")


def main() -> int:
    cfg = load_config()
    start_year = cfg["study_period"]["main_panel_start_year"]
    end_year_main = cfg["study_period"]["main_panel_end_year"]
    old_new_filter = cfg["transactions"]["old_new_filter"]
    min_price = cfg["transactions"]["min_price"]
    min_floor_area = cfg["transactions"]["min_floor_area_sqm"]
    trim_low = cfg["transactions"]["outlier_trim_pct_low"]
    trim_high = cfg["transactions"]["outlier_trim_pct_high"]

    if not TRANS_CSV.exists():
        log.error(f"{TRANS_CSV} not found - run src/01_inventory_ucl_files.py first.")
        return 1

    ensure_cpih()

    QA_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    con = get_duckdb_connection(memory_limit="10GB")
    parse_cpih_monthly(con)

    # --- Inspect categorytype / recordstatus before deciding the filter ---
    dist = con.sql(
        f"""
        SELECT categorytype, recordstatus, count(*) AS n
        FROM read_csv_auto('{TRANS_CSV.as_posix()}')
        WHERE dateoftransfer BETWEEN '{start_year}-01-01' AND '{end_year_main}-12-31'
        GROUP BY 1, 2 ORDER BY n DESC
        """
    ).df()
    dist.to_csv(QA_DIR / "transactions_categorytype_recordstatus.csv", index=False)
    log.info(f"categorytype x recordstatus distribution:\n{dist}")

    # --- Raw extract restricted to 2012-2023, with renamed/typed columns ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE trans_raw AS
        SELECT
            transactionid AS transaction_id,
            postcode,
            TRY_CAST(price AS BIGINT) AS transaction_price,
            TRY_CAST(dateoftransfer AS DATE) AS transaction_date,
            propertytype AS property_type_ppd,
            oldnew AS old_new,
            duration,
            categorytype,
            recordstatus,
            LMK_KEY AS lmk_key,
            classt,
            TRY_CAST(priceper AS DOUBLE) AS priceper_ucl,
            TRY_CAST(TOTAL_FLOOR_AREA AS DOUBLE) AS total_floor_area,
            TRY_CAST(UPRN AS BIGINT) AS uprn,
            TRANSACTION_TYPE AS epc_transaction_type,
            TENURE AS tenure_raw,
            CONSTRUCTION_AGE_BAND AS construction_age_band,
            PROPERTY_TYPE AS property_type_epc,
            BUILT_FORM AS built_form,
            TRY_CAST(NUMBER_HABITABLE_ROOMS AS DOUBLE) AS number_habitable_rooms,
            TRY_CAST(LODGEMENT_DATE AS DATE) AS lodgement_date,
            TRY_CAST(INSPECTION_DATE AS DATE) AS inspection_date,
            lad23cd
        FROM read_csv_auto('{TRANS_CSV.as_posix()}')
        WHERE TRY_CAST(dateoftransfer AS DATE) BETWEEN DATE '{start_year}-01-01' AND DATE '{end_year_main}-12-31'
        """
    )
    n_raw = con.sql("SELECT count(*) FROM trans_raw").fetchone()[0]
    log.info(f"Step 0 ({start_year}-01-01 to {end_year_main}-12-31, all property types): {n_raw:,} rows")

    steps = [("00_date_restricted_all_types", n_raw)]

    # --- Step: established homes only ---
    con.execute(f"CREATE OR REPLACE TABLE trans_established AS SELECT * FROM trans_raw WHERE old_new = '{old_new_filter}'")
    n = con.sql("SELECT count(*) FROM trans_established").fetchone()[0]
    steps.append(("01_established_old_new_N", n))
    log.info(f"Step 1 (old_new='{old_new_filter}'): {n:,} rows")

    # --- Step: standard/full-market-value transactions where identifiable ---
    # HMLR PPD Category Type: A = standard price paid entry (full market
    # value, freehold/leasehold, registered), B = additional price paid
    # entry (buy-to-lets, repossessions, etc.) -- keep A where present.
    has_categorytype_a = con.sql("SELECT count(*) FROM trans_established WHERE categorytype = 'A'").fetchone()[0]
    if has_categorytype_a > 0:
        con.execute("CREATE OR REPLACE TABLE trans_standard AS SELECT * FROM trans_established WHERE categorytype = 'A'")
    else:
        log.warning("categorytype == 'A' matched no rows; skipping this filter and flagging in QA.")
        con.execute("CREATE OR REPLACE TABLE trans_standard AS SELECT * FROM trans_established")
    n = con.sql("SELECT count(*) FROM trans_standard").fetchone()[0]
    steps.append(("02_standard_category_A", n))
    log.info(f"Step 2 (categorytype='A' where identifiable): {n:,} rows")

    # --- Step: valid price / floor area ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE trans_valid AS
        SELECT * FROM trans_standard
        WHERE transaction_price >= {min_price}
          AND total_floor_area IS NOT NULL
          AND total_floor_area >= {min_floor_area}
        """
    )
    n = con.sql("SELECT count(*) FROM trans_valid").fetchone()[0]
    steps.append(("03_valid_price_and_floor_area", n))
    log.info(f"Step 3 (valid price & floor area): {n:,} rows")

    n_missing_floor_area = con.sql(
        "SELECT count(*) FROM trans_standard WHERE total_floor_area IS NULL OR total_floor_area < 1"
    ).fetchone()[0]
    n_missing_uprn = con.sql("SELECT count(*) FROM trans_standard WHERE uprn IS NULL").fetchone()[0]
    log.info(f"Floor-area missing/invalid at step 2: {n_missing_floor_area:,} ({n_missing_floor_area/n*100 if n else 0:.2f}%)")
    log.info(f"UPRN missing at step 2: {n_missing_uprn:,}")

    # --- Price-per-sqm construction + discrepancy vs UCL priceper ---
    con.execute(
        """
        CREATE OR REPLACE TABLE trans_ppsqm AS
        SELECT
            *,
            transaction_price / total_floor_area AS calculated_ppsqm,
            (transaction_price / total_floor_area) - priceper_ucl AS ppsqm_discrepancy,
            EXTRACT(year FROM transaction_date) AS transaction_year,
            EXTRACT(month FROM transaction_date) AS transaction_month
        FROM trans_valid
        """
    )
    discrepancy_stats = con.sql(
        """
        SELECT
            count(*) AS n,
            count(priceper_ucl) AS n_with_ucl_priceper,
            avg(ppsqm_discrepancy) AS mean_discrepancy,
            median(ppsqm_discrepancy) AS median_discrepancy,
            stddev(ppsqm_discrepancy) AS sd_discrepancy,
            sum(CASE WHEN abs(ppsqm_discrepancy) < 0.01 THEN 1 ELSE 0 END)::DOUBLE / count(*) AS share_matching_to_1p
        FROM trans_ppsqm
        """
    ).df()
    discrepancy_stats.to_csv(QA_DIR / "ppsqm_discrepancy_vs_ucl.csv", index=False)
    log.info(f"calculated_ppsqm vs UCL priceper discrepancy:\n{discrepancy_stats}")

    # --- nominal / log / real ppsqm ---
    con.execute(
        """
        CREATE OR REPLACE TABLE trans_final AS
        SELECT
            t.*,
            t.calculated_ppsqm AS nominal_ppsqm,
            ln(t.calculated_ppsqm) AS log_ppsqm,
            t.calculated_ppsqm / (c.cpih_index / 100.0) AS real_ppsqm
        FROM trans_ppsqm t
        LEFT JOIN cpih_monthly c
          ON t.transaction_year = c.cpih_year AND t.transaction_month = c.cpih_month
        """
    )
    n_missing_cpih = con.sql("SELECT count(*) FROM trans_final WHERE real_ppsqm IS NULL").fetchone()[0]
    log.info(f"Rows without a matched CPIH month (real_ppsqm null): {n_missing_cpih:,}")

    # --- classt distribution ---
    classt_dist = con.sql(
        "SELECT classt, count(*) AS n FROM trans_final GROUP BY 1 ORDER BY n DESC"
    ).df()
    classt_dist.to_csv(QA_DIR / "transactions_classt_distribution.csv", index=False)

    # --- Outlier trimming: per-year 0.5/99.5 pct on nominal_ppsqm ---
    con.execute(
        f"""
        CREATE OR REPLACE TABLE trans_trim_bounds AS
        SELECT
            transaction_year,
            quantile_cont(nominal_ppsqm, {trim_low/100}) AS lower_bound,
            quantile_cont(nominal_ppsqm, {trim_high/100}) AS upper_bound
        FROM trans_final
        GROUP BY transaction_year
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE trans_trimmed AS
        SELECT t.*
        FROM trans_final t
        JOIN trans_trim_bounds b USING (transaction_year)
        WHERE t.nominal_ppsqm >= b.lower_bound AND t.nominal_ppsqm <= b.upper_bound
        """
    )
    n_untrimmed = con.sql("SELECT count(*) FROM trans_final").fetchone()[0]
    n_trimmed = con.sql("SELECT count(*) FROM trans_trimmed").fetchone()[0]
    steps.append(("04_final_untrimmed", n_untrimmed))
    steps.append(("05_final_trimmed", n_trimmed))
    log.info(f"Outlier trim ({trim_low}/{trim_high} pct per year): {n_untrimmed:,} -> {n_trimmed:,} "
              f"({n_untrimmed - n_trimmed:,} removed)")

    import pandas as pd
    pd.DataFrame(steps, columns=["step", "row_count"]).to_csv(
        QA_DIR / "transactions_cleaning_steps.csv", index=False
    )

    by_year = con.sql(
        """
        SELECT transaction_year, count(*) AS n_transactions,
               median(nominal_ppsqm) AS median_ppsqm
        FROM trans_final GROUP BY 1 ORDER BY 1
        """
    ).df()
    by_year.to_csv(QA_DIR / "transactions_by_year.csv", index=False)

    # --- Write outputs: 2012-2023, trimmed (primary) & untrimmed (robustness) ---
    for label, table in [("trimmed", "trans_trimmed"), ("untrimmed", "trans_final")]:
        con.execute(
            f"""
            COPY (SELECT * FROM {table} WHERE transaction_year BETWEEN {start_year} AND {end_year_main})
            TO '{(INTERIM_DIR / f"transactions_2012_2023_{label}.parquet").as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    log.info(f"Wrote transaction parquet outputs -> {INTERIM_DIR}")
    log.info("Transaction cleaning complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
