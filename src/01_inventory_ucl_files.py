"""Locate the UCL ReShare 857911 bundle zips, extract the two core CSVs
(linked transactions, EPC source) plus the bundled NSPL and LRPPD files,
and write out full schema inspections before any cleaning happens.

Per the brief: print all column names, save schemas to outputs/qa/, and
identify the exact datatype of every required variable, for both the linked
transactions and the EPC source.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger

log = get_logger("01_inventory_ucl_files")

# The bundle zips may sit either inside working/data/raw or one level up in
# the parent dissertation/ project folder (where the user placed them).
CANDIDATE_RAW_DIRS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT.parent / "data" / "raw",
]

BUNDLE_1 = "857911_bundle_1.zip"
BUNDLE_2 = "857911_bundle_2.zip"

EPC_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "ucl" / "epc_source"
TRANS_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "ucl" / "linked_transactions"
PPD_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "ucl" / "ppd_source"
NSPL_TARGET_DIR = PROJECT_ROOT / "data" / "raw" / "geography" / "nspl"

QA_DIR = PROJECT_ROOT / "outputs" / "qa"


def find_bundle(name: str) -> Path:
    for d in CANDIDATE_RAW_DIRS:
        candidate = d / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {name} in any of {CANDIDATE_RAW_DIRS}. "
        "Place the UCL ReShare 857911 bundle zip(s) in one of these locations."
    )


def extract_member(outer_zip: Path, member_name: str, extract_to: Path) -> Path:
    """Extract a single member from a zip into extract_to, return its path."""
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(outer_zip) as zf:
        names = zf.namelist()
        matches = [n for n in names if n.endswith(member_name)]
        if not matches:
            raise KeyError(f"{member_name} not found in {outer_zip} (has: {names})")
        member = matches[0]
    log.info(f"Extracting {member} from {outer_zip.name} -> {extract_to} (this can take a while for multi-GB members)")
    # Use system unzip for speed on multi-GB members rather than Python's zipfile.
    subprocess.run(
        ["unzip", "-o", str(outer_zip), member, "-d", str(extract_to)],
        check=True,
    )
    extracted = extract_to / member
    return extracted


def extract_inner_csv(inner_zip: Path, final_dir: Path) -> Path:
    """inner_zip is itself a zip containing exactly one CSV; extract that CSV
    directly into final_dir and remove the intermediate inner zip."""
    final_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(inner_zip) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one CSV in {inner_zip}, found {names}")
        csv_name = names[0]
    log.info(f"Extracting {csv_name} from {inner_zip.name} -> {final_dir} (multi-GB, please wait)")
    subprocess.run(["unzip", "-o", str(inner_zip), csv_name, "-d", str(final_dir)], check=True)
    final_csv = final_dir / csv_name
    inner_zip.unlink()  # clean up the intermediate zip, keep only the final CSV
    return final_csv


def write_schema(con, csv_path: Path, out_name: str) -> None:
    log.info(f"Inferring schema for {csv_path.name} ...")
    rel = con.sql(f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}', sample_size=200000)")
    schema_df = con.sql(f"DESCRIBE SELECT * FROM read_csv_auto('{csv_path.as_posix()}', sample_size=200000)").df()
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QA_DIR / out_name
    schema_df.to_csv(out_path, index=False)
    log.info(f"Wrote schema ({len(schema_df)} columns) -> {out_path}")
    print(f"\n=== {csv_path.name}: {len(schema_df)} columns ===")
    for _, row in schema_df.iterrows():
        print(f"  {row['column_name']:35s} {row['column_type']}")

    row_count = con.sql(f"SELECT count(*) FROM read_csv_auto('{csv_path.as_posix()}')").fetchone()[0]
    log.info(f"{csv_path.name}: {row_count:,} rows")
    with open(QA_DIR / f"{out_name.replace('.csv','')}_rowcount.txt", "w") as f:
        f.write(f"{csv_path.name}: {row_count}\n")


def main() -> int:
    bundle1 = find_bundle(BUNDLE_1)
    bundle2 = find_bundle(BUNDLE_2)
    log.info(f"bundle_1 (EPC source): {bundle1}")
    log.info(f"bundle_2 (linked transactions): {bundle2}")

    # --- EPC source (bundle 1) ---
    epc_csv = EPC_TARGET_DIR / "epc_2024.csv"
    if not epc_csv.exists():
        epc_inner_zip = extract_member(bundle1, "epc_2024.zip", EPC_TARGET_DIR)
        epc_csv = extract_inner_csv(epc_inner_zip, EPC_TARGET_DIR)
    else:
        log.info(f"{epc_csv} already extracted, skipping.")

    # --- Linked transactions (bundle 2) ---
    trans_csv = TRANS_TARGET_DIR / "tranall_link_26122024.csv"
    if not trans_csv.exists():
        trans_inner_zip = extract_member(bundle2, "tranall_link_26122024.zip", TRANS_TARGET_DIR)
        trans_csv = extract_inner_csv(trans_inner_zip, TRANS_TARGET_DIR)
    else:
        log.info(f"{trans_csv} already extracted, skipping.")

    # --- Bundled NSPL (postcode -> MSOA/LAD), vintage-matched to the transactions ---
    nspl_csv_glob = list(NSPL_TARGET_DIR.glob("*.csv"))
    if not nspl_csv_glob:
        nspl_inner_zip = extract_member(bundle2, "NSPL21_NOV_2024_UK.zip", NSPL_TARGET_DIR)
        with zipfile.ZipFile(nspl_inner_zip) as zf:
            zf.extractall(NSPL_TARGET_DIR)
        nspl_inner_zip.unlink()
    else:
        log.info(f"NSPL already extracted: {nspl_csv_glob}")

    # --- Original Land Registry PPD (kept per file structure; not used to
    # derive outcomes, see brief) ---
    ppd_csv_glob = list(PPD_TARGET_DIR.glob("*.csv"))
    if not ppd_csv_glob:
        ppd_inner_zip = extract_member(bundle2, "LRPPD_26122024.zip", PPD_TARGET_DIR)
        extract_inner_csv(ppd_inner_zip, PPD_TARGET_DIR)
    else:
        log.info(f"LRPPD already extracted: {ppd_csv_glob}")

    # --- Schema inspection ---
    con = get_duckdb_connection()
    write_schema(con, trans_csv, "schema_tranall_link.csv")
    write_schema(con, epc_csv, "schema_epc_source.csv")

    log.info("Inventory + schema inspection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
