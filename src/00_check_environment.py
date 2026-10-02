"""Verify the runtime environment before any data processing starts:
package versions, directory skeleton, and available disk space against the
known size of the source data (two ~23GB CSVs inside the UCL bundles)."""

from __future__ import annotations

import shutil
import sys
from importlib.metadata import PackageNotFoundError, version

from utils import PROJECT_ROOT, get_logger

log = get_logger("00_check_environment")

REQUIRED_PACKAGES = [
    "duckdb",
    "polars",
    "pyarrow",
    "pandas",
    "openpyxl",
    "xlsxwriter",
    "xlrd",
    "yaml",
    "requests",
    "scipy",
    "sklearn",
    "matplotlib",
    "pytest",
]

# Package name on PyPI vs. importable module name differ for a couple of these.
DIST_NAME_OVERRIDES = {"yaml": "pyyaml", "sklearn": "scikit-learn"}

REQUIRED_DIRS = [
    "config",
    "data/raw/ucl/linked_transactions",
    "data/raw/ucl/epc_source",
    "data/raw/ucl/ppd_source",
    "data/raw/geography/nsul",
    "data/raw/geography/onsud",
    "data/raw/geography/nspl",
    "data/raw/geography/msoa_crosswalks",
    "data/raw/census/tenure_ts054",
    "data/raw/income/ons_small_area_income",
    "data/raw/macro/cpih",
    "data/interim/transactions_clean",
    "data/interim/epc_clean",
    "data/interim/newbuilds",
    "data/interim/geography_crosswalks",
    "data/processed",
    "outputs/tables",
    "outputs/figures",
    "outputs/excel",
    "outputs/qa",
    "outputs/logs",
]

# Known raw-source footprint: two ~23GB CSVs extracted from the UCL bundles,
# plus their zips already on disk, plus interim/processed working copies.
MIN_FREE_GB_REQUIRED = 150


def check_packages() -> bool:
    ok = True
    for mod_name in REQUIRED_PACKAGES:
        dist_name = DIST_NAME_OVERRIDES.get(mod_name, mod_name)
        try:
            v = version(dist_name)
            log.info(f"OK  {mod_name:12s} {v}")
        except PackageNotFoundError:
            try:
                __import__(mod_name)
                log.info(f"OK  {mod_name:12s} (importable, version unknown)")
            except ImportError:
                log.error(f"MISSING {mod_name}")
                ok = False
    return ok


def check_directories() -> None:
    for d in REQUIRED_DIRS:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        gitkeep = path / ".gitkeep"
        if not any(path.iterdir()):
            gitkeep.touch()
    log.info(f"Verified/created {len(REQUIRED_DIRS)} directories under {PROJECT_ROOT}")


def check_disk_space() -> bool:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    log.info(f"Free disk space at {PROJECT_ROOT}: {free_gb:,.1f} GB")
    if free_gb < MIN_FREE_GB_REQUIRED:
        log.error(
            f"Free space {free_gb:.1f}GB is below the {MIN_FREE_GB_REQUIRED}GB "
            "recommended for extracting/processing the two ~23GB UCL source CSVs."
        )
        return False
    return True


def main() -> int:
    log.info(f"Python: {sys.version}")
    log.info(f"Project root: {PROJECT_ROOT}")

    packages_ok = check_packages()
    check_directories()
    disk_ok = check_disk_space()

    if not packages_ok:
        log.error("Missing required packages - run: pip install -r requirements.txt")
        return 1
    if not disk_ok:
        log.error("Insufficient disk space - see message above.")
        return 1

    log.info("Environment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
