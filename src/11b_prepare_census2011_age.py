"""Baseline age structure: Census 2011 KS102EW (age structure), natively at MSOA11 - the
PRIMARY analysis geography, so no crosswalk. England only.

Source file: data/raw/census2011/agestructure_ks102ew.xlsx (Nomis export, all usual residents,
2011 super output area - middle layer, 16 age bands plus the total). The raw file is read, never
modified.

The 16 bands are grouped into five non-overlapping age groups that sum to the total:
    under 16   = 0-4, 5-7, 8-9, 10-14, 15
    16 to 24   = 16-17, 18-19, 20-24
    25 to 44   = 25-29, 30-44
    45 to 64   = 45-59, 60-64
    65 and over = 65-74, 75-84, 85-89, 90+
Each is a share of all usual residents. Because the five shares sum to one, at most four can enter
a regression together. z-scores are England-only and unweighted (higher = larger share).
"""

from __future__ import annotations

import sys

import openpyxl
import pandas as pd

from utils import PROJECT_ROOT, get_logger

log = get_logger("11b_prepare_census2011_age")

RAW = PROJECT_ROOT / "data" / "raw" / "census2011" / "agestructure_ks102ew.xlsx"
DENSITY = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_census2011_density.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa"

BANDS = ["Age 0 to 4", "Age 5 to 7", "Age 8 to 9", "Age 10 to 14", "Age 15", "Age 16 to 17", "Age 18 to 19",
         "Age 20 to 24", "Age 25 to 29", "Age 30 to 44", "Age 45 to 59", "Age 60 to 64", "Age 65 to 74",
         "Age 75 to 84", "Age 85 to 89", "Age 90 and over"]
GROUPS = {
    "under16": ["Age 0 to 4", "Age 5 to 7", "Age 8 to 9", "Age 10 to 14", "Age 15"],
    "16_24": ["Age 16 to 17", "Age 18 to 19", "Age 20 to 24"],
    "25_44": ["Age 25 to 29", "Age 30 to 44"],
    "45_64": ["Age 45 to 59", "Age 60 to 64"],
    "65plus": ["Age 65 to 74", "Age 75 to 84", "Age 85 to 89", "Age 90 and over"],
}


def read_nomis_sheet() -> pd.DataFrame:
    wb = openpyxl.load_workbook(RAW, read_only=True, data_only=True)
    rows = list(wb.worksheets[0].iter_rows(values_only=True))
    header_i = next(i for i, r in enumerate(rows) if r[0] and "super output area" in str(r[0]).lower())
    header = [str(c).strip() if c is not None else "" for c in rows[header_i]]
    log.info(f"Header row {header_i}: {header}")
    body = [r for r in rows[header_i + 1:] if r[0] and ":" in str(r[0])]
    df = pd.DataFrame(body, columns=header)
    first = df.columns[0]
    df["msoa11cd"] = df[first].str.split(":").str[0].str.strip()
    df["msoa11nm_raw"] = df[first].str.split(":", n=1).str[1].str.strip()
    return df.drop(columns=[first])


def main() -> int:
    if not RAW.exists():
        log.error(f"{RAW} not found.")
        return 1
    raw = read_nomis_sheet()
    missing = [c for c in ["All usual residents"] + BANDS if c not in raw.columns]
    if missing:
        log.error(f"STOP: expected columns missing from the workbook: {missing}")
        return 1
    log.info(f"Rows read: {len(raw):,} ({raw.msoa11cd.str.startswith('E02').sum():,} England, "
             f"{raw.msoa11cd.str.startswith('W02').sum():,} Wales)")
    df = raw[raw.msoa11cd.str.startswith("E02")].copy()
    for c in ["All usual residents"] + BANDS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    # --- hard QA stops ---
    if df.msoa11cd.duplicated().any():
        log.error("STOP: duplicate MSOA11 codes in the age structure file.")
        return 1
    if len(df) != 6791:
        log.error(f"STOP: expected 6,791 England MSOA11s, found {len(df):,}.")
        return 1
    band_sum = df[BANDS].sum(axis=1)
    n_bad_sum = int((band_sum != df["All usual residents"]).sum())
    if n_bad_sum:
        log.error(f"STOP: age bands do not sum to the total in {n_bad_sum} MSOA11s.")
        return 1
    if (df["All usual residents"] <= 0).any():
        log.error("STOP: non-positive total population.")
        return 1
    dn = pd.read_parquet(DENSITY)[["msoa11cd", "population_2011"]]
    chk = df.merge(dn, on="msoa11cd", how="outer", indicator=True)
    if (chk["_merge"] != "both").any():
        log.error("STOP: MSOA11 universe differs from the density (QS102EW) table.")
        return 1
    diff = (chk["All usual residents"] - chk["population_2011"]).abs()
    log.info(f"Cross-check against QS102EW population: max absolute difference {diff.max():.0f} persons "
             f"({int((diff > 0).sum())} MSOA11s differ).")
    if diff.max() > 0:
        log.error("STOP: KS102EW total differs from QS102EW population.")
        return 1

    out = pd.DataFrame({"msoa11cd": df.msoa11cd.values, "age_total_population_2011": df["All usual residents"].values})
    for g, cols in GROUPS.items():
        out[f"age_{g}_2011"] = df[cols].sum(axis=1).values
        out[f"age_share_{g}_2011"] = out[f"age_{g}_2011"] / out["age_total_population_2011"]
    share_cols = [f"age_share_{g}_2011" for g in GROUPS]
    if not ((out[share_cols].sum(axis=1) - 1).abs() < 1e-9).all():
        log.error("STOP: grouped age shares do not sum to 1.")
        return 1
    for c in share_cols:
        out[c.replace("_2011", "_z")] = (out[c] - out[c].mean()) / out[c].std(ddof=0)

    summ = out[share_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    summ.to_csv(QA_DIR / "census2011_age_structure_summary.csv")
    log.info(f"Age-group shares (England MSOA11):\n{summ.round(4)}")
    corr = out[share_cols].corr()
    corr.to_csv(QA_DIR / "census2011_age_structure_correlations.csv")
    log.info(f"Correlations among age-group shares:\n{corr.round(3)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "msoa11_census2011_age.parquet", compression="zstd", index=False)
    log.info(f"Wrote {OUT_DIR / 'msoa11_census2011_age.parquet'} ({len(out):,} rows x {out.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
