"""Older accessibility baseline: DfT Accessibility Statistics 2011 (LSOA tables on 2001 LSOA codes) -> MSOA11.

Built for a COMPARISON with the 2014 Journey Time Statistics (11c), not as a replacement: the two series use different
software and methods and DfT says they are not comparable. Output: data/interim/geography_crosswalks/msoa11_accessibility2011.parquet
(not merged into the panel).

Steps: read the eight ACS tables (2001 LSOA codes, numbers with thousands commas) -> expand to 2011 LSOAs with the ONS
LSOA01->LSOA11 best-fit lookup (merged LSOAs: each 2001 LSOA keeps its own weight; split/irregular: the 2001 LSOA's weight
is divided equally among its 2011 pieces, since actual shares are not published) -> population-weighted mean of minutes by MSOA11.
Same weights logic and same 8-service composite as 11c.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import PROJECT_ROOT, get_logger, load_config

log = get_logger("11d_prepare_accessibility2011")
LSOA11_MSOA11 = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa_crosswalks" / "LSOA11_MSOA11_LAD11_EW_distinct.csv"
DENSITY = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_census2011_density.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa" / "accessibility"


def fail(msg: str) -> int:
    log.error(f"STOP: {msg}")
    return 1


def read_acs(path) -> pd.DataFrame:
    import csv
    rows = list(csv.reader(open(path, encoding="latin1")))
    hi = next(i for i, r in enumerate(rows) if r and r[0].strip().lower().startswith("lsoa_code"))
    header = [c.strip() for c in rows[hi]]
    body = [r for r in rows[hi + 1:] if r and r[0].startswith("E01")]
    return pd.DataFrame(body, columns=header)


def col(df: pd.DataFrame, prefix: str) -> str:
    hits = [c for c in df.columns if c.startswith(prefix)]
    if len(hits) != 1:
        raise KeyError(f"expected exactly one column starting '{prefix}', found {hits}")
    return hits[0]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="raise")


def main() -> int:
    cfg = load_config()["accessibility_2011"]
    d = PROJECT_ROOT / cfg["data_dir"]
    universe = set(pd.read_parquet(DENSITY)["msoa11cd"])
    lk = pd.read_csv(PROJECT_ROOT / cfg["lsoa01_lsoa11_lookup"], dtype=str)
    lk = lk[lk.LSOA01CD.str.startswith("E01")][["LSOA01CD", "LSOA11CD", "CHGIND"]]
    l11 = pd.read_csv(LSOA11_MSOA11, dtype=str).rename(columns={"LSOA11CD": "LSOA11CD", "MSOA11CD": "msoa11cd"})[["LSOA11CD", "msoa11cd"]]
    lk = lk.merge(l11, on="LSOA11CD", how="left", validate="m:1")
    if lk.msoa11cd.isna().any():
        return fail(f"{int(lk.msoa11cd.isna().sum())} lookup rows have no MSOA11.")
    n_pieces = lk.groupby("LSOA01CD")["LSOA11CD"].transform("nunique")
    lk["piece_share"] = 1.0 / n_pieces                                    # equal split of a 2001 LSOA among its 2011 pieces
    log.info(f"Lookup: {lk.LSOA01CD.nunique():,} LSOA01s -> {lk.LSOA11CD.nunique():,} LSOA11s; CHGIND (rows): {lk.CHGIND.value_counts().to_dict()}")

    wide = None
    for svc, sc in cfg["services"].items():
        t = read_acs(d / sc["file"])
        pc, ptc, cc = col(t, sc["pop_prefix"]), col(t, sc["pt_prefix"]), col(t, sc["car_prefix"])
        log.info(f"{sc['file']} [{svc}]: {len(t):,} LSOA01 rows; pop='{pc}', pt='{ptc}', car='{cc}'")
        t = t.rename(columns={"LSOA_code": "LSOA01CD"}) if "LSOA_code" in t.columns else t.rename(columns={t.columns[0]: "LSOA01CD"})
        if t.LSOA01CD.duplicated().any():
            return fail(f"{sc['file']}: duplicate LSOA01 codes.")
        part = pd.DataFrame({"LSOA01CD": t.LSOA01CD, f"{svc}_pop": num(t[pc]), f"{svc}_pt": num(t[ptc]), f"{svc}_car": num(t[cc])})
        if part.iloc[:, 1:].isna().any().any() or (part.iloc[:, 1:] < 0).any().any():
            return fail(f"{sc['file']}: missing or negative values.")
        wide = part if wide is None else wide.merge(part, on="LSOA01CD", how="outer", validate="1:1")
    if len(wide) != cfg["expected_n_lsoa01"] or wide.isna().any().any():
        return fail(f"LSOA01 sets differ across tables or count != {cfg['expected_n_lsoa01']} (found {len(wide)}).")
    if set(wide.LSOA01CD) != set(lk.LSOA01CD):
        return fail("LSOA01 codes in the tables differ from the lookup.")
    log.info("Assert OK: 32,482 LSOA01s in all eight tables, all in the lookup, no missing values.")

    x = lk.merge(wide, on="LSOA01CD", how="left", validate="m:1")
    tot_w = x["employment_pop"] * x["piece_share"]
    affected = x.CHGIND.isin(["S", "X"])
    log.info(f"Share of employment weight in split/irregular LSOAs (equal-split assumption): {tot_w[affected].sum() / tot_w.sum():.3%}")

    out = pd.DataFrame(index=sorted(universe)); out.index.name = "msoa11cd"
    for svc in cfg["services"]:
        w = x[f"{svc}_pop"] * x["piece_share"]
        if (w.groupby(x.msoa11cd).sum() <= 0).any():
            return fail(f"zero total weight for {svc} in some MSOA11.")
        for m in ("pt", "car"):
            out[f"access2011_{svc}_{m}_min"] = ((x[f"{svc}_{m}"] * w).groupby(x.msoa11cd).sum() / w.groupby(x.msoa11cd).sum())
    if set(out.index) != set(x.msoa11cd.unique()) or out.isna().any().any():
        return fail("MSOA11 coverage differs from the universe or missing values.")
    for m in ("pt", "car"):
        out[f"access2011_keyservices_{m}_min"] = out[[f"access2011_{s}_{m}_min" for s in cfg["services"]]].mean(axis=1)
        out[f"access2011_keyservices7_{m}_min"] = out[[f"access2011_{s}_{m}_min" for s in cfg["services"] if s != "town_centre"]].mean(axis=1)
    log.info(f"England means (MSOA11): key-services (8) pt {out['access2011_keyservices_pt_min'].mean():.1f}, car {out['access2011_keyservices_car_min'].mean():.1f} min; "
             f"7 services (DfT 2011 headline excludes town centres; DfT published about 14 pt / 6 car, working-age weighted): pt {out['access2011_keyservices7_pt_min'].mean():.1f}, car {out['access2011_keyservices7_car_min'].mean():.1f}")
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(OUT_DIR / "msoa11_accessibility2011.parquet", compression="zstd", index=False)
    log.info(f"Wrote msoa11_accessibility2011.parquet ({len(out):,} rows x {out.shape[1] + 1} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
